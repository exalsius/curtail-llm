"""Core data sharding infrastructure for federated learning."""

import itertools
from torch.utils.data import IterableDataset


# Medical Meadow dataset sizes (samples)
# Based on: https://github.com/kbressem/medAlpaca
MEDICAL_DATASETS = {
    # Curated combined dataset (medAlpaca-inspired, excluding USMLE for testing)
    "medalpaca/medical_meadow_curated": 226535,  # ~227K samples (excludes USMLE test set)

    # Core Medical Meadow datasets
    "medalpaca/medical_meadow_wikidoc": 67704,
    "medalpaca/medical_meadow_medical_flashcards": 33955,
    "medalpaca/medical_meadow_medqa": 10178,
    "medalpaca/medical_meadow_cord19": 13778,
    "medalpaca/medical_meadow_mmmlu": 3787,
    "medalpaca/medical_meadow_pubmed": 200000,

    # Stack Exchange medical topics
    "medalpaca/medical_meadow_health_care_magic": 112165,
    "medalpaca/medical_meadow_stack_exchange_biology": 27326,
    "medalpaca/medical_meadow_stack_exchange_fitness": 9326,

    # Specialized medical datasets
    "medalpaca/medical_meadow_wikidoc_patient_information": 5942,
    "medalpaca/medical_meadow_mediqa": 2208,

    # USMLE (Medical licensing exams) - reserved for testing
    "medalpaca/medical_meadow_usmle_self_assessment": 2903,
}


# Known dataset sizes for streaming datasets
DATASET_SIZES = {
    # Vision datasets
    "uoft-cs/cifar10": 50000,
    "imagenet-1k": 1281167,
    # LLM datasets
    "HuggingFaceH4/ultrachat_200k": 207865,
    "OpenAssistant/oasst2": 161443,
    "tatsu-lab/alpaca": 52002,
    "timdettmers/openassistant-guanaco": 9846,
}

# Add all Medical Meadow datasets
DATASET_SIZES.update(MEDICAL_DATASETS)


class ShardManager:
    """Manages multiple data shards, tracking progress through the dataset."""

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.shard_states = [0 for _ in range(num_shards)]

    def assign_workers(self, node_ids: list[int]):
        """Assign workers to shards with least progress."""
        if len(node_ids) > self.num_shards:
            raise ValueError(f"Cannot assign {len(node_ids)} workers to {self.num_shards} shards.")

        shard_progress = [(shard_id, batches) for shard_id, batches in enumerate(self.shard_states)]
        shard_progress.sort(key=lambda x: x[1])

        assignments = {}
        for i, node_id in enumerate(node_ids):
            shard_id, processed_batches = shard_progress[i]
            assignments[node_id] = (shard_id, processed_batches)

        return assignments

    def add(self, shard_id: int, processed_batches: int):
        """Update the state of a shard after training."""
        self.shard_states[shard_id] += processed_batches

    def __repr__(self):
        return f"ShardManager(num_shards={self.num_shards}, states={self.shard_states})"


class ShardedDataset(IterableDataset):
    """Unified dataset that handles both indexed and streaming datasets with sharding.

    For streaming datasets, applies a pre-shuffle before sharding and varies the
    shuffle seed effectively per shard-epoch to provide different samples across
    epochs while maintaining even coverage over time.
    """

    def __init__(self, dataset, shard_id, num_shards, processed_batches, batch_size,
                 dataset_name=None, streaming=False, shuffle_seed: int = 0, shuffle_buffer_size: int | None = None):
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.processed_batches = processed_batches
        self.batch_size = batch_size
        self.streaming = streaming
        self.dataset_name = dataset_name
        self.shuffle_seed = shuffle_seed

        # Default shuffle buffer sizing for streaming datasets (trade off RAM vs. decorrelation)
        if shuffle_buffer_size is None and streaming:
            # Heuristic defaults: larger buffers for very large datasets
            if dataset_name and "imagenet" in dataset_name.lower():
                shuffle_buffer_size = 100_000
            else:
                shuffle_buffer_size = 50_000
        self.shuffle_buffer_size = shuffle_buffer_size

        # Calculate shard size
        if streaming:
            if dataset_name is None:
                raise ValueError("dataset_name required for streaming datasets")
            total_size = DATASET_SIZES[dataset_name]
            self.shard_size = total_size // num_shards
        else:
            total_size = len(dataset)
            shard_size = total_size // num_shards
            self.start_idx = shard_id * shard_size
            self.end_idx = self.start_idx + shard_size
            self.shard_size = self.end_idx - self.start_idx

        # Calculate batches per epoch
        self.batches_per_shard_epoch = self.shard_size // batch_size

        # Calculate current position
        self.current_epoch = processed_batches // self.batches_per_shard_epoch
        self.batch_in_epoch = processed_batches % self.batches_per_shard_epoch

    def __iter__(self):
        """Yield samples infinitely, wrapping around the shard."""
        if self.streaming:
            yield from self._iter_streaming()
        else:
            yield from self._iter_indexed()

    def _iter_indexed(self):
        """Iterator for map-style datasets (e.g., CIFAR-10, UltraChat).

        Performs pre-shuffle before sharding for each shard-epoch using a
        deterministic seed schedule. Resumes within-epoch by skipping
        already-consumed samples, then reshuffles and continues next epoch.
        """
        current_epoch = self.current_epoch
        samples_to_skip = self.batch_in_epoch * self.batch_size

        while True:
            # Pre-shuffle globally, then shard interleaved for balance
            ds = self.dataset.shuffle(seed=self.shuffle_seed + current_epoch)
            # Interleaved sharding for balance across shards
            ds = ds.shard(num_shards=self.num_shards, index=self.shard_id, contiguous=False)

            # Determine how many samples to serve this epoch for this shard
            epoch_shard_len = len(ds)
            take_count = min(self.shard_size, epoch_shard_len)

            # Resume inside the epoch if needed
            start = min(samples_to_skip, take_count)
            if start:
                samples_to_skip = 0

            for i in range(start, take_count):
                yield ds[i]

            # Next shard-epoch
            current_epoch += 1

    def _iter_streaming(self):
        """Iterator for streaming datasets (ImageNet-1k, UltraChat).

        Applies pre-shuffle then sharding. On each shard-epoch rollover, varies the
        effective shuffle seed (via epoch or seed offset) to reshuffle before
        re-iterating, producing different samples per epoch while keeping shard
        sizes and load balanced.
        """
        if self.dataset_name is None:
            raise ValueError("dataset_name required for streaming datasets")

        samples_to_skip = self.batch_in_epoch * self.batch_size
        current_epoch = self.current_epoch

        while True:
            # Pre-shuffle, then shard. Use a stable base seed and vary with epoch
            # to avoid accumulating transforms and to ensure reproducibility.
            ds = self.dataset.shuffle(seed=self.shuffle_seed + current_epoch,
                                      buffer_size=self.shuffle_buffer_size)
            ds = ds.shard(num_shards=self.num_shards, index=self.shard_id)

            # Inform HF IterableDataset of the epoch for deterministic reshuffle/shard
            try:
                ds.set_epoch(current_epoch)
            except AttributeError:
                # Older datasets versions may not support set_epoch; safe to ignore
                pass

            iterator = iter(ds)

            if samples_to_skip > 0:
                iterator = itertools.islice(iterator, samples_to_skip, None)
                samples_to_skip = 0

            sample_count = 0
            for sample in iterator:
                yield sample
                sample_count += 1
                if sample_count >= self.shard_size:
                    break

            # Advance to next shard-epoch and reshuffle before next pass
            current_epoch += 1

    def __len__(self):
        return self.shard_size
