"""Core data sharding infrastructure for federated learning."""

import itertools
from datasets import load_dataset
from torch.utils.data import IterableDataset


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
    "medalpaca/medical_meadow_wikidoc": 67704,
}


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
    """Unified dataset that handles both indexed and streaming datasets with sharding."""

    def __init__(self, dataset, shard_id, num_shards, processed_batches, batch_size,
                 dataset_name=None, streaming=False):
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.processed_batches = processed_batches
        self.batch_size = batch_size
        self.streaming = streaming

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
        """Iterator for indexed datasets (CIFAR-10)."""
        current_batch = self.batch_in_epoch

        while True:
            sample_start = self.start_idx + (current_batch * self.batch_size)
            sample_end = min(sample_start + self.batch_size, self.end_idx)

            for idx in range(sample_start, sample_end):
                yield self.dataset[idx]

            current_batch = (current_batch + 1) % self.batches_per_shard_epoch

    def _iter_streaming(self):
        """Iterator for streaming datasets (ImageNet-1k, UltraChat)."""
        sharded_dataset = self.dataset.shard(num_shards=self.num_shards, index=self.shard_id)
        samples_to_skip = self.batch_in_epoch * self.batch_size

        while True:
            iterator = iter(sharded_dataset)

            if samples_to_skip > 0:
                iterator = itertools.islice(iterator, samples_to_skip, None)
                samples_to_skip = 0

            sample_count = 0
            for sample in iterator:
                yield sample
                sample_count += 1

                if sample_count >= self.shard_size:
                    break

    def __len__(self):
        return self.shard_size
