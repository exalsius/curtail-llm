"""Data loading utilities for federated learning with queue-based data management."""

import itertools
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

# Known dataset sizes for streaming datasets
DATASET_SIZES = {
    "uoft-cs/cifar10": 50000,
    "imagenet-1k": 1281167,
}


class ShardManager:
    """Manages multiple data shards, tracking progress through the dataset.

    Note: Future improvement - assign workers expected to progress the most to shards
    furthest behind. Currently we don't track expected progress per worker.
    """

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        # Track number of batches processed for each shard
        self.shard_states = [0 for _ in range(num_shards)]

    def assign_workers(self, node_ids: list[int]):
        """Assign workers to shards with least progress.

        Returns:
            Dict mapping node_id to (shard_id, processed_batches) tuples
        """
        # Fail if more workers than shards
        if len(node_ids) > self.num_shards:
            raise ValueError(f"Cannot assign {len(node_ids)} workers to {self.num_shards} shards.")

        # Sort shards by progress (batches processed) to find those with least progress
        shard_progress = [
            (shard_id, processed_batches)
            for shard_id, processed_batches in enumerate(self.shard_states)
        ]
        # Sort by processed_batches - shards with lower values have less progress
        shard_progress.sort(key=lambda x: x[1])

        # Assign workers to the shards with least progress
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

    Yields samples continuously from the shard, wrapping around when reaching the end.
    Supports both:
    - Indexed datasets (CIFAR-10): Uses direct indexing
    - Streaming datasets (ImageNet-1k): Uses HuggingFace's native .shard()

    The iterator yields samples infinitely - the consumer (train function) controls
    how many batches to process.
    """

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
            # For streaming, use known dataset sizes
            if dataset_name is None:
                raise ValueError("dataset_name required for streaming datasets")
            total_size = DATASET_SIZES[dataset_name]
            self.shard_size = total_size // num_shards
        else:
            # For indexed datasets, calculate from dataset length
            total_size = len(dataset)
            shard_size = total_size // num_shards
            self.start_idx = shard_id * shard_size
            self.end_idx = self.start_idx + shard_size
            self.shard_size = self.end_idx - self.start_idx

        # Calculate batches per epoch for this shard
        self.batches_per_shard_epoch = self.shard_size // batch_size  # Complete batches only

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
            # Calculate sample range for current batch
            sample_start = self.start_idx + (current_batch * self.batch_size)
            sample_end = min(sample_start + self.batch_size, self.end_idx)

            # Yield all samples in this batch
            for idx in range(sample_start, sample_end):
                yield self.dataset[idx]

            # Move to next batch, wrapping around if needed
            current_batch = (current_batch + 1) % self.batches_per_shard_epoch

    def _iter_streaming(self):
        """Iterator for streaming datasets (ImageNet-1k)."""
        # Use HuggingFace's native sharding
        sharded_dataset = self.dataset.shard(num_shards=self.num_shards, index=self.shard_id)

        # Calculate samples to skip to resume from current position
        samples_to_skip = self.batch_in_epoch * self.batch_size

        while True:
            # Create iterator for this epoch
            iterator = iter(sharded_dataset)

            # Skip to current position if resuming
            if samples_to_skip > 0:
                iterator = itertools.islice(iterator, samples_to_skip, None)
                samples_to_skip = 0  # Only skip on first epoch

            # Yield samples from this epoch
            sample_count = 0
            for sample in iterator:
                yield sample
                sample_count += 1

                # Stop at shard boundary (complete batches only)
                if sample_count >= self.shard_size:
                    break

    def __len__(self):
        return self.shard_size


def get_transform_fn(dataset_name: str):
    """Get the appropriate transform function and image key for a dataset.

    Returns:
        tuple: (transform_fn, image_key)
    """
    if "imagenet" in dataset_name.lower():
        transforms = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image_key = "image"

        def transform_fn(batch):
            batch[image_key] = [transforms(img.convert("RGB")) for img in batch[image_key]]
            return batch

        return transform_fn, image_key
    else:
        # CIFAR-10 and similar
        transforms = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image_key = "img"

        def transform_fn(batch):
            batch[image_key] = [transforms(img) for img in batch[image_key]]
            return batch

        return transform_fn, image_key


def get_train_loader(
    dataset_name: str,
    shard_id: int,
    num_shards: int,
    processed_batches: int,
    batch_size: int,
    streaming: bool = False,
    **kwargs  # Ignore extra kwargs
):
    """Load a shard of the training dataset, resuming from a specific position.

    The loader provides batches continuously from the shard, wrapping around when
    reaching the end. The train function controls how many batches to consume.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10", "imagenet-1k")
        shard_id: Which shard this loader belongs to (0 to num_shards-1)
        num_shards: Total number of shards (can be more than workers)
        processed_batches: Number of batches already processed on this shard
        batch_size: Batch size for training
        streaming: Whether to use streaming mode for large datasets

    Returns:
        DataLoader that yields batches infinitely from the assigned shard

    Note:
        - The current epoch is automatically inferred from processed_batches
        - Workers continue on the same shard across epochs for better data locality
        - The iterator wraps around the shard infinitely - caller controls batch count
        - For streaming datasets (ImageNet-1k), uses native HuggingFace sharding
    """
    # Determine if dataset should use streaming mode
    use_streaming = streaming or (dataset_name in DATASET_SIZES and dataset_name != "uoft-cs/cifar10")

    # Load the training dataset
    dataset = load_dataset(dataset_name, split="train", streaming=use_streaming)

    # Get transform function for this dataset
    transform_fn, _ = get_transform_fn(dataset_name)

    # Apply transforms for non-streaming datasets
    if not use_streaming:
        dataset = dataset.with_format("torch").with_transform(transform_fn)

    # Create unified sharded dataset
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        dataset_name=dataset_name if use_streaming else None,
        streaming=use_streaming,
    )

    return DataLoader(sharded_dataset, batch_size=batch_size, shuffle=False)


def get_test_loader(dataset_name: str, batch_size: int):
    """Load the full test dataset for evaluation.

    Args:
        dataset_name: HuggingFace dataset name
        batch_size: Batch size for evaluation

    Returns:
        DataLoader for the test set
    """
    # Select appropriate split (ImageNet uses 'validation' instead of 'test')
    split = "validation" if "imagenet" in dataset_name.lower() else "test"

    # Get transform function for this dataset
    transform_fn, _ = get_transform_fn(dataset_name)

    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.with_format("torch").with_transform(transform_fn)
    return DataLoader(dataset, batch_size=batch_size)
