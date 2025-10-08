"""Data loading utilities for federated learning with queue-based data management."""

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import Compose, Normalize, ToTensor


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

    def update(self, shard_id: int, processed_batches: int):
        """Update the state of a shard after training."""
        self.shard_states[shard_id] = processed_batches

    def __repr__(self):
        return f"ShardManager(num_shards={self.num_shards}, states={self.shard_states})"


class ShardedDataset(IterableDataset):
    """Dataset that shards data and can resume from a specific batch index.

    Yields batches continuously from the shard, wrapping around when reaching the end.
    The current epoch is inferred from processed_batches for progress tracking.
    The iterator yields samples infinitely - the consumer (train function) controls
    how many batches to process.

    Note: For streaming datasets, you must provide shard_size explicitly since
    len(dataset) is not available. Currently only supports non-streaming datasets.
    """

    def __init__(self, dataset, shard_id, num_shards, processed_batches, batch_size, shard_size=None):
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.processed_batches = processed_batches
        self.batch_size = batch_size

        # Calculate this shard's slice of the dataset
        if shard_size is None:
            total_size = len(dataset)
            shard_size = total_size // num_shards
        self.start_idx = shard_id * shard_size
        self.end_idx = self.start_idx + shard_size if shard_id < num_shards - 1 else self.start_idx + shard_size
        self.shard_size = self.end_idx - self.start_idx

        # Calculate batches per epoch for this shard
        self.batches_per_shard_epoch = (self.shard_size + batch_size - 1) // batch_size  # ceil division

        # Calculate current epoch and position within epoch (for tracking)
        self.current_epoch = processed_batches // self.batches_per_shard_epoch
        self.batch_in_epoch = processed_batches % self.batches_per_shard_epoch

    def __iter__(self):
        # Start from the current position within the shard
        current_batch = self.batch_in_epoch

        # Yield batches infinitely, wrapping around the shard
        while True:
            # Calculate sample range for current batch
            sample_start = self.start_idx + (current_batch * self.batch_size)
            sample_end = min(sample_start + self.batch_size, self.end_idx)

            # Yield all samples in this batch
            for idx in range(sample_start, sample_end):
                yield self.dataset[idx]

            # Move to next batch, wrapping around if needed
            current_batch = (current_batch + 1) % self.batches_per_shard_epoch

    def __len__(self):
        return self.end_idx - self.start_idx


def apply_cifar10_transforms(batch):
    """Apply transforms to CIFAR-10 images."""
    transforms = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def get_train_loader(
    dataset_name: str,
    shard_id: int,
    num_shards: int,
    processed_batches: int,
    batch_size: int,
    streaming: bool = False,
    shard_size: int = None,
    **kwargs  # Ignore extra kwargs
):
    """Load a shard of the training dataset, resuming from a specific position.

    The loader provides batches continuously from the shard, wrapping around when
    reaching the end. The train function controls how many batches to consume.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10")
        shard_id: Which shard this loader belongs to (0 to num_shards-1)
        num_shards: Total number of shards (can be more than workers)
        processed_batches: Number of batches already processed on this shard
        batch_size: Batch size for training
        streaming: Whether to use streaming mode for large datasets
        shard_size: Size of each shard (required for streaming datasets)

    Returns:
        DataLoader that yields batches infinitely from the assigned shard

    Note:
        - The current epoch is automatically inferred from processed_batches
        - Workers continue on the same shard across epochs for better data locality
        - The iterator wraps around the shard infinitely - caller controls batch count
        - For streaming datasets, shard_size must be provided explicitly
    """
    # Load the full training dataset
    dataset = load_dataset(dataset_name, split="train", streaming=streaming)

    # Apply transforms
    if not streaming:
        dataset = dataset.with_format("torch").with_transform(apply_cifar10_transforms)
    else:
        # For streaming datasets, transforms are applied differently
        dataset = dataset.with_format("torch")
        # Note: May need to handle transforms in the dataset iterator for streaming

    # Create sharded dataset that can resume from specific batch
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        shard_size=shard_size,
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
    dataset = load_dataset(dataset_name, split="test")
    dataset = dataset.with_format("torch").with_transform(apply_cifar10_transforms)
    return DataLoader(dataset, batch_size=batch_size)
