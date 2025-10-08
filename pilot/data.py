"""Data loading utilities for federated learning with queue-based data management."""

from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import Compose, Normalize, ToTensor


class QueueManager:
    """Manages multiple data queues, tracking progress through the dataset."""

    def __init__(self, num_queues: int):
        self.num_queues = num_queues
        # Track (batch_idx, epoch) for each queue
        self.queue_states = [(0, 0) for _ in range(num_queues)]

    def assign_workers(self, node_ids: list[int]):
        """Assign workers to queues with least progress.

        Returns:
            List of (queue_id, epoch, batch_idx) tuples, one per worker
        """
        # Fail if more workers than queues
        if len(node_ids) > self.num_queues:
            raise ValueError(f"Cannot assign {len(node_ids)} workers to {self.num_queues} queues.")

        # Sort queues by progress (epoch, batch_idx) to find those with least progress
        queue_progress = [
            (queue_id, epoch, batch_idx)
            for queue_id, (batch_idx, epoch) in enumerate(self.queue_states)
        ]
        # Sort by (epoch, batch_idx) - queues with lower values have less progress
        queue_progress.sort(key=lambda x: (x[1], x[2]))

        # Assign workers to the queues with least progress
        assignments = {}
        for i, node_id in enumerate(node_ids):
            queue_id, epoch, batch_idx = queue_progress[i]
            assignments[node_id] = (queue_id, epoch, batch_idx)

        return assignments

    def update(self, queue_id: int, batch_idx: int, epoch: int):
        """Update the state of a queue after training."""
        self.queue_states[queue_id] = (batch_idx, epoch)

    def __repr__(self):
        return f"QueueManager(num_queues={self.num_queues}, states={self.queue_states})"


class ShardedDataset(IterableDataset):
    """Dataset that shards data and can resume from a specific batch index."""

    def __init__(self, dataset, shard_id, num_shards, start_batch_idx, epoch, batch_size):
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.start_batch_idx = start_batch_idx
        self.epoch = epoch
        self.batch_size = batch_size

        # Calculate this shard's slice of the dataset
        total_size = len(dataset)
        shard_size = total_size // num_shards
        self.start_idx = shard_id * shard_size
        self.end_idx = self.start_idx + shard_size if shard_id < num_shards - 1 else total_size

    def __iter__(self):
        # Start from the specified batch index
        start_sample_idx = self.start_idx + (self.start_batch_idx * self.batch_size)

        for idx in range(start_sample_idx, self.end_idx):
            yield self.dataset[idx]

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
    start_batch_idx: int,
    epoch: int,
    batch_size: int,
    **kwargs  # Ignore extra kwargs
):
    """Load a shard of the training dataset, resuming from a specific position.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10")
        shard_id: Which shard/queue this loader belongs to (0 to num_shards-1)
        num_shards: Total number of shards/queues
        start_batch_idx: Batch index to resume from
        epoch: Current epoch number
        batch_size: Batch size for training

    Returns:
        DataLoader for the assigned shard
    """
    # Load the full training dataset
    dataset = load_dataset(dataset_name, split="train")

    # Apply transforms
    dataset = dataset.with_format("torch").with_transform(apply_cifar10_transforms)

    # Create sharded dataset that can resume from specific batch
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        start_batch_idx=start_batch_idx,
        epoch=epoch,
        batch_size=batch_size,
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
