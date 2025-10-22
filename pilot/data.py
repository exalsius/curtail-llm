"""Data loading utilities for federated learning with epoch-based data management."""

import itertools
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor


class WorkerProgressTracker:
    """Tracks per-worker progress through the dataset.

    Each worker has a unique seed and progresses through epochs independently.
    The shuffle seed for each epoch is: worker_seed + epoch
    """

    def __init__(self, dataset_size: int, batch_size: int):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.batches_per_epoch = (dataset_size + batch_size - 1) // batch_size
        # Track state per worker: {node_id: {"seed": int, "epoch": int, "batch_within_epoch": int}}
        self.worker_states = {}
        self.next_seed = 0  # Auto-increment seed for new workers

    def get_or_create_worker_state(self, node_id: int) -> dict:
        """Get worker state, creating new one if doesn't exist."""
        if node_id not in self.worker_states:
            self.worker_states[node_id] = {
                "seed": self.next_seed,
                "epoch": 0,
                "batch_within_epoch": 0,
            }
            self.next_seed += 100  # Increment by 100 for next worker
        return self.worker_states[node_id]

    def update_worker_progress(self, node_id: int, batches_processed: int):
        """Update worker's progress after processing batches."""
        state = self.worker_states[node_id]
        new_batch_position = state["batch_within_epoch"] + batches_processed

        # Calculate how many epochs were completed
        epochs_completed = new_batch_position // self.batches_per_epoch
        state["epoch"] += epochs_completed
        state["batch_within_epoch"] = new_batch_position % self.batches_per_epoch

    def get_shuffle_seed(self, node_id: int) -> int:
        """Get the current shuffle seed for a worker (seed + epoch)."""
        state = self.worker_states[node_id]
        return state["seed"] + state["epoch"]

    def __repr__(self):
        return f"WorkerProgressTracker(workers={len(self.worker_states)}, states={self.worker_states})"


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
    worker_seed: int,
    start_epoch: int,
    start_batch: int,
    batch_size: int,
):
    """Get the training data loader for federated learning.

    Returns an iterator that automatically wraps across epochs with new shuffling.
    Each epoch uses shuffle seed = worker_seed + epoch for deterministic, unique data.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "uoft-cs/cifar10")
        worker_seed: Unique seed for this worker (stays constant)
        start_epoch: Which epoch to start from
        start_batch: Which batch within the start epoch to begin at
        batch_size: Batch size for training

    Yields:
        Batches continuously, wrapping to next epoch when current one completes

    Note:
        - Each worker sees unique data due to unique worker_seed
        - Different epochs = different shuffles (seed = worker_seed + epoch)
        - Consumer controls how many batches to take
    """
    # Load the full training dataset once
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.with_format("torch").with_transform(apply_cifar10_transforms)
    dataset_size = len(dataset)
    batches_per_epoch = (dataset_size + batch_size - 1) // batch_size

    current_epoch = start_epoch
    current_batch = start_batch

    while True:
        # Create shuffled dataset for current epoch
        shuffle_seed = worker_seed + current_epoch
        generator = torch.Generator().manual_seed(shuffle_seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        shuffled_dataset = Subset(dataset, indices)
        loader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False)

        # Skip to current batch if resuming mid-epoch
        iterator = iter(loader)
        if current_batch > 0:
            for _ in itertools.islice(iterator, current_batch):
                pass

        # Yield batches from this epoch
        for batch in iterator:
            yield batch

        # Epoch complete, move to next
        current_epoch += 1
        current_batch = 0


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
