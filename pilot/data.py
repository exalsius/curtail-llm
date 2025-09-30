"""Data loading and preprocessing utilities for federated learning."""

import torch
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


# Cache FederatedDataset to avoid reloading
fds = None

# Image transforms for different model types
CIFAR10_TRANSFORMS = {
    "simple_cnn": Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    "resnet18": Compose([
        Resize((224, 224)),  # ResNet expects 224x224 input
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    ])
}


def apply_transforms(batch, model_type="resnet18"):
    """Apply transforms to the partition from FederatedDataset."""
    transforms = CIFAR10_TRANSFORMS.get(model_type, CIFAR10_TRANSFORMS["resnet18"])
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def load_cifar10_data(partition_id: int, num_partitions: int, batch_size: int, model_type="resnet18"):
    """Load partition CIFAR10 data for image classification models."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Apply transforms based on model type
    transform_fn = lambda batch: apply_transforms(batch, model_type)
    partition_train_test = partition_train_test.with_transform(transform_fn)

    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)

    return trainloader, testloader


def load_cifar10_centralized(model_type="resnet18", batch_size=128):
    """Load centralized CIFAR-10 test set for evaluation."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    # Apply transforms based on model type
    transform_fn = lambda batch: apply_transforms(batch, model_type)
    dataset = test_dataset.with_format("torch").with_transform(transform_fn)

    return DataLoader(dataset, batch_size=batch_size)


def load_text_data(partition_id: int, num_partitions: int, batch_size: int, dataset_name="wikitext-2-raw-v1"):
    """Load partition text data for language models."""
    # This is a placeholder for text data loading
    # Will be implemented in the next prompt as mentioned
    raise NotImplementedError("Text data loading for Mistral model will be implemented in the next prompt")


def prepare_text_batch(batch, tokenizer, max_length=512):
    """Prepare a batch of text data for training."""
    # This is a placeholder for text preprocessing
    # Will be implemented in the next prompt as mentioned
    raise NotImplementedError("Text preprocessing for Mistral model will be implemented in the next prompt")


# Factory function for data loading
def get_data_loaders(
    data_type="cifar10",
    partition_id=0,
    num_partitions=2,
    batch_size=32,
    model_type="resnet18",
    centralized=False,
    **kwargs
):
    """Factory function to get data loaders based on data type."""
    if data_type == "cifar10":
        if centralized:
            return load_cifar10_centralized(model_type=model_type, batch_size=batch_size)
        else:
            return load_cifar10_data(
                partition_id=partition_id,
                num_partitions=num_partitions,
                batch_size=batch_size,
                model_type=model_type
            )
    elif data_type == "text":
        if centralized:
            raise NotImplementedError("Centralized text data loading not implemented yet")
        else:
            return load_text_data(
                partition_id=partition_id,
                num_partitions=num_partitions,
                batch_size=batch_size,
                **kwargs
            )
    else:
        raise ValueError(f"Unknown data type: {data_type}")


