from typing import Dict

import torch
from datasets import Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


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


def formatting_prompts_func(example):
    """Construct prompts following the Flowertune convention."""

    output_texts = []
    header = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    for idx in range(len(example["instruction"])):
        text = (
            f"{header}\n### Instruction:\n{example['instruction'][idx]}\n"
            f"### Response: {example['response'][idx]}"
        )
        output_texts.append(text)

    return output_texts


def get_tokenizer_and_data_collator_and_prompt_formatting(model_name: str):
    """Mirror Flowertune tokenizer/collator setup for completion-only tuning."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    response_template = "\n### Response:"
    response_template_ids = tokenizer.encode(
        response_template,
        add_special_tokens=False,
    )[2:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer,
    )

    return tokenizer, data_collator, formatting_prompts_func


def _formatting_for_code(dataset: Dataset) -> Dataset:
    def _formatting(example: Dict[str, str]) -> Dict[str, str]:
        return {
            "instruction": example["instruction"] + " " + example["input"],
            "response": example["response"],
        }

    return dataset.map(_formatting, remove_columns=["input"])


def reformat(dataset: Dataset, llm_task: str) -> Dataset:
    """Apply Flowertune task-specific formatting to a dataset."""

    dataset = dataset.rename_column("output", "response")

    if llm_task in {"finance", "code"}:
        return _formatting_for_code(dataset)

    if llm_task == "medical":
        dataset = dataset.remove_columns(["instruction"])
        return dataset.rename_column("input", "instruction")

    return dataset


def load_flowertune_partition(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    llm_task: str = "medical",
) -> Dataset:
    """Load a federated partition using the Flowertune benchmark helper."""

    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id, "train")
    return reformat(partition, llm_task=llm_task)


def load_text_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    *,
    dataset_name: str,
    llm_task: str = "medical",
) -> Dataset:
    """Load Flowertune-style text data and return the federated partition."""

    return load_flowertune_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        dataset_name=dataset_name,
        llm_task=llm_task,
    )


def replace_keys(input_dict: Dict, match: str = "-", target: str = "_") -> Dict:
    """Recursively replace characters in dictionary keys (Flowertune helper)."""

    new_dict: Dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


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
