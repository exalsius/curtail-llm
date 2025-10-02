from typing import Dict

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM


# Cache datasets to avoid reloading
_dataset_cache = {}

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
    """Apply transforms to the batch."""
    transforms = CIFAR10_TRANSFORMS.get(model_type, CIFAR10_TRANSFORMS["resnet18"])
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def load_cifar10_centralized(model_type="resnet18", batch_size=128):
    """Load centralized CIFAR-10 test set for global evaluation."""
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


def load_sharded_data(
    dataset_name: str,
    shard_id: int,
    num_shards: int,
    start_batch_idx: int,
    epoch: int,
    batch_size: int,
    streaming: bool = False,
    data_format: str = "text",
    **format_kwargs
) -> DataLoader:
    """
    Load data from a specific shard, resuming from start_batch_idx.

    This function implements queue-based data loading where each shard
    represents an independent queue through the dataset. Workers are
    assigned to shards and can resume from arbitrary positions.

    Args:
        dataset_name: HuggingFace dataset name
        shard_id: Which shard/queue to load (0 to num_shards-1)
        num_shards: Total number of shards
        start_batch_idx: Resume from this batch index within the shard
        epoch: Current epoch number (for shuffling seed)
        batch_size: Batch size for DataLoader
        streaming: Use streaming dataset API
        data_format: "text" for LLM data, "image" for vision data
        **format_kwargs: Additional formatting arguments (llm_task, model_type, etc.)

    Returns:
        DataLoader starting from the specified position in the shard
    """

    if streaming:
        # Streaming approach: use HuggingFace's built-in sharding
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        dataset = dataset.shuffle(seed=epoch, buffer_size=10000)
        dataset = dataset.shard(num_shards=num_shards, index=shard_id)

        # Skip to start position
        skip_samples = start_batch_idx * batch_size
        dataset = dataset.skip(skip_samples)

        # Apply formatting for text data
        if data_format == "text":
            llm_task = format_kwargs.get("llm_task", "medical")
            # For streaming datasets, we need to apply formatting differently
            # This is a simplified version - may need more sophisticated handling
            pass  # Formatting will be handled by the trainer

        elif data_format == "image":
            model_type = format_kwargs.get("model_type", "resnet18")
            dataset = dataset.map(lambda batch: apply_transforms(batch, model_type), batched=True)

    else:
        # Full download approach: load entire dataset and manually shard
        cache_key = f"{dataset_name}:epoch{epoch}"

        if cache_key not in _dataset_cache:
            raw_dataset = load_dataset(dataset_name, split="train")
            if not isinstance(raw_dataset, Dataset):
                raise ValueError(f"Expected Dataset but got {type(raw_dataset)}")

            # Shuffle with epoch-specific seed
            shuffled = raw_dataset.shuffle(seed=epoch)

            # Apply formatting based on data type
            if data_format == "text":
                llm_task = format_kwargs.get("llm_task", "medical")
                shuffled = reformat(shuffled, llm_task=llm_task)

            _dataset_cache[cache_key] = shuffled

        full_dataset = _dataset_cache[cache_key]

        # Manual sharding: take every Nth sample starting from shard_id
        total_samples = len(full_dataset)
        shard_indices = list(range(shard_id, total_samples, num_shards))
        shard = full_dataset.select(shard_indices)

        # Skip to start position within shard
        start_sample = start_batch_idx * batch_size
        if start_sample < len(shard):
            remaining = shard.select(range(start_sample, len(shard)))
        else:
            # Shard exhausted, return empty dataset
            # Client should handle this by wrapping to next epoch
            remaining = shard.select([])

        # Apply transforms for images
        if data_format == "image":
            model_type = format_kwargs.get("model_type", "resnet18")
            remaining = remaining.with_transform(lambda b: apply_transforms(b, model_type))

        dataset = remaining

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_data_loaders(
    dataset_name: str,
    shard_id: int,
    num_shards: int,
    start_batch_idx: int,
    epoch: int,
    batch_size: int,
    streaming: bool = False,
    data_format: str = "text",
    **kwargs
) -> DataLoader:
    """
    Factory function for loading sharded data.

    Args:
        dataset_name: HuggingFace dataset name
        shard_id: Which shard/queue to load
        num_shards: Total number of shards (use 1 for centralized/full dataset)
        start_batch_idx: Resume from this batch index
        epoch: Current epoch number
        batch_size: Batch size
        streaming: Use streaming API
        data_format: "text" or "image"
        **kwargs: Additional format-specific arguments

    Returns:
        DataLoader for the requested data
    """
    return load_sharded_data(
        dataset_name=dataset_name,
        shard_id=shard_id,
        num_shards=num_shards,
        start_batch_idx=start_batch_idx,
        epoch=epoch,
        batch_size=batch_size,
        streaming=streaming,
        data_format=data_format,
        **kwargs
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


class QueueManager:
    """Tracks progress of N independent data queues across FL rounds.

    Each queue maintains its position (epoch, batch_idx) in the dataset.
    Workers are dynamically assigned to queues with lowest progress each round,
    ensuring uniform coverage without duplicate work.
    """

    def __init__(self, num_queues: int):
        """Initialize N queues, all starting at (epoch=0, batch_idx=0).

        Args:
            num_queues: Number of independent queues (typically max expected workers)
        """
        self.num_queues = num_queues
        self.queue_states = [(0, 0) for _ in range(num_queues)]  # (epoch, batch_idx)

    def assign_workers(self, num_available: int) -> list[tuple[int, int, int]]:
        """Assign workers to queues with lowest progress.

        Queues are sorted by (epoch, batch_idx) and workers are assigned
        to the N lowest-progress queues.

        Args:
            num_available: Number of workers available this round

        Returns:
            List of (queue_id, epoch, batch_idx) for each worker
        """
        # Sort queues by progress (epoch first, then batch_idx)
        sorted_queues = sorted(
            enumerate(self.queue_states),
            key=lambda x: (x[1][0], x[1][1])
        )

        # Assign workers to N lowest-progress queues
        assignments = []
        for i in range(min(num_available, self.num_queues)):
            queue_id, (epoch, batch_idx) = sorted_queues[i]
            assignments.append((queue_id, epoch, batch_idx))

        return assignments

    def update(self, queue_id: int, final_batch_idx: int, final_epoch: int):
        """Update queue state after worker completes training.

        Args:
            queue_id: Which queue to update
            final_batch_idx: Final batch index reached by worker
            final_epoch: Final epoch reached by worker
        """
        if queue_id < 0 or queue_id >= self.num_queues:
            raise ValueError(f"Invalid queue_id {queue_id}, must be 0-{self.num_queues-1}")

        self.queue_states[queue_id] = (final_epoch, final_batch_idx)

    def __repr__(self) -> str:
        """String representation showing all queue states."""
        queue_strs = [f"Q{i}: (epoch={e}, batch={b})"
                      for i, (e, b) in enumerate(self.queue_states)]
        return f"QueueManager({self.num_queues} queues: {', '.join(queue_strs)})"
