"""Medical AI module for medAlpaca integration with Flower federated learning.

This module provides medical domain-specific fine-tuning capabilities using
the medAlpaca approach with Medical Meadow datasets.

Based on: https://github.com/kbressem/medAlpaca
"""

import json
import os
from logging import INFO
from typing import Optional

import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from flwr.common import log
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot.data import ShardedDataset, MEDICAL_DATASETS


# ============================================================================
# Model Loading
# ============================================================================


def get_model(model_name: str = "chavinlo/alpaca-13b", **kwargs):
    """Load Alpaca base model for medical fine-tuning.

    Args:
        model_name: HuggingFace model name (default: chavinlo/alpaca-13b)
        **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained

    Returns:
        Loaded model
    """
    log(INFO, f"Loading medical base model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        **kwargs
    )

    log(INFO, f"Model loaded: {model_name}")
    return model


def apply_lora(model, lora_config: Optional[dict] = None):
    """Apply LoRA adapters for medical domain fine-tuning.

    Args:
        model: Base model to apply LoRA to
        lora_config: LoRA configuration dict (r, lora_alpha, target_modules, etc.)

    Returns:
        Model with LoRA adapters applied
    """
    if lora_config is None:
        # Enhanced LoRA config for more computationally expensive training
        # Significantly increased from original medAlpaca config (r=8, 2 modules)
        lora_config = {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

    log(INFO, f"Applying LoRA with config: {lora_config}")

    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    log(INFO, f"Trainable params: {trainable_params:,} / {all_params:,} "
             f"({100 * trainable_params / all_params:.2f}%)")

    return model


# ============================================================================
# Medical Data Handling
# ============================================================================


class MedicalDataHandler:
    """Handles medical prompt formatting and tokenization.

    Adapted from medAlpaca's DataHandler to work with Flower's federated learning.
    """

    def __init__(self, tokenizer, prompt_template_path: str, max_length: int = 512):
        """Initialize medical data handler.

        Args:
            tokenizer: HuggingFace tokenizer
            prompt_template_path: Path to JSON prompt template
            max_length: Maximum sequence length (default: 512)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load prompt template
        with open(prompt_template_path, 'r') as f:
            self.prompt_template = json.load(f)

    def generate_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """Generate formatted prompt from instruction/input/output.

        Args:
            instruction: Task instruction
            input_text: Optional input context
            output: Optional output text (for training)

        Returns:
            Formatted prompt string
        """
        prompt = self.prompt_template["primer"]
        prompt += self.prompt_template["instruction"] + instruction

        if input_text:
            prompt += self.prompt_template["input"] + input_text

        prompt += self.prompt_template["output"]

        if output:
            prompt += output

        return prompt

    def tokenize(self, prompt: str, add_eos_token: bool = True):
        """Tokenize a prompt.

        Args:
            prompt: Text prompt to tokenize
            add_eos_token: Whether to add EOS token

        Returns:
            Tokenized result dict
        """
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        if add_eos_token and result["input_ids"][-1] != self.tokenizer.eos_token_id:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        # Add labels for causal LM
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, example: dict) -> dict:
        """Generate prompt from example and tokenize it.

        Args:
            example: Dataset example with 'instruction', 'input', 'output' fields

        Returns:
            Tokenized prompt dict
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        full_prompt = self.generate_prompt(instruction, input_text, output)
        return self.tokenize(full_prompt)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_tokenizer(model_type: str):
    """Load tokenizer with pad_token fallback."""
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _create_data_handler(tokenizer, max_length: int = 512):
    """Create medical data handler with prompt template."""
    template_path = os.path.join(
        os.path.dirname(__file__),
        "prompt_templates/medalpaca.json"
    )
    return MedicalDataHandler(tokenizer, template_path, max_length)


def _create_collate_fn(tokenizer):
    """Create collate function for padding batches."""
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn


# ============================================================================
# Data Loading
# ============================================================================


def get_combined_medical_meadow(shuffle_seed: int = 42):
    """Load curated Medical Meadow combination with global shuffle.

    IMPORTANT: After combining, globally shuffled to ensure unbiased train/eval splits.

    Args:
        shuffle_seed: Random seed for reproducibility

    Returns:
        Combined and globally shuffled HuggingFace Dataset (77,192 samples)
    """
    log(INFO, "Loading curated Medical Meadow combination...")

    datasets_to_combine = [
        load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train"),
        load_dataset("medalpaca/medical_meadow_wikidoc", split="train"),
        load_dataset("medalpaca/medical_meadow_wikidoc_patient_information", split="train"),
        load_dataset("medalpaca/medical_meadow_health_advice", split="train"),
        load_dataset("medalpaca/medical_meadow_mediqa", split="train"),
        load_dataset("medalpaca/medical_meadow_mmmlu", split="train"),
        load_dataset("medalpaca/medical_meadow_medqa", split="train"),
        load_dataset("medalpaca/medical_meadow_pubmed_causal", split="train"),
    ]

    # Combine and shuffle globally for unbiased train/eval splits
    combined = concatenate_datasets(datasets_to_combine)
    combined = combined.shuffle(seed=shuffle_seed)

    log(INFO, f"Combined Medical Meadow: {len(combined)} samples (globally shuffled)")
    return combined


def get_train_loader(
    dataset_name: str,
    shard_id: int,
    num_shards: int,
    processed_batches: int,
    batch_size: int,
    model_type: str,
    max_length: int = 512,
):
    """Create training dataloader for curated Medical Meadow dataset.

    Args:
        dataset_name: Ignored (always uses curated dataset)
        shard_id: Shard ID for this client
        num_shards: Total number of shards
        processed_batches: Number of batches already processed
        batch_size: Batch size
        model_type: Model type (for tokenizer selection)
        max_length: Maximum sequence length

    Returns:
        DataLoader for training
    """
    log(INFO, "Loading medical training data (curated dataset)")

    tokenizer = _get_tokenizer(model_type)
    data_handler = _create_data_handler(tokenizer, max_length)

    # Load curated combined dataset
    dataset = get_combined_medical_meadow()

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        data_handler.generate_and_tokenize_prompt,
        remove_columns=dataset.column_names,
    )

    # Create sharded dataset
    sharded_dataset = ShardedDataset(
        dataset=tokenized_dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        dataset_name=None,
        streaming=False,
    )

    return DataLoader(
        sharded_dataset,
        batch_size=batch_size,
        collate_fn=_create_collate_fn(tokenizer),
    )


def get_eval_loader(
    dataset_name: str,
    batch_size: int,
    model_type: str,
    max_length: int = 512,
):
    """Create evaluation dataloader (last 10% of curated dataset).

    Args:
        dataset_name: Ignored (always uses curated dataset)
        batch_size: Batch size
        model_type: Model type (for tokenizer selection)
        max_length: Maximum sequence length

    Returns:
        DataLoader for evaluation
    """
    log(INFO, "Loading medical evaluation data (curated dataset, last 10%)")

    tokenizer = _get_tokenizer(model_type)
    data_handler = _create_data_handler(tokenizer, max_length)

    # Load curated dataset and use last 10% for validation
    dataset = get_combined_medical_meadow()
    total_size = len(dataset)
    val_dataset = dataset.select(range(int(0.9 * total_size), total_size))

    # Tokenize dataset
    tokenized_dataset = val_dataset.map(
        data_handler.generate_and_tokenize_prompt,
        remove_columns=val_dataset.column_names,
    )

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=_create_collate_fn(tokenizer),
    )


def get_server_eval_loader(
    dataset_name: str,
    batch_size: int,
    model_type: str,
    max_length: int = 512,
    holdout_fraction: float = 0.1,
):
    """Create server-side evaluation dataloader using holdout fraction.

    Args:
        dataset_name: Ignored (always uses curated dataset)
        batch_size: Batch size for evaluation
        model_type: Model type (for tokenizer selection)
        max_length: Maximum sequence length
        holdout_fraction: Fraction of training data to use for evaluation (default 10%)

    Returns:
        DataLoader for server evaluation
    """
    log(INFO, f"Loading medical server evaluation data (curated dataset, holdout: {holdout_fraction*100:.0f}%)")

    tokenizer = _get_tokenizer(model_type)
    data_handler = _create_data_handler(tokenizer, max_length)

    # Load curated dataset and calculate holdout range
    dataset = get_combined_medical_meadow()
    total_size = len(dataset)
    holdout_start = int(total_size * (1 - holdout_fraction))
    holdout_dataset = dataset.select(range(holdout_start, total_size))

    # Tokenize dataset
    tokenized_dataset = holdout_dataset.map(
        data_handler.generate_and_tokenize_prompt,
        remove_columns=holdout_dataset.column_names,
    )

    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=_create_collate_fn(tokenizer),
    )


# ============================================================================
# Training (Reused from llm.py with medical-specific adaptations)
# ============================================================================


def train(model, trainloader, num_batches, lr, device, weight_decay=0.01,
          gradient_accumulation_steps=1, log_interval=None, server_round=None,
          round_end_time=None, cumulative_batches=0):
    """Train medical model with LoRA using BF16 mixed precision until `round_end_time`.

    This function is adapted from pilot/llm.py to work with medical datasets.
    """
    import time as time_module

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    total_loss = 0.0
    batches_processed = 0
    pbar = tqdm(enumerate(trainloader), total=num_batches, desc="Training")

    for batch_idx, batch in pbar:
        # Update progress bar with time remaining
        if round_end_time is not None:
            current_time = time_module.time()
            remaining = max(0, round_end_time - current_time)
            pbar.set_description(f"Training (time left: {remaining:.0f}s)")

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass in BF16 (parameters stay FP32, compute happens in BF16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        # Backward pass (gradients computed in BF16, accumulated in FP32)
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Compute gradient norm before optimizer step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            optimizer.step()
            optimizer.zero_grad()

            # Log periodically to W&B
            if log_interval and (batch_idx + 1) % log_interval == 0:
                log_dict = {
                    "train_loss": loss.item() * gradient_accumulation_steps,
                    "train_ppl": float(torch.exp(loss * gradient_accumulation_steps).item()),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm.item(),
                    "batch_idx": batch_idx,
                }

                # Use cumulative batches as global step for accurate progress tracking
                global_step = cumulative_batches + batches_processed
                wandb.log(log_dict, step=global_step)

        total_loss += loss.item() * gradient_accumulation_steps
        batches_processed += 1
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        # Check time limit AFTER completing batch (including optimizer step if applicable)
        # This ensures we don't lose accumulated gradients
        if round_end_time is not None:
            current_time = time_module.time()
            if current_time >= round_end_time:
                log(INFO, f"Round time expired (current: {current_time:.1f} >= end: {round_end_time:.1f}), stopping training")
                break

    # Flush any remaining accumulated gradients if training stopped mid-accumulation
    if (batches_processed % gradient_accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
        log(INFO, "Flushed accumulated gradients before returning (stopped mid-accumulation)")

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, batches_processed


@torch.no_grad()
def evaluate(model, evalloader, device):
    model.eval()
    total_loss = 0.0
    batches_evaluated = 0

    pbar = tqdm(evalloader, desc="Evaluating")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        total_loss += loss.item()
        batches_evaluated += 1
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / batches_evaluated if batches_evaluated > 0 else 0.0
    return avg_loss


# ============================================================================
# Federated Learning Client Handler
# ============================================================================


def train_client(msg, config, context, cumulative_batches=0):
    """Federated learning client handler for medical training.

    Args:
        msg: Flower message containing model weights
        config: Configuration dict with training parameters
        context: Flower context
        cumulative_batches: Total batches processed by this client across all rounds

    Returns:
        Tuple of (state_dict, metrics_dict, batches_processed)
    """
    import time as time_module

    # Extract configuration
    shard_id: int = config["shard_id"]
    processed_batches: int = config["processed_batches"]
    lr: float = config["lr"]
    dataset_name: str = config["dataset_name"]
    num_shards: int = config["num_shards"]

    batch_size: int = context.run_config["batch_size"]
    model_type: str = context.run_config["model_type"]
    max_length: int = context.run_config.get("max_length", 512)
    weight_decay: float = context.run_config.get("weight_decay", 0.01)
    gradient_accumulation_steps = context.run_config.get("gradient_accumulation_steps", 1)
    lora_config = context.run_config.get("lora_config", None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Extract round timing
    server_round = config.get("server_round", 0)
    round_end_time = config.get("round_end_time")
    round_start_time = time_module.time()  # Track when we actually start training

    # W&B configuration for client
    wandb_run_id = config["wandb_run_id"]
    wandb_project = config["wandb_project"]
    wandb_group = config["wandb_group"]
    wandb_entity = config.get("wandb_entity")
    log_interval = context.run_config.get("log_interval")

    # Get client ID from context
    client_id = context.node_id

    # Initialize W&B for this client
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=f"medical_client_{client_id}",
        id=f"{wandb_run_id}_medical_{client_id}",
        group=wandb_group,
        resume="allow",
        reinit=True,
    )
    log(INFO, f"Medical Client {client_id}: W&B initialized with run_id {wandb_run_id}_medical_{client_id}")

    if dataset_name not in MEDICAL_DATASETS:
        raise ValueError(f"Unknown medical dataset: {dataset_name}")

    # Calculate max batches for full shard (but time limit may stop earlier)
    total_size = MEDICAL_DATASETS[dataset_name]
    shard_size = total_size // num_shards
    num_batches = max(1, shard_size // batch_size)

    if round_end_time:
        remaining_time = round_end_time - round_start_time
        log(INFO, f"Medical Client {client_id}: Round {server_round}, max {num_batches} batches, time budget: {remaining_time:.1f}s")
    else:
        log(INFO, f"Medical Client {client_id}: Training for {num_batches} batches (no time limit)")

    base_model = get_model(model_type)
    model = apply_lora(base_model, lora_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)

    # Get data loaders
    trainloader = get_train_loader(
        dataset_name=dataset_name,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        model_type=model_type,
        max_length=max_length,
    )

    # Train
    train_loss, batches_processed = train(
        model=model,
        trainloader=trainloader,
        num_batches=num_batches,
        lr=lr,
        device=device,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_interval=log_interval,
        server_round=server_round,
        round_end_time=round_end_time,
        cumulative_batches=cumulative_batches,
    )

    # Log actual training time
    actual_train_time = time_module.time() - round_start_time
    log(INFO, f"Medical Client {client_id}: Actual training time: {actual_train_time:.2f}s, processed {batches_processed} batches")
    if wandb.run:
        wandb.log({"client/actual_train_time": actual_train_time}, step=server_round)

    # Compute perplexity from training loss
    train_ppl = float(torch.exp(torch.tensor(train_loss)).item()) if train_loss > 0 else float("inf")

    train_metrics = {"train_loss": train_loss, "train_ppl": train_ppl}

    # Extract state dict before cleanup
    model_state = model.state_dict()

    # TEMPORARY: Aggressive cleanup for simulation mode (in production, each client on separate node)
    # This prevents memory accumulation across rounds in local simulation
    log(INFO, f"Medical Client {client_id}: Cleaning up model memory...")
    del model
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return model_state, train_metrics, batches_processed
