"""LLM models, data loading, and training with LoRA fine-tuning."""

from logging import INFO

import torch
import wandb
from flwr.common import log
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from pilot.data import ShardedDataset


# ============================================================================
# Models
# ============================================================================

def get_model(model_name, **kwargs):
    """Load HuggingFace LLM model."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
        **kwargs
    )


def apply_lora(model, lora_config=None):
    """Apply LoRA adapters to model."""
    if lora_config is None:
        lora_config = {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }
    peft_config = LoraConfig(**lora_config)
    return get_peft_model(model, peft_config)


def get_trainable_params(model):
    """Get trainable parameter statistics."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percentage": 100 * trainable / total,
    }


# ============================================================================
# Data Loading & Tokenization
# ============================================================================

class TokenizedDataLoader:
    """Wraps raw dataloader with tokenization."""

    def __init__(self, raw_loader, tokenizer, max_length):
        self.raw_loader = raw_loader
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for batch in self.raw_loader:
            # Extract texts from batch
            texts = self._extract_texts(batch)

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            yield tokenized

    def _extract_texts(self, batch):
        """Extract text from various dataset formats."""
        if "messages" in batch:
            # UltraChat format: always use chat template from the tokenizer
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer must support apply_chat_template for UltraChat datasets")
            msgs_list = batch["messages"]
            return [
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for msgs in msgs_list
            ]
        raise ValueError("Unsupported dataset sample format for UltraChat — expected 'messages' key")

    def _format_chat(self, messages):
        """Format chat messages into single string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(formatted)


def _ultrachat_splits(dataset_name: str) -> tuple[str, str]:
    """Return the exact splits for UltraChat SFT. Raise for other datasets."""
    if dataset_name != "HuggingFaceH4/ultrachat_200k":
        raise ValueError("This codepath is specialized for HuggingFaceH4/ultrachat_200k only")
    return "train_sft", "test_sft"


def get_train_loader(dataset_name, shard_id, num_shards, processed_batches,
                     batch_size, model_type, max_length=512):
    """Get training dataloader for LLM dataset with tokenization."""
    from datasets import load_dataset

    # Use map-style datasets (Arrow, memory-mapped). No streaming.
    use_streaming = False

    # Load dataset (map-style) with UltraChat SFT split
    train_split, _ = _ultrachat_splits(dataset_name)
    dataset = load_dataset(dataset_name, split=train_split, streaming=use_streaming)

    # Create sharded dataset
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        dataset_name=None,
        streaming=use_streaming,
    )

    # Collate into dict of lists (avoids default tensor stacking on variable-length fields)
    def collate_to_lists(samples):
        if not samples:
            return {}
        keys = samples[0].keys()
        return {k: [s[k] for s in samples] for k in keys}

    raw_loader = DataLoader(sharded_dataset, batch_size=batch_size, collate_fn=collate_to_lists)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap with tokenization
    return TokenizedDataLoader(raw_loader, tokenizer, max_length)


def get_eval_loader(dataset_name, batch_size, model_type, max_length=512):
    """Get evaluation dataloader (non-sharded) for LLM dataset with tokenization."""
    from datasets import load_dataset

    # Map-style datasets
    use_streaming = False

    _, eval_split = _ultrachat_splits(dataset_name)
    dataset = load_dataset(dataset_name, split=eval_split, streaming=use_streaming)

    # Wrap as a simple PyTorch DataLoader with dict-of-lists collation
    def collate_to_lists(samples):
        if not samples:
            return {}
        keys = samples[0].keys()
        return {k: [s[k] for s in samples] for k in keys}
    raw_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_to_lists)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return TokenizedDataLoader(raw_loader, tokenizer, max_length)


def get_server_eval_loader(dataset_name, batch_size, model_type, max_length=512, holdout_fraction=0.1):
    """Get server-side evaluation dataloader using holdout fraction of training data.

    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for evaluation
        model_type: Model type for tokenizer
        max_length: Maximum sequence length
        holdout_fraction: Fraction of training data to use for evaluation (default 10%)

    Returns:
        TokenizedDataLoader for evaluation
    """
    from datasets import load_dataset

    if dataset_name != "HuggingFaceH4/ultrachat_200k":
        raise ValueError("get_server_eval_loader is specialized for HuggingFaceH4/ultrachat_200k")

    # Load training split
    train_split, _ = _ultrachat_splits(dataset_name)
    dataset = load_dataset(dataset_name, split=train_split, streaming=False)

    # Calculate holdout range (last 10% of training data)
    total_size = len(dataset)
    holdout_size = int(total_size * holdout_fraction)
    holdout_start = total_size - holdout_size

    # Select holdout subset
    holdout_dataset = dataset.select(range(holdout_start, total_size))

    # Wrap as DataLoader with dict-of-lists collation
    def collate_to_lists(samples):
        if not samples:
            return {}
        keys = samples[0].keys()
        return {k: [s[k] for s in samples] for k in keys}
    raw_loader = DataLoader(holdout_dataset, batch_size=batch_size, collate_fn=collate_to_lists)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return TokenizedDataLoader(raw_loader, tokenizer, max_length)


# ============================================================================
# Training
# ============================================================================

def train(model, trainloader, num_batches, lr, device, weight_decay=0.01,
          gradient_accumulation_steps=1, log_interval=None, server_round=None,
          round_end_time=None, cumulative_batches=0):
    """Train LLM with LoRA using BF16 mixed precision until `round_end_time`"""
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
    batches = 0

    for batch in tqdm(evalloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Inference in BF16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        total_loss += outputs.loss.item()
        batches += 1

    return (total_loss / batches) if batches > 0 else 0.0


# ============================================================================
# Federated Learning Client Handler
# ============================================================================

def train_client(msg, config, context, cumulative_batches=0):
    """Handle LLM training on federated client.

    Args:
        cumulative_batches: Total batches processed by this client across all rounds

    Returns:
        tuple: (model_state_dict, metrics_payload, batches_processed)
    """
    import time as time_module

    model_type = context.run_config["model_type"]
    dataset_name = config["dataset_name"]
    shard_id = config["shard_id"]
    num_shards = config["num_shards"]
    processed_batches = config["processed_batches"]
    batch_size = context.run_config["batch_size"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.01)
    max_length = context.run_config.get("max_length", 512)
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
        name=f"client_{client_id}",
        id=f"{wandb_run_id}_{client_id}",
        group=wandb_group,
        resume="allow",
        reinit=True,
    )
    log(INFO, f"Client {client_id}: W&B initialized with run_id {wandb_run_id}_{client_id}")

    if dataset_name != "HuggingFaceH4/ultrachat_200k":
        raise ValueError("LLM train_client is specialized for HuggingFaceH4/ultrachat_200k")

    # Calculate max batches for full shard (but time limit may stop earlier)
    from datasets import load_dataset
    train_split, _ = _ultrachat_splits(dataset_name)
    total_size = len(load_dataset(dataset_name, split=train_split))
    shard_size = total_size // num_shards
    num_batches = max(1, shard_size // batch_size)

    if round_end_time:
        remaining_time = round_end_time - round_start_time
        log(INFO, f"Client {client_id}: Round {server_round}, max {num_batches} batches, time budget: {remaining_time:.1f}s")
    else:
        log(INFO, f"Client {client_id}: Training for {num_batches} batches (no time limit)")

    base_model = get_model(model_type)
    model = apply_lora(base_model, lora_config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(device)

    trainloader = get_train_loader(
        dataset_name=dataset_name,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        model_type=model_type,
        max_length=max_length,
    )

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
    log(INFO, f"Client {client_id}: Actual training time: {actual_train_time:.2f}s, processed {batches_processed} batches")

    # Compute perplexity from training loss
    train_ppl = float(torch.exp(torch.tensor(train_loss)).item()) if train_loss > 0 else float("inf")

    train_metrics = {
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "actual_train_time": actual_train_time,
    }

    # Extract state dict before cleanup
    model_state = model.state_dict()

    # TEMPORARY: Aggressive cleanup for simulation mode (in production, each client on separate node)
    # This prevents memory accumulation across rounds in local simulation
    log(INFO, f"Client {client_id}: Cleaning up model memory...")
    del model
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return model_state, train_metrics, batches_processed
