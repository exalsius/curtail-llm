"""Nanochat FL integration for Flower federated learning.

This module adapts the nanochat GPT model for federated learning using Flower.
"""

import time
from logging import INFO

import torch
import wandb
from flwr.common import log
from tqdm import tqdm

from pilot.nanochat.gpt import GPT, GPTConfig


# ============================================================================
# Model Configuration & Loading
# ============================================================================

def get_nanochat_config(model_type="nanochat_d20", max_length=2048):
    """Get nanochat model configuration.

    Args:
        model_type: Model variant (nanochat_d20 or nanochat_d32)
        max_length: Maximum sequence length

    Returns:
        GPTConfig object
    """
    if model_type == "nanochat_d20":
        # Original d20 speedrun config: 560M params
        config = GPTConfig(
            sequence_len=max_length,
            vocab_size=65536,
            n_layer=20,
            n_head=10,
            n_kv_head=10,
            n_embd=1280,
        )
    elif model_type == "nanochat_d32":
        # Larger d32 config: 1.9B params
        config = GPTConfig(
            sequence_len=max_length,
            vocab_size=65536,
            n_layer=32,
            n_head=16,
            n_kv_head=16,
            n_embd=2048,
        )
    else:
        raise ValueError(f"Unknown nanochat model type: {model_type}")

    return config


def get_model(model_type="nanochat_d20", max_length=2048):
    """Load nanochat GPT model.

    Args:
        model_type: Model variant
        max_length: Maximum sequence length

    Returns:
        GPT model
    """
    config = get_nanochat_config(model_type, max_length)
    model = GPT(config)
    model.init_weights()
    return model


# ============================================================================
# Simple Data Loading (for initial testing)
# ============================================================================

class SimpleTextDataLoader:
    """Simple dataloader for text data during initial testing.

    For full nanochat training, use the streaming parquet loader from
    pilot.nanochat.dataloader. This simplified version is for getting
    federated learning working first.
    """

    def __init__(self, texts, tokenizer, batch_size, max_length, device):
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.num_batches = len(texts) // batch_size

    def __iter__(self):
        bos_token = self.tokenizer.get_bos_token_id()

        for i in range(self.num_batches):
            batch_texts = self.texts[i * self.batch_size: (i + 1) * self.batch_size]

            # Tokenize batch
            token_lists = self.tokenizer.encode(batch_texts, prepend=bos_token)

            # Pad/truncate to max_length + 1 (for targets)
            batch_tokens = []
            for tokens in token_lists:
                if len(tokens) > self.max_length + 1:
                    tokens = tokens[:self.max_length + 1]
                else:
                    # Pad with BOS token (simple padding strategy)
                    tokens = tokens + [bos_token] * (self.max_length + 1 - len(tokens))
                batch_tokens.append(tokens)

            # Convert to tensor
            tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)

            # Split into inputs and targets
            inputs = tokens_tensor[:, :-1]  # All but last token
            targets = tokens_tensor[:, 1:]   # All but first token

            yield inputs, targets

    def __len__(self):
        return self.num_batches


# ============================================================================
# Training
# ============================================================================

def train(model, trainloader, num_batches, lr, device, weight_decay=0.01,
          gradient_accumulation_steps=1, log_interval=None, server_round=None,
          round_end_time=None, cumulative_batches=0):
    """Train nanochat model with time-based rounds.

    Args:
        model: GPT model
        trainloader: Data loader yielding (inputs, targets)
        num_batches: Maximum number of batches to process
        lr: Learning rate
        device: Device to train on
        weight_decay: Weight decay for optimizer
        gradient_accumulation_steps: Gradient accumulation steps
        log_interval: Log metrics every N batches
        server_round: Current round number
        round_end_time: Time when round should end
        cumulative_batches: Total batches processed across all rounds

    Returns:
        tuple: (average_loss, batches_processed)
    """
    model.to(device)

    # Use standard AdamW optimizer (simpler than Muon for initial implementation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    total_loss = 0.0
    batches_processed = 0

    pbar = tqdm(enumerate(trainloader), total=num_batches, desc="Training")

    for batch_idx, (inputs, targets) in pbar:
        # Update progress bar with time remaining
        if round_end_time is not None:
            current_time = time.time()
            remaining = max(0, round_end_time - current_time)
            pbar.set_description(f"Training (time left: {remaining:.0f}s)")

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass in BF16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            loss = model(inputs, targets=targets)
            loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            optimizer.step()
            optimizer.zero_grad()

            # Log periodically to W&B
            if log_interval and (batch_idx + 1) % log_interval == 0:
                log_dict = {
                    "train_loss": loss.item() * gradient_accumulation_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm.item(),
                    "batch_idx": batch_idx,
                }

                # Use cumulative batches as global step
                global_step = cumulative_batches + batches_processed
                wandb.log(log_dict, step=global_step)

        total_loss += loss.item() * gradient_accumulation_steps
        batches_processed += 1
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        # Check time limit AFTER completing batch
        if round_end_time is not None:
            current_time = time.time()
            if current_time >= round_end_time:
                log(INFO, f"Round time expired (current: {current_time:.1f} >= end: {round_end_time:.1f}), stopping training")
                break

        # Stop if we've processed enough batches
        if batches_processed >= num_batches:
            break

    # Flush any remaining accumulated gradients
    if (batches_processed % gradient_accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
        log(INFO, "Flushed accumulated gradients before returning")

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, batches_processed


@torch.no_grad()
def evaluate(model, evalloader, device):
    """Evaluate nanochat model.

    Args:
        model: GPT model
        evalloader: Data loader yielding (inputs, targets)
        device: Device to evaluate on

    Returns:
        float: Average loss
    """
    model.eval()
    total_loss = 0.0
    batches = 0

    for inputs, targets in tqdm(evalloader, desc="Evaluating"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference in BF16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            loss = model(inputs, targets=targets)

        total_loss += loss.item()
        batches += 1

    return (total_loss / batches) if batches > 0 else 0.0


# ============================================================================
# Federated Learning Client Handler
# ============================================================================

def train_client(msg, config, context, cumulative_batches=0):
    """Handle nanochat training on federated client.

    Args:
        msg: Message containing model weights
        config: Configuration dict
        context: Flower context
        cumulative_batches: Total batches processed across all rounds

    Returns:
        tuple: (model_state_dict, metrics, batches_processed)
    """
    import time as time_module

    model_type = context.run_config["model_type"]
    batch_size = context.run_config["batch_size"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.01)
    max_length = context.run_config.get("max_length", 2048)
    gradient_accumulation_steps = context.run_config.get("gradient_accumulation_steps", 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Extract round timing
    server_round = config.get("server_round", 0)
    round_end_time = config.get("round_end_time")
    round_start_time = time_module.time()

    # W&B configuration
    wandb_run_id = config["wandb_run_id"]
    wandb_project = config["wandb_project"]
    wandb_group = config["wandb_group"]
    wandb_entity = config.get("wandb_entity")
    log_interval = context.run_config.get("log_interval")

    # Get client ID
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

    # Load model
    model = get_model(model_type, max_length)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    model.to(device)

    # For now, use dummy data (will integrate streaming parquet loader later)
    # TODO: Integrate nanochat.dataloader.tokenizing_distributed_data_loader
    log(INFO, f"Client {client_id}: Using dummy data (streaming loader not yet integrated)")

    # Create dummy dataset for testing
    # Try to load nanochat tokenizer, fallback to HuggingFace tokenizer
    try:
        from pilot.nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
        log(INFO, f"Client {client_id}: Using nanochat tokenizer")
    except Exception as e:
        log(INFO, f"Client {client_id}: Nanochat tokenizer not available ({e}), using GPT-2 tokenizer as fallback")
        from pilot.nanochat.tokenizer import HuggingFaceTokenizer
        tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2")
        log(INFO, f"Client {client_id}: Loaded GPT-2 tokenizer (vocab_size={tokenizer.get_vocab_size()})")

    # Generate some dummy text data
    dummy_texts = ["This is a test document for nanochat training. " * 100] * (batch_size * 10)

    trainloader = SimpleTextDataLoader(
        texts=dummy_texts,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    num_batches = len(trainloader)

    if round_end_time:
        remaining_time = round_end_time - round_start_time
        log(INFO, f"Client {client_id}: Round {server_round}, max {num_batches} batches, time budget: {remaining_time:.1f}s")
    else:
        log(INFO, f"Client {client_id}: Training for {num_batches} batches (no time limit)")

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
    log(INFO, f"Client {client_id}: Actual training time: {actual_train_time:.2f}s, processed {batches_processed} batches")

    train_metrics = {
        "train_loss": train_loss,
        "actual_train_time": actual_train_time,
    }

    # Extract state dict before cleanup
    model_state = model.state_dict()

    # Cleanup
    log(INFO, f"Client {client_id}: Cleaning up model memory...")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return model_state, train_metrics, batches_processed
