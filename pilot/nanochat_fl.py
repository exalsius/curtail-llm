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
from pilot.data import fl_shard_dataloader


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
# Training
# ============================================================================

def train(model, trainloader, lr, device, weight_decay=0.01,
          gradient_accumulation_steps=1, log_interval=None, server_round=None,
          round_end_time=None, cumulative_batches=0):
    """Train nanochat model with FL shard dataloader (time-based rounds).

    Args:
        model: GPT model
        trainloader: FL shard dataloader yielding (inputs, targets, shard_id, rows_processed)
        lr: Learning rate
        device: Device to train on
        weight_decay: Weight decay for optimizer
        gradient_accumulation_steps: Gradient accumulation steps
        log_interval: Log metrics every N batches
        server_round: Current round number
        round_end_time: Time when round should end
        cumulative_batches: Total batches processed across all rounds

    Returns:
        tuple: (average_loss, batches_processed, total_rows_processed, shard_updates)
            - shard_updates: list of (shard_id, rows_processed) tuples
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    total_loss = 0.0
    batches_processed = 0
    total_rows_processed = 0

    # Track per-shard progress
    shard_progress = {}  # {shard_id: rows_processed}

    pbar = tqdm(trainloader, desc="Training")

    for inputs, targets, shard_id, rows_in_shard in pbar:
        # Update progress bar with time remaining
        if round_end_time is not None:
            current_time = time.time()
            remaining = max(0, round_end_time - current_time)
            pbar.set_description(f"Training shard_{shard_id} (time left: {remaining:.0f}s)")

        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass in BF16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            loss = model(inputs, targets=targets)
            loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Increment counter
        batches_processed += 1

        if batches_processed % gradient_accumulation_steps == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            optimizer.step()
            optimizer.zero_grad()

            # Log periodically to W&B
            if log_interval and batches_processed % log_interval == 0:
                log_dict = {
                    "train_loss": loss.item() * gradient_accumulation_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm.item(),
                    "batches_processed": batches_processed,
                    "current_shard": shard_id,
                }

                global_step = cumulative_batches + batches_processed
                wandb.log(log_dict, step=global_step)

        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps, "shard": shard_id})

        # Update shard progress (rows_in_shard is cumulative within current shard)
        shard_progress[shard_id] = rows_in_shard
        total_rows_processed = sum(shard_progress.values())

        # Check time limit AFTER completing batch (ensures no partial rows)
        if round_end_time is not None:
            current_time = time.time()
            if current_time >= round_end_time:
                log(INFO, f"Round time expired, stopping after batch {batches_processed}")
                break

    # Flush remaining accumulated gradients
    if (batches_processed % gradient_accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()
        log(INFO, "Flushed accumulated gradients before returning")

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    shard_updates = list(shard_progress.items())

    return avg_loss, batches_processed, total_rows_processed, shard_updates


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

    # Get shard assignments
    shard_assignments_raw = config.get("shard_assignments", [])
    shard_assignments = [(sid, start) for sid, start in shard_assignments_raw]

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

    # Check if we have shard assignments
    if not shard_assignments:
        log(INFO, f"Client {client_id}: No shard assignments (training complete)")
        # Return empty metrics
        train_metrics = {
            "train_loss": 0.0,
            "actual_train_time": 0.0,
            "batches_processed": 0,
            "total_rows_processed": 0,
            "shard_updates": [],
            "client_id": client_id,
        }
        model_state = model.state_dict()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return model_state, train_metrics, 0

    log(INFO, f"Client {client_id}: Using FL shard dataloader with {len(shard_assignments)} shards")
    log(INFO, f"Client {client_id}: Shard assignments: {shard_assignments}")

    # Create FL shard dataloader
    trainloader = fl_shard_dataloader(
        shard_assignments=shard_assignments,
        B=batch_size,
        T=max_length,
        tokenizer_threads=context.run_config.get("tokenizer_threads", 4),
        tokenizer_batch_size=context.run_config.get("tokenizer_batch_size", 128),
        device=device.type,
    )

    if round_end_time:
        remaining_time = round_end_time - round_start_time
        log(INFO, f"Client {client_id}: Round {server_round}, time budget: {remaining_time:.1f}s")
    else:
        log(INFO, f"Client {client_id}: Training without time limit")

    # Train
    train_loss, batches_processed, total_rows_processed, shard_updates = train(
        model=model,
        trainloader=trainloader,
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
    log(INFO, f"Client {client_id}: Actual training time: {actual_train_time:.2f}s")
    log(INFO, f"Client {client_id}: Processed {batches_processed} batches, {total_rows_processed} rows")
    log(INFO, f"Client {client_id}: Shard updates: {shard_updates}")

    train_metrics = {
        "train_loss": train_loss,
        "actual_train_time": actual_train_time,
        "batches_processed": batches_processed,
        "total_rows_processed": total_rows_processed,
        "shard_updates": [[sid, rows] for sid, rows in shard_updates],  # Convert to list for serialization
        "client_id": client_id,
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
