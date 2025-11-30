import time
from logging import INFO

import redis
import torch
import wandb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import ConfigRecord, log
from tqdm import tqdm

from pilot.model import get_model
from pilot.data import fl_shard_dataloader

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    client_id = context.node_config["partition-id"]
    config: ConfigRecord = msg.content["config"]

    if config.get("debug_port_client") and client_id == 0:
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=config["debug_port_client"], stdout_to_server=True, stderr_to_server=True)

    if "cumulative_batches" in context.state:
        cumulative_batches = int(context.state["cumulative_batches"]["cumulative_batches"])
    else:
        cumulative_batches = 0

    # Extract config
    model_type = context.run_config["model_type"]
    batch_size = context.run_config["batch_size"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.01)
    max_length = context.run_config.get("max_length", 2048)
    gradient_accumulation_steps = context.run_config.get("gradient_accumulation_steps", 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    server_round = config.get("server_round", 0)

    redis_url = context.run_config["redis_url"]
    shard_ids = config.get("shard_ids", [])
    shard_starts = config.get("shard_starts", [])
    shard_assignments = list(zip(shard_ids, shard_starts))

    # W&B init
    wandb.init(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=f"client_{client_id}",
        group=config["wandb_group"],
        resume="allow",
        reinit=True,
    )
    log(INFO, f"Client {client_id}: W&B initialized")

    # Load model
    model = get_model(model_type, max_length)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    model.to(device)

    log(INFO, f"Client {client_id}: {len(shard_assignments)} shards: {shard_assignments}")

    # Create dataloader
    trainloader = fl_shard_dataloader(
        shard_assignments=shard_assignments,
        B=batch_size,
        T=max_length,
        tokenizer_threads=context.run_config.get("tokenizer_threads", 4),
        tokenizer_batch_size=context.run_config.get("tokenizer_batch_size", 128),
        device=device.type,
    )

    # Setup Redis pubsub for stop signal
    pubsub = redis.from_url(redis_url).pubsub()
    pubsub.subscribe(f"round:{server_round}:stop")

    # Training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    batches_processed = 0
    shard_progress = {}
    log_interval = context.run_config.get("log_interval")

    pbar = tqdm(trainloader, desc="Training")

    for inputs, targets, shard_id, rows_in_shard in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            loss = model(inputs, targets=targets)
            loss = loss / gradient_accumulation_steps

        loss.backward()
        batches_processed += 1

        if batches_processed % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            optimizer.zero_grad()

            if log_interval and batches_processed % log_interval == 0:
                wandb.log({
                    "train_loss": loss.item() * gradient_accumulation_steps,
                    "learning_rate": lr,
                    "gradient_norm": grad_norm.item(),
                    "batches_processed": batches_processed,
                    "current_shard": shard_id,
                }, step=cumulative_batches + batches_processed)

        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps, "shard": shard_id})

        shard_progress[shard_id] = rows_in_shard

        # Check for stop signal (non-blocking)
        msg = pubsub.get_message(timeout=0)
        if msg and msg['type'] == 'message':
            log(INFO, f"Round stop signal received after batch {batches_processed}")
            break

    # Flush gradients
    if (batches_processed % gradient_accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    total_rows_processed = sum(shard_progress.values())

    log(INFO, f"Client {client_id}: {batches_processed} batches, {total_rows_processed} rows")

    new_cumulative_batches = cumulative_batches + batches_processed
    context.state["cumulative_batches"] = MetricRecord({"cumulative_batches": new_cumulative_batches})

    # Convert shard_progress to two parallel lists
    shard_ids = list(shard_progress.keys())
    shard_rows = list(shard_progress.values())

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord({
                "client_id": client_id,
                "train_loss": avg_loss,
                "batches_processed": batches_processed,
                "total_rows_processed": total_rows_processed,
                "shard_ids": shard_ids,
                "shard_rows": shard_rows,
                "cumulative_batches": new_cumulative_batches,
            }),
        }),
        reply_to=msg,
    )
