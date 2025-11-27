import time
from logging import INFO

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
    round_end_time = config.get("round_end_time")
    round_start_time = time.time()

    shard_assignments_raw = config.get("shard_assignments", [])
    shard_assignments = [(sid, start) for sid, start in shard_assignments_raw]

    # W&B init
    wandb.init(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=f"client_{client_id}",
        id=f"{config['wandb_run_id']}_{client_id}",
        group=config["wandb_group"],
        resume="allow",
        reinit=True,
    )
    log(INFO, f"Client {client_id}: W&B initialized")

    # Load model
    model = get_model(model_type, max_length)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)
    model.to(device)

    # Handle empty shard assignments
    if not shard_assignments:
        log(INFO, f"Client {client_id}: No shards (training complete)")
        state_dict = model.state_dict()
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return Message(
            content=RecordDict({
                "arrays": ArrayRecord(state_dict),
                "metrics": MetricRecord({
                    "client_id": client_id,
                    "train_loss": 0.0,
                    "batches_processed": 0,
                    "total_rows_processed": 0,
                    "shard_updates": [],
                }),
            }),
            reply_to=msg,
        )

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

    # Training loop
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_loss = 0.0
    batches_processed = 0
    shard_progress = {}
    log_interval = context.run_config.get("log_interval")

    pbar = tqdm(trainloader, desc="Training")

    for inputs, targets, shard_id, rows_in_shard in pbar:
        if round_end_time:
            remaining = max(0, round_end_time - time.time())
            pbar.set_description(f"shard_{shard_id} ({remaining:.0f}s)")

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

        if round_end_time and time.time() >= round_end_time:
            log(INFO, f"Round time expired after batch {batches_processed}")
            break

    # Flush gradients
    if (batches_processed % gradient_accumulation_steps) != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    total_rows_processed = sum(shard_progress.values())
    actual_train_time = time.time() - round_start_time

    log(INFO, f"Client {client_id}: {batches_processed} batches, {total_rows_processed} rows, {actual_train_time:.2f}s")

    # Cleanup
    state_dict = model.state_dict()
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    new_cumulative_batches = cumulative_batches + batches_processed
    context.state["cumulative_batches"] = MetricRecord({"cumulative_batches": new_cumulative_batches})

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord({
                "client_id": client_id,
                "train_loss": avg_loss,
                "actual_train_time": actual_train_time,
                "batches_processed": batches_processed,
                "total_rows_processed": total_rows_processed,
                "shard_updates": [[sid, rows] for sid, rows in shard_progress.items()],
                "cumulative_batches": new_cumulative_batches,
            }),
        }),
        reply_to=msg,
    )
