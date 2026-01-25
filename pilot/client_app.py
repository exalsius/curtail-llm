import json
import math
import os
import sys
import time
from logging import INFO

import redis
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import ConfigRecord, log
from nanochat.common import get_base_dir
from pilot.model import get_model
from pilot.data import fl_shard_dataloader


app = ClientApp()


def _setup_ddp(rank, world_size, client_id):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355 + int(client_id))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def _gather_progress(rank, world_size, shard_progress, shard_total_rows):
    if world_size <= 1:
        return shard_progress, shard_total_rows

    local_progress_list = list(shard_progress.items())
    local_totals_list = list(shard_total_rows.items())
    
    if rank == 0:
        gathered_progress_lists = [None] * world_size
        gathered_totals_lists = [None] * world_size
        dist.gather_object(local_progress_list, gathered_progress_lists, dst=0)
        dist.gather_object(local_totals_list, gathered_totals_lists, dst=0)
    else:
        dist.gather_object(local_progress_list, dst=0)
        dist.gather_object(local_totals_list, dst=0)

    if rank == 0:
        global_shard_progress = {}
        global_shard_totals = {}
        for progress_list in gathered_progress_lists:
            for shard_id, current_row in progress_list:
                # Update only if this rank's progress is further
                if current_row > global_shard_progress.get(shard_id, -1):
                    global_shard_progress[shard_id] = current_row
        for totals_list in gathered_totals_lists:
            global_shard_totals.update(dict(totals_list))
        return global_shard_progress, global_shard_totals
    
    return {}, {}


def run_training_process(rank, world_size, msg, context, result_dict, round_start_time):
    client_id = context.node_config.get("partition-id", 0)
    node_name = context.node_config.get("name", f"client_{client_id}")
    config: ConfigRecord = msg.content["config"]

    experiment_start_time = int(config.get("experiment_start_time", time.time()))

    if world_size > 1:
        device = _setup_ddp(rank, world_size, client_id)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)  # Assuming single node
    os.environ["WORLD_SIZE"] = str(world_size)

    if config.get("debug_port_client") and client_id == 0 and rank == 0:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost",
            port=config["debug_port_client"],
            stdout_to_server=True,
            stderr_to_server=True,
        )

    if "cumulative_batches" in context.state:
        cumulative_batches = int(context.state["cumulative_batches"]["cumulative_batches"])
    else:
        cumulative_batches = 0

    device_batch_size = int(config["device_batch_size"])
    max_seq_len = int(config["max_seq_len"])
    total_batch_size = int(config["total_batch_size"])
    num_iterations = int(config["num_iterations"])
    warmup_ratio = float(config["warmup_ratio"])
    warmdown_ratio = float(config["warmdown_ratio"])
    final_lr_frac = float(config["final_lr_frac"])

    def get_lr_multiplier(it):
        warmup_iters = round(warmup_ratio * num_iterations)
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        if it <= num_iterations - warmdown_iters:
            return 1.0
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(it):
        return float(config["weight_decay_scaled"]) * (1 - it / num_iterations)

    tokens_per_fwdbwd = device_batch_size * max_seq_len * world_size
    gradient_accumulation_steps = total_batch_size // tokens_per_fwdbwd
    assert total_batch_size % tokens_per_fwdbwd == 0
    server_round = config.get("server_round", 0)
    redis_url = context.run_config["redis_url"]
    
    shard_assignments = list(zip(config.get("shard_ids", []), config.get("shard_starts", [])))
    global_tokens_processed_start = int(config.get("global_tokens_processed_start", 0))

    if rank == 0:
        log(INFO, f"Assigned {len(shard_assignments)} shards to {world_size} workers: {shard_assignments}")

    model = get_model(config, max_seq_len)
    # Clean state dict keys from `_orig_mod.` prefix (added by torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in msg.content["arrays"].to_torch_state_dict().items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    if sys.platform == "linux" and device.type == "cuda":
        if rank == 0: log(INFO, "Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=False)
    # No DDP wrapper: optimizers handle gradient sync via reduce_scatter/all_reduce
    trainloader = fl_shard_dataloader(
        shard_assignments=shard_assignments,
        batch_size=device_batch_size,
        sequence_length=max_seq_len,
        tokenizer_threads=context.run_config["tokenizer_threads"],
        tokenizer_batch_size=context.run_config["tokenizer_batch_size"],
        device=device.type if world_size == 1 else "cuda",
        rank=rank,
        world_size=world_size,
    )

    pubsub = redis.from_url(redis_url).pubsub()
    pubsub.subscribe(f"round:{server_round}:stop", f"{node_name}:stop")

    model.train()
    raw_model = getattr(model, "module", model)
    lr_scale = float(config["batch_lr_scale"])
    optimizers = raw_model.setup_optimizers(
        unembedding_lr=float(config["unembedding_lr"]) * lr_scale,
        embedding_lr=float(config["embedding_lr"]) * lr_scale,
        matrix_lr=float(config["matrix_lr"]) * lr_scale,
        weight_decay=float(config["weight_decay_scaled"]),
        adam_betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
        scalar_lr=float(config["scalar_lr"]) * lr_scale,
    )
    adamw_optimizer, muon_optimizer = optimizers

    # Setup autocast context
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext

        autocast_ctx = nullcontext()

    batches_processed = 0
    shard_progress = {}  # Tracks absolute current_row
    shard_total_rows = {} # Tracks total_rows per shard
    log_interval = context.run_config["log_interval"]
    last_log_time = time.time()
    redis_client = redis.from_url(redis_url)
    log_key = f"logs:{node_name}"
    redis_client.delete(log_key)

    iterator = iter(trainloader)

    # Prefetch the first batch
    try:
        inputs, targets, shard_id, current_row, total_rows = next(iterator)
    except StopIteration:
        log(INFO, f"Rank {rank}: Dataloader is empty, skipping training loop.")
        iterator = None # Mark as exhausted

    while iterator:
        with autocast_ctx:
            loss = model(inputs, targets=targets)
            loss = loss / gradient_accumulation_steps

        loss.backward()

        # Capture state of the batch we just finished before prefetching overwrites it
        prev_shard_id = shard_id
        prev_total_rows = total_rows

        # Prefetch the next batch while the GPU is busy
        try:
            inputs, targets, shard_id, current_row, total_rows = next(iterator)
        except StopIteration:
            iterator = None # Mark as exhausted to exit loop

        # If the shard ID changed, it means the previous shard is fully consumed.
        # We explicitly mark it as complete (max rows) to prevent "stuck at 99%" issues.
        if iterator and shard_id != prev_shard_id:
            shard_progress[prev_shard_id] = prev_total_rows
            shard_total_rows[prev_shard_id] = prev_total_rows

        batches_processed += 1

        if batches_processed % gradient_accumulation_steps == 0:
            step_count = batches_processed // gradient_accumulation_steps
            current_step = (global_tokens_processed_start // total_batch_size) + step_count - 1

            lrm = get_lr_multiplier(current_step)
            muon_momentum = get_muon_momentum(current_step)
            muon_weight_decay = get_weight_decay(current_step)

            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lrm

            for group in muon_optimizer.param_groups:
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)

            if rank == 0 and log_interval and step_count % log_interval == 0:
                log_interval_duration = time.time() - last_log_time
                loss_scalar = loss.item() * gradient_accumulation_steps

                tokens_processed_locally = batches_processed * device_batch_size * max_seq_len * world_size
                total_global_tokens = global_tokens_processed_start + tokens_processed_locally
                current_step = total_global_tokens // total_batch_size

                tokens_processed_since_last_log = (
                    log_interval * gradient_accumulation_steps * device_batch_size * max_seq_len * world_size
                )
                tok_per_sec = tokens_processed_since_last_log / log_interval_duration

                flops_per_sec = raw_model.estimate_flops() * tok_per_sec
                promised_flops_per_sec_a100 = 312e12
                mfu = (100 * flops_per_sec / (promised_flops_per_sec_a100 * world_size)) if world_size > 0 else 0.0

                log(
                    INFO,
                    f"Step {current_step} ({log_interval_duration*1000:.0f}ms) | Shard: {shard_id} (row {current_row}/{total_rows}) | "
                    f"Loss: {loss_scalar:.4f} | Tok/sec: {int(tok_per_sec):,} | MFU: {mfu:.2f}% | "
                    f"Round duration: {(time.time() - round_start_time):.0f}s",
                )

                log_entry = {
                    "step": current_step,
                    "time": int(time.time() - experiment_start_time),
                    f"client_{client_id}/train_loss": loss_scalar,
                    f"client_{client_id}/train_ppl": math.exp(loss_scalar),
                    f"client_{client_id}/lrm": lrm,
                    f"client_{client_id}/momentum": muon_momentum,
                    f"client_{client_id}/weight_decay": muon_weight_decay,
                    f"client_{client_id}/tok_per_sec": tok_per_sec,
                    f"client_{client_id}/mfu": mfu,
                }
                redis_client.rpush(log_key, json.dumps(log_entry))
                last_log_time = time.time()

            redis_msg = pubsub.get_message(timeout=0)
            if redis_msg and redis_msg["type"] == "message":
                if rank == 0:
                    log(INFO, f"🛑 Stop signal received ({redis_msg['data'].decode('utf-8')}) after batch {batches_processed}")
                break

        shard_progress[shard_id] = current_row
        shard_total_rows[shard_id] = total_rows
        if not iterator:
            break

    if batches_processed % gradient_accumulation_steps != 0:
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    avg_loss = loss.item() * gradient_accumulation_steps if batches_processed > 0 else 0.0
    shard_progress, shard_total_rows = _gather_progress(rank, world_size, shard_progress, shard_total_rows)

    if rank == 0:
        log(INFO, f"Client {client_id}: {batches_processed} batches")

        new_cumulative_batches = cumulative_batches + batches_processed
        context.state["cumulative_batches"] = MetricRecord(
            {"cumulative_batches": new_cumulative_batches}
        )

        shard_ids = list(shard_progress.keys())
        shard_rows = list(shard_progress.values())
        shard_totals = [shard_total_rows.get(s_id, 0) for s_id in shard_ids]

        unwrapped_model = model
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module
        if hasattr(unwrapped_model, "_orig_mod"):
            unwrapped_model = unwrapped_model._orig_mod
        state_dict = unwrapped_model.state_dict()

        tokens_processed_round = (
            batches_processed * device_batch_size * max_seq_len * world_size
        )

        result_dict["message"] = Message(
            content=RecordDict(
                {
                    "arrays": ArrayRecord(state_dict),
                    "metrics": MetricRecord(
                        {
                            "client_id": client_id,
                            "train_loss": avg_loss,
                            "batches_processed": batches_processed,
                            "num_tokens_processed": tokens_processed_round,
                            "shard_ids": shard_ids,
                            "shard_rows": shard_rows,
                            "shard_totals": shard_totals,
                            "cumulative_batches": new_cumulative_batches,
                        }
                    ),
                }
            ),
            reply_to=msg,
        )


@app.train()
def train(msg: Message, context: Context):
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    round_start_time = time.time()

    if num_gpus > 1:
        log(INFO, f"Spawning {num_gpus} processes for DDP training")
        manager = mp.Manager()
        result_dict = manager.dict()
        mp.spawn(
            run_training_process,
            args=(num_gpus, msg, context, result_dict, round_start_time),
            nprocs=num_gpus,
            join=True,
        )
        if "message" not in result_dict:
            raise RuntimeError("Training failed to produce a result message.")
        return result_dict["message"]

    result_dict = {}
    run_training_process(0, 1, msg, context, result_dict, round_start_time)
    return result_dict["message"]


@app.query()
def query(msg: Message, context: Context):
    node_name = context.node_config.get(
        "name", f"client_{context.node_config.get('partition-id', 0)}"
    )
    log(INFO, f"HI! I'm CLIENT {node_name}")
    return Message(
        content=RecordDict({"config": ConfigRecord({"name": node_name})}),
        reply_to=msg,
    )