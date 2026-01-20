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
import pyarrow.parquet as pq
from flwr.common import ConfigRecord, log
from torch.nn.parallel import DistributedDataParallel as DDP

from nanochat.common import get_base_dir
from pilot.model import get_model
from pilot.data import fl_shard_dataloader


app = ClientApp()


def run_training_process(rank, world_size, msg, context, result_dict, round_start_time):
    client_id = context.node_config.get("partition-id", 0)
    node_name = context.node_config.get("name", f"client_{client_id}")
    config: ConfigRecord = msg.content["config"]

    experiment_start_time = int(config.get("experiment_start_time", time.time()))

    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12355 + int(client_id))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set env vars for nanochat helpers (get_dist_info)
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
        cumulative_batches = int(
            context.state["cumulative_batches"]["cumulative_batches"]
        )
    else:
        cumulative_batches = 0

    # Extract config
    device_batch_size = int(config["device_batch_size"])
    max_seq_len = int(config["max_seq_len"])
    total_batch_size = int(config["total_batch_size"])

    matrix_lr = float(config["matrix_lr"])
    embedding_lr = float(config["embedding_lr"])
    unembedding_lr = float(config["unembedding_lr"])
    scalar_lr = float(config["scalar_lr"])
    weight_decay_base = float(config["weight_decay"])
    adam_beta1 = float(config["adam_beta1"])
    adam_beta2 = float(config["adam_beta2"])

    # --- Scheduler ---
    num_iterations = int(config["num_iterations"])
    warmup_ratio = float(config["warmup_ratio"])
    warmdown_ratio = float(config["warmdown_ratio"])
    final_lr_frac = float(config["final_lr_frac"])

    def get_lr_multiplier(it):
        warmup_iters = round(warmup_ratio * num_iterations)
        warmdown_iters = round(warmdown_ratio * num_iterations)
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        elif it <= num_iterations - warmdown_iters:
            return 1.0
        else:
            progress = (num_iterations - it) / warmdown_iters
            return progress * 1.0 + (1 - progress) * final_lr_frac

    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    # Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2
    depth = int(config["n_layer"])
    weight_decay_scaled = weight_decay_base * (12 / depth) ** 2
    if depth != 12:
        log(
            INFO,
            f"Scaling weight decay from {weight_decay_base:.6f} to {weight_decay_scaled:.6f} for depth {depth}",
        )
    def get_weight_decay(it):
        return weight_decay_scaled * (1 - it / num_iterations)

    # Batch size scaling for learning rates
    batch_lr_scale = 1.0
    reference_batch_size = 2**19
    batch_ratio = total_batch_size / reference_batch_size
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio**0.5
        log(
            INFO,
            f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}",
        )
    # --- End Scheduler ---

    # Calculate gradient accumulation steps
    # Note: total_batch_size is global, so we divide by (device_batch_size * world_size)
    tokens_per_fwdbwd = device_batch_size * max_seq_len * world_size
    assert total_batch_size % tokens_per_fwdbwd == 0
    gradient_accumulation_steps = total_batch_size // tokens_per_fwdbwd

    server_round = config.get("server_round", 0)

    redis_url = context.run_config["redis_url"]
    shard_ids = config.get("shard_ids", [])
    shard_starts = config.get("shard_starts", [])
    
    # Create a single list of assignments for the whole client
    all_shard_assignments = list(zip(shard_ids, shard_starts))
    
    # All ranks get the same full list of assignments. The dataloader is responsible
    # for ensuring each rank processes a unique subset of data *within* each shard.
    shard_assignments = all_shard_assignments

    # Get total_rows for assigned shards (only on rank 0 to avoid redundant reads)
    shard_total_rows = {}
    if rank == 0:
        base_dir = get_base_dir()
        data_dir = os.path.join(base_dir, "base_data")
        for shard_id, _ in shard_assignments:
            filepath = os.path.join(data_dir, f"shard_{shard_id:05d}.parquet")
            if os.path.exists(filepath):
                pf = pq.ParquetFile(filepath)
                shard_total_rows[shard_id] = pf.metadata.num_rows

    # Scheduler Config
    # LRs and momentum are now scheduled by the server per round
    muon_momentum = float(config.get("muon_momentum", 0.95))
    global_tokens_processed_start = int(config.get("global_tokens_processed_start", 0))

    if rank == 0:
        log(
            INFO,
            f"Assigned {len(shard_assignments)} shards to {world_size} workers: {shard_assignments}",
        )

    # Load model
    model = get_model(config, max_seq_len)
    # Clean state dict keys from `_orig_mod.` prefix (added by torch.compile)
    state_dict = msg.content["arrays"].to_torch_state_dict()
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # Compile model (Linux + CUDA only)
    if sys.platform == "linux" and device.type == "cuda":
        if rank == 0:
            log(INFO, "Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=False)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Create dataloader
    trainloader = fl_shard_dataloader(
        shard_assignments=shard_assignments,
        B=device_batch_size,
        T=max_seq_len,
        tokenizer_threads=context.run_config["tokenizer_threads"],
        tokenizer_batch_size=context.run_config["tokenizer_batch_size"],
        device=device.type if world_size == 1 else "cuda",
        rank=rank,
        world_size=world_size,
    )

    # Setup Redis pubsub for stop signal
    # IMPORTANT: If a stop signal is published before the client successfully subscribes
    # to these channels, the signal will be missed by this client.
    pubsub = redis.from_url(redis_url).pubsub()
    pubsub.subscribe(f"round:{server_round}:stop")
    pubsub.subscribe(f"{node_name}:stop")

    # Training loop
    model.train()

    # Initialize optimizers (Muon + AdamW)
    raw_model = getattr(model, "module", model)

    # Flop estimation
    num_flops_per_token = raw_model.estimate_flops()
    promised_flops_per_sec_a100 = 312e12 * world_size

    optimizers = raw_model.setup_optimizers(
        unembedding_lr=unembedding_lr * batch_lr_scale,
        embedding_lr=embedding_lr * batch_lr_scale,
        matrix_lr=matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled, # Note: this is base decay, per-step is applied later
        adam_betas=(adam_beta1, adam_beta2),
        scalar_lr=scalar_lr * batch_lr_scale,
    )
    adamw_optimizer, muon_optimizer = optimizers

    # Per-step scheduling is now handled in the training loop.

    # Setup autocast context
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext

        autocast_ctx = nullcontext()

    total_loss = 0.0
    batches_processed = 0
    shard_progress = {}  # Tracks absolute current_row
    log_interval = context.run_config["log_interval"]
    last_log_time = time.time()

    # Setup Redis for logging
    redis_client = redis.from_url(redis_url)
    log_key = f"logs:{node_name}"
    redis_client.delete(log_key)

    iterator = iter(trainloader)

    # Prefetch the first batch
    try:
        inputs, targets, shard_id, current_row, shard_progress_val = next(iterator)
    except StopIteration:
        log(INFO, f"Rank {rank}: Dataloader is empty, skipping training loop.")
        iterator = None # Mark as exhausted

    while iterator:
        # Note: 'inputs' and 'targets' are from the *previous* iteration's prefetch
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast_ctx:
            loss = model(inputs, targets=targets)
            loss = loss / gradient_accumulation_steps

        loss.backward()

        # Prefetch the next batch while the GPU is busy
        try:
            inputs, targets, shard_id, current_row, shard_progress_val = next(iterator)
        except StopIteration:
            iterator = None # Mark as exhausted to exit loop

        batches_processed += 1

        if batches_processed % gradient_accumulation_steps == 0:
            # Calculate current global step
            local_steps_done = (batches_processed // gradient_accumulation_steps) - 1
            start_step = global_tokens_processed_start // total_batch_size
            current_step = start_step + local_steps_done

            # Get scheduled hyperparameter values for this step
            lrm = get_lr_multiplier(current_step)
            muon_momentum = get_muon_momentum(current_step)
            muon_weight_decay = get_weight_decay(current_step)

            # Update optimizers with scheduled values
            for opt in optimizers:
                for group in opt.param_groups:
                    # The optimizers are initialized with initial_lr by setup_optimizers
                    group["lr"] = group["initial_lr"] * lrm

            for group in muon_optimizer.param_groups:
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)

            step_count = batches_processed // gradient_accumulation_steps
            # This is the main Redis log, happens per optimizer step
            if rank == 0 and log_interval and step_count % log_interval == 0:
                # Note: dt here is for the Redis logging interval, not the console log
                log_interval_duration = time.time() - last_log_time

                # (logging calculations)
                tokens_processed_locally = (
                    batches_processed * device_batch_size * max_seq_len * world_size
                )
                total_global_tokens = (
                    global_tokens_processed_start + tokens_processed_locally
                )
                current_step = total_global_tokens // total_batch_size
                
                # We calculate tok/sec for redis based on its own interval
                redis_log_steps = log_interval
                tokens_processed_since_redis_log = (
                    redis_log_steps
                    * gradient_accumulation_steps
                    * device_batch_size
                    * max_seq_len
                    * world_size
                )
                tok_per_sec = tokens_processed_since_redis_log / log_interval_duration

                flops_per_sec = num_flops_per_token * tok_per_sec
                mfu = (
                    100 * flops_per_sec / promised_flops_per_sec_a100
                    if promised_flops_per_sec_a100 > 0
                    else 0.0
                )

                eff_unembedding_lr = adamw_optimizer.param_groups[0]["lr"]
                eff_embedding_lr = adamw_optimizer.param_groups[1]["lr"]
                eff_scalar_lr = adamw_optimizer.param_groups[3]["lr"]
                eff_matrix_lr = muon_optimizer.param_groups[0]["lr"]
                loss_scalar = loss.item() * gradient_accumulation_steps

                # Console log for a quick pulse-check
                log(
                    INFO,
                    f"Step {current_step} | Loss: {loss_scalar:.4f} | Tok/sec: {int(tok_per_sec):,} | MFU: {mfu:.2f}% | Step duration: {log_interval_duration*1000:.0f}ms | Round duration: {(time.time() - round_start_time):.0f}s",
                )

                log_entry = {
                    "step": current_step,
                    "time": int(time.time() - experiment_start_time),
                    f"client_{client_id}/train_loss": loss_scalar,
                    f"client_{client_id}/train_ppl": math.exp(loss_scalar),
                    f"client_{client_id}/lr_multiplier": lrm,
                    f"client_{client_id}/eff_matrix_lr": eff_matrix_lr,
                    f"client_{client_id}/eff_embedding_lr": eff_embedding_lr,
                    f"client_{client_id}/eff_unembedding_lr": eff_unembedding_lr,
                    f"client_{client_id}/eff_scalar_lr": eff_scalar_lr,
                    f"client_{client_id}/momentum": muon_momentum,
                    f"client_{client_id}/weight_decay": muon_weight_decay,
                    f"client_{client_id}/tok_per_sec": tok_per_sec,
                    f"client_{client_id}/mfu": mfu,
                }
                redis_client.rpush(log_key, json.dumps(log_entry))
                last_log_time = time.time()

        total_loss += loss.item() * gradient_accumulation_steps

        # Track progress
        shard_progress[shard_id] = current_row

        # Check for stop signal (non-blocking)
        redis_msg = pubsub.get_message(timeout=0)
        if redis_msg and redis_msg["type"] == "message":
            log(
                INFO,
                f"Rank {rank}: Stop signal received ({redis_msg['data'].decode('utf-8')}) after batch {batches_processed}",
            )
            break
        
        # Break loop if dataloader is exhausted
        if not iterator:
            break


    # Flush gradients
    if (batches_processed % gradient_accumulation_steps) != 0:
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0

    # Gather shard progress from all ranks if DDP is used
    if world_size > 1:
        # Convert local shard_progress dict to list of tuples for gathering
        local_progress_list = [(s_id, c_row) for s_id, c_row in shard_progress.items()]
        
        # Gather all local_progress_lists on rank 0
        if rank == 0:
            gathered_progress_lists = [None for _ in range(world_size)]
            dist.gather_object(local_progress_list, gathered_progress_lists, dst=0)
        else:
            dist.gather_object(local_progress_list, dst=0)

        # On rank 0, combine all gathered progress into a single dictionary
        if rank == 0:
            global_shard_progress = {}
            for progress_list in gathered_progress_lists:
                for shard_id, current_row in progress_list:
                    # Update only if this rank's progress for this shard is further
                    if current_row > global_shard_progress.get(shard_id, -1):
                        global_shard_progress[shard_id] = current_row
            shard_progress = global_shard_progress # Update the shard_progress dict for rank 0

        # Destroy process group after gathering progress
        dist.destroy_process_group()

    if rank == 0:
        log(INFO, f"Client {client_id}: {batches_processed} batches")

        new_cumulative_batches = cumulative_batches + batches_processed
        context.state["cumulative_batches"] = MetricRecord(
            {"cumulative_batches": new_cumulative_batches}
        )

        # Convert shard_progress to two parallel lists
        shard_ids = list(shard_progress.keys())
        shard_rows = list(shard_progress.values())
        shard_totals = [shard_total_rows.get(s_id, 0) for s_id in shard_ids]

        # Unwrap model from DDP and torch.compile wrappers
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
    # Detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0

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
        if "message" in result_dict:
            return result_dict["message"]
        else:
            raise RuntimeError("Training failed to produce a result message.")
    else:
        # Run single process
        # Use a dummy dict to capture result
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
