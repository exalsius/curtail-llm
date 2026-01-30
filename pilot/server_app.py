from logging import INFO

import redis
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp

from pilot.model import get_model
from pilot.provisioner import SubprocessProvisioner
from pilot.strategy import Client, PilotAvg

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_shards: int = context.run_config["num_shards"]
    device_batch_size: int = context.run_config["device_batch_size"]
    total_batch_size: int = context.run_config["total_batch_size"]
    max_seq_len: int = context.run_config["max_seq_len"]

    debug_port_server: int = context.run_config.get("debug_port_server", None)
    if debug_port_server:
        log(INFO, "Debug mode enabled")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=debug_port_server, stdout_to_server=True, stderr_to_server=True)

    # Initialize wandb
    wandb_project: str = context.run_config["wandb_project"]
    num_iterations: int = context.run_config["num_iterations"]

    # Create run name with hyperparameters
    run_name = f"bs{device_batch_size},tbs{total_batch_size},seq{max_seq_len}"

    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "matrix_lr": context.run_config["matrix_lr"],
            "embedding_lr": context.run_config["embedding_lr"],
            "unembedding_lr": context.run_config["unembedding_lr"],
            "device_batch_size": device_batch_size,
            "total_batch_size": total_batch_size,
            "max_seq_len": max_seq_len,
            "num_shards": num_shards,
        }
    )
    log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)

    redis_url: str = context.run_config["redis_url"]
    redis_client = redis.from_url(redis_url)
    redis_client.flushdb()
    log(INFO, "Redis DB flushed.")

    # Create clients and provisioner from config
    client_names = [c.strip() for c in context.run_config["clients"].split(",")]
    clients = {name: Client(name=name, redis_url=redis_url) for name in client_names}
    log(INFO, "Configured clients: %s", list(clients.keys()))

    curtailment_threshold: float = context.run_config.get("curtailment_threshold", 100)
    local_provisioning: bool = context.run_config.get("local_provisioning", False)

    provisioner = None
    if local_provisioning:
        # Default superlink address for local deployment
        provisioner = SubprocessProvisioner(superlink_address="127.0.0.1:9092")
        log(INFO, "Using local SubprocessProvisioner")

    strategy = PilotAvg(
        clients=clients,
        dataset_name=context.run_config["dataset_name"],
        num_shards=num_shards,
        debug_port_client=context.run_config.get("debug_port_client", None),
        redis_url=redis_url,
        round_min_duration=context.run_config["round_min_duration"],
        provisioner=provisioner,
        mci_api_url=context.run_config["mci_api_url"],
        curtailment_threshold=curtailment_threshold,
        wandb_project=wandb_project,
    )

    # Calculate derived parameters
    # Batch size scaling for learning rates
    batch_lr_scale = 1.0
    reference_batch_size = 2**19
    batch_ratio = total_batch_size / reference_batch_size
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio**0.5
        log(INFO, f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

    # Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2
    depth = int(context.run_config["n_layer"])
    weight_decay_base = float(context.run_config["weight_decay"])
    weight_decay_scaled = weight_decay_base * (12 / depth) ** 2
    if depth != 12:
        log(INFO, f"Scaling weight decay from {weight_decay_base:.6f} to {weight_decay_scaled:.6f} for depth {depth}")

    # Extract scheduler config
    train_config_dict = {
        "matrix_lr": float(context.run_config["matrix_lr"]),
        "embedding_lr": float(context.run_config["embedding_lr"]),
        "unembedding_lr": float(context.run_config["unembedding_lr"]),
        "scalar_lr": float(context.run_config["scalar_lr"]),
        "weight_decay": float(context.run_config["weight_decay"]),
        "adam_beta1": float(context.run_config["adam_beta1"]),
        "adam_beta2": float(context.run_config["adam_beta2"]),
        "num_iterations": num_iterations,
        "warmup_ratio": float(context.run_config["warmup_ratio"]),
        "warmdown_ratio": float(context.run_config["warmdown_ratio"]),
        "final_lr_frac": float(context.run_config["final_lr_frac"]),
        "total_batch_size": total_batch_size,
        "device_batch_size": device_batch_size,
        "max_seq_len": max_seq_len,
        "vocab_size": int(context.run_config["vocab_size"]),
        "n_layer": int(context.run_config["n_layer"]),
        "n_embd": int(context.run_config["n_embd"]),
        "n_head": int(context.run_config["n_head"]),
        "n_kv_head": int(context.run_config["n_kv_head"]),
        "batch_lr_scale": batch_lr_scale,
        "weight_decay_scaled": weight_decay_scaled,
    }

    # Load global model
    global_model = get_model(train_config_dict, max_seq_len)

    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord(train_config_dict),
    )

    # Finish wandb run
    wandb.finish()
    log(INFO, "Wandb run finished")
