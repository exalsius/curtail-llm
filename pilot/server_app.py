from collections import defaultdict
from logging import INFO

import redis
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp

from pilot.model import get_model
from pilot.provisioner import ExlsProvisioner
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
        print("[Server] Debug mode enabled...")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=debug_port_server, stdout_to_server=True, stderr_to_server=True)

    # Initialize wandb
    wandb_project: str = context.run_config["wandb_project"]

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
    clients, node_id_mapping = _clients_from_run_config(context.run_config)
    log(INFO, "Configured clients: %s", list(clients.keys()))

    exls_cluster_id: str = context.run_config.get("exls_cluster_id")
    provisioner = ExlsProvisioner(exls_cluster_id, node_id_mapping) if exls_cluster_id else None

    strategy = PilotAvg(
        clients=clients,
        dataset_name=context.run_config["dataset_name"],
        num_shards=num_shards,
        debug_port_client=context.run_config.get("debug_port_client", None),
        redis_url=context.run_config["redis_url"],
        round_min_duration=context.run_config["round_min_duration"],
        provisioner=provisioner,
        mci_api_url=context.run_config["mci_api_url"],
        wandb_project=wandb_project,
    )

    # Extract scheduler config
    train_config_dict = {
        "matrix_lr": float(context.run_config["matrix_lr"]),
        "embedding_lr": float(context.run_config["embedding_lr"]),
        "unembedding_lr": float(context.run_config["unembedding_lr"]),
        "scalar_lr": float(context.run_config["scalar_lr"]),
        "weight_decay": float(context.run_config["weight_decay"]),
        "adam_beta1": float(context.run_config["adam_beta1"]),
        "adam_beta2": float(context.run_config["adam_beta2"]),
        "num_iterations": int(context.run_config["num_iterations"]),
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


def _clients_from_run_config(flat_dict: dict) -> tuple[dict[str, Client], dict[str, str]]:
    fields_by_client = defaultdict(dict)
    for key, value in flat_dict.items():
        if key.startswith("clients."):
            _, client_name, field = key.split(".")
            fields_by_client[client_name][field] = value

    clients = {}
    node_id_mapping = {}
    redis_url = flat_dict["redis_url"]
    for client_name, fields in fields_by_client.items():
        exls_node_id = fields.pop("exls_node_id")
        node_id_mapping[client_name] = exls_node_id
        clients[client_name] = Client(name=client_name, redis_url=redis_url, **fields)

    return clients, node_id_mapping
