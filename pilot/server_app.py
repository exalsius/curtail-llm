import asyncio
from collections import defaultdict
from logging import INFO

import redis
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.common import log
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp

from pilot.model import get_model
from pilot.strategy import Client, PilotAvg

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    lr: float = context.run_config["lr"]
    dataset_name: str = context.run_config["dataset_name"]
    num_shards: int = context.run_config["num_shards"]
    batch_size: int = context.run_config["batch_size"]
    model_type: str = context.run_config["model_type"]
    redis_url: str = context.run_config["redis_url"]
    round_min_duration: int = context.run_config["round_min_duration"]

    debug_port_server: int = context.run_config.get("debug_port_server", None)
    if debug_port_server:
        print("[Server] Debug mode enabled...")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=debug_port_server, stdout_to_server=True, stderr_to_server=True)

    # Initialize wandb
    wandb_project: str = context.run_config["wandb_project"]
    wandb_entity: str | None = context.run_config.get("wandb_entity")

    # Create run name with hyperparameters
    run_name = f"sh{num_shards},bs{batch_size}"

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        group=run_name,  # Group all runs together
        config={
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_shards": num_shards,
            "model_type": model_type,
            "dataset_name": dataset_name,
        }
    )
    log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)

    redis_client = redis.from_url(redis_url)

    # Create clients from config
    clients = _clients_from_run_config(context.run_config)
    log(INFO, "Configured clients: %s", list(clients.keys()))

    strategy = PilotAvg(
        clients=clients,
        dataset_name=dataset_name,
        num_shards=num_shards,
        debug_port_client=context.run_config.get("debug_port_client", None),
        redis_url=redis_url,
        redis_client=redis_client,
        round_min_duration=round_min_duration,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_group=run_name,
    )

    # Load global model
    max_length = context.run_config.get("max_length", 2048)
    global_model = get_model(model_type, max_length)

    asyncio.run(strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord(dict(lr=lr)),
    ))

    # Finish wandb run
    wandb.finish()
    log(INFO, "Wandb run finished")


def _clients_from_run_config(flat_dict: dict) -> dict[str, Client]:
    fields_by_client = defaultdict(dict)
    for key, value in flat_dict.items():
        if key.startswith("clients."):
            _, client_name, field = key.split(".")
            fields_by_client[client_name][field] = value
    return {client_name: Client(name=client_name, **fields) for client_name, fields in fields_by_client.items()}
