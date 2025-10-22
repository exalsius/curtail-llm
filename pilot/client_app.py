import itertools
import random
from logging import INFO

import torch
import wandb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import ConfigRecord, log

from pilot.models import get_model
from pilot.data import get_train_loader
from pilot.train_test import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on queue-assigned data."""
    client_id = context.node_config["partition-id"]
    config: ConfigRecord = msg.content["config"]

    client_debug_port = config.get("client_debug_port", None)
    if client_debug_port and client_id == 0:
        print("[Client 0] Debug mode enabled...")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=client_debug_port, stdout_to_server=True, stderr_to_server=True)

    # Load the model and initialize it with the received weights
    model_type = context.run_config["model_type"]
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_name: str = config["dataset_name"]
    worker_seed: int = config["worker_seed"]
    epoch: int = config["epoch"]
    batch_within_epoch: int = config["batch_within_epoch"]
    server_round: int = config["server_round"]
    num_batches: int = 80 + int(40 * random.random())  # TODO determine this via time

    # Get train loader
    trainloader = get_train_loader(
        dataset_name=dataset_name,
        worker_seed=worker_seed,
        start_epoch=epoch,
        start_batch=batch_within_epoch,
        batch_size=context.run_config["batch_size"],
    )

    # Print training information
    shuffle_seed = worker_seed + epoch
    log(INFO, f"[Client {client_id}] Worker seed: {worker_seed}, Epoch: {epoch}, Batch: {batch_within_epoch}")
    log(INFO, f"[Client {client_id}] Shuffle seed: {shuffle_seed}, Processing {num_batches} batches")
    log(INFO, f"[Client {client_id}] Device: {device}")

    # Call the training function with progress tracking
    train_loss, batches_processed = train_fn(
        model,
        trainloader,
        num_batches=num_batches,
        lr=config["lr"],
        device=device,
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Construct and return reply Message with client metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "client_id": client_id,
        "train_loss": train_loss,
        "batches_processed": batches_processed,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


# NOTE: For now we only evaluate globally

# @app.evaluate()
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""
#
#     # Load the model and initialize it with the received weights
#     model_type = context.run_config.get("model-type", "resnet18")
#     model = get_model(model_type)
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     batch_size = context.run_config["batch-size"]
#     _, valloader = get_data_loaders(
#         data_type="cifar10",
#         partition_id=partition_id,
#         num_partitions=num_partitions,
#         batch_size=batch_size,
#         model_type=model_type
#     )
#
#     # Print evaluation information
#     print(f"[Client {partition_id}] Evaluating on {len(valloader.dataset)} samples")
#
#     # Call the evaluation function
#     eval_loss, eval_acc = test_fn(
#         model,
#         valloader,
#         device,
#     )
#
#     # Construct and return reply Message
#     metrics = {
#         "eval_loss": eval_loss,
#         "eval_acc": eval_acc,
#         "num-examples": len(valloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)
