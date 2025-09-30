import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pilot.models import get_model
from pilot.data import get_data_loaders
from pilot.train_test import test as test_fn
from pilot.train_test import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    if context.run_config["debug"] and context.node_config["partition-id"] == 0:
        print("[Client 0] Debug mode enabled...")
        try:
            import pydevd_pycharm
            pydevd_pycharm.settrace('localhost', port=5679, stdout_to_server=True, stderr_to_server=True)
        except ImportError:
            print("[Client 0] pydevd_pycharm not available, skipping debug setup")

    # Load the model and initialize it with the received weights
    model_type = context.run_config.get("model-type", "resnet18")
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = get_data_loaders(
        data_type="cifar10",
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        model_type=model_type
    )

    # Print training information
    print(f"[Client {partition_id}] Training on {len(trainloader.dataset)} samples")
    print(f"[Client {partition_id}] Device: {device}")
    print(f"[Client {partition_id}] Batch size: {batch_size}, Epochs: {context.run_config['local-epochs']}")

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model_type = context.run_config.get("model-type", "resnet18")
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = get_data_loaders(
        data_type="cifar10",
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
        model_type=model_type
    )

    # Print evaluation information
    print(f"[Client {partition_id}] Evaluating on {len(valloader.dataset)} samples")

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
