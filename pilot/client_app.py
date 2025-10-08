import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pilot.models import get_model
from pilot.data import get_train_loader
from pilot.train_test import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on queue-assigned data."""
    partition_id = context.node_config["partition-id"]  # Worker index (0, 1, 2...)

    client_debug_port = context.run_config["client_debug_port"]
    if client_debug_port and partition_id == 0:
        print("[Client 0] Debug mode enabled...")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=client_debug_port, stdout_to_server=True, stderr_to_server=True)

    # Load the model and initialize it with the received weights
    model_type = context.run_config["model-type"]
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract queue assignment for this worker
    queue_id, epoch, start_batch_idx = msg.content["config"]["assignments"][partition_id]

    trainloader = get_train_loader(
        dataset_name=msg.content["config"]["dataset-name"],
        shard_id=queue_id,
        num_shards=msg.content["config"]["num-queues"],
        start_batch_idx=start_batch_idx,
        epoch=epoch,
        batch_size=context.run_config["batch-size"],
    )

    # Print training information
    print(f"[Worker {partition_id}] Assigned to Queue {queue_id}")
    print(f"[Worker {partition_id}] Starting at epoch {epoch}, batch {start_batch_idx}")
    print(f"[Worker {partition_id}] Device: {device}")

    # Call the training function with progress tracking
    train_loss, batches_processed, epochs_completed = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Calculate final position
    final_batch_idx = start_batch_idx + batches_processed
    final_epoch = epoch + epochs_completed

    print(f"[Worker {partition_id}] Completed: processed {batches_processed} batches")
    print(f"[Worker {partition_id}] Final position: epoch {final_epoch}, batch {final_batch_idx}")

    # Construct and return reply Message with queue state
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": batches_processed * batch_size,
        "queue-id": queue_id,
        "final-batch-idx": final_batch_idx,
        "final-epoch": final_epoch,
        "batches-processed": batches_processed,
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
