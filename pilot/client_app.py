from logging import INFO

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import ConfigRecord, log

import pilot.vision as vision
import pilot.llm as llm
import pilot.medical as medical
import pilot.nanochat_fl as nanochat

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on queue-assigned data."""
    client_id = context.node_config["partition-id"]
    config: ConfigRecord = msg.content["config"]

    debug_port_client = config.get("debug_port_client", None)
    if debug_port_client and client_id == 0:
        print("[Client 0] Debug mode enabled...")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=debug_port_client, stdout_to_server=True, stderr_to_server=True)

    # Get cumulative batches from client state (persists across rounds)
    # context.state stores MetricRecord, so we need to extract the value
    if "cumulative_batches" in context.state:
        cumulative_batches = int(context.state["cumulative_batches"]["cumulative_batches"])
    else:
        cumulative_batches = 0

    model_type = context.run_config["model_type"]
    task_type = context.run_config["task_type"]

    log(INFO, f"[Client {client_id}] Task: {task_type}, Model: {model_type}")
    log(INFO, f"[Client {client_id}] Shard {config['shard_id']}/{config['num_shards']}, "
              f"batch {config['processed_batches']}, cumulative: {cumulative_batches}")

    if task_type == "vision":
        state_dict, train_loss, batches_processed = vision.train_client(msg, config, context, cumulative_batches)
        metrics_payload = {"train_loss": train_loss}
    elif task_type == "llm":
        # LLM returns a metrics dict for richer reporting
        state_dict, metrics_payload, batches_processed = llm.train_client(msg, config, context, cumulative_batches)
    elif task_type == "medical":
        # Medical returns same format as LLM (metrics dict for richer reporting)
        state_dict, metrics_payload, batches_processed = medical.train_client(msg, config, context, cumulative_batches)
    elif task_type == "nanochat":
        # Nanochat returns same format as LLM (metrics dict for richer reporting)
        state_dict, metrics_payload, batches_processed = nanochat.train_client(msg, config, context, cumulative_batches)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Update cumulative batches in client state
    # Store as MetricRecord since context.state only accepts Record types
    new_cumulative_batches = cumulative_batches + batches_processed
    context.state["cumulative_batches"] = MetricRecord({"cumulative_batches": new_cumulative_batches})

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord({
                "client_id": client_id,
                "shard_id": config["shard_id"],
                "batches_processed": batches_processed,
                "cumulative_batches": new_cumulative_batches,
                **metrics_payload,
            }),
        }),
        reply_to=msg,
    )
