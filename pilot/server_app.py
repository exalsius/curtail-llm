from collections.abc import Iterator
from logging import INFO
from typing import Iterable, Optional

import pandas as pd
import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.common import log, RecordDict, MessageType
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    validate_message_reply_consistency,
)

from pilot.data import get_test_loader, WorkerProgressTracker
from pilot.models import get_model
from pilot.train_test import test

app = ServerApp()


class PilotAvg(Strategy):
    """Custom Pilot Strategy based on FedAvg."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        dataset_name: str,
        dataset_size: int,
        batch_size: int,
        client_debug_port: bool,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.progress_tracker = WorkerProgressTracker(dataset_size=dataset_size, batch_size=batch_size)
        self.client_debug_port = client_debug_port
        self.weighted_by_key = "batches_processed"

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t└──> Dataset: '%s'", self.dataset_name)
        log(INFO, "\t└──> Progress Tracker: %s", self.progress_tracker)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, base_config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        node_ids = list(grid.get_node_ids())
        log(INFO, "configure_train: Training on all %s nodes", len(node_ids))

        base_config["server_round"] = server_round
        base_config["dataset_name"] = self.dataset_name
        if self.client_debug_port:
            base_config["client_debug_port"] = self.client_debug_port

        messages = []
        for node_id in node_ids:
            # Get or create worker state
            worker_state = self.progress_tracker.get_or_create_worker_state(node_id)
            shuffle_seed = self.progress_tracker.get_shuffle_seed(node_id)

            log(INFO, f"└──> Client {node_id}: seed={worker_state['seed']}, epoch={worker_state['epoch']}, "
                     f"batch={worker_state['batch_within_epoch']}, shuffle_seed={shuffle_seed}")

            # Add worker-specific config
            config = ConfigRecord({
                **base_config,
                "worker_seed": worker_state["seed"],
                "epoch": worker_state["epoch"],
                "batch_within_epoch": worker_state["batch_within_epoch"],
            })
            record = RecordDict({"arrays": arrays, "config": config})
            message = Message(content=record, message_type=MessageType.TRAIN, dst_node_id=node_id)
            messages.append(message)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Update per-worker progress from results
        for reply in replies:
            metrics: MetricRecord = reply.content["metrics"]
            node_id = reply.metadata.src_node_id
            batches_processed = metrics["batches_processed"]

            self.progress_tracker.update_worker_progress(node_id, batches_processed)

            worker_state = self.progress_tracker.worker_states[node_id]
            log(INFO, f"[Round {server_round}] Worker {node_id} processed {batches_processed} batches. "
                     f"New state: epoch={worker_state['epoch']}, batch={worker_state['batch_within_epoch']}")

        # Default FedAvg aggregation
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)  # can be customized

            # Log to wandb if enabled
            if wandb.run is not None:
                log_dict = {
                    "server/num_clients": len(valid_replies),
                }

                # Log aggregated server metrics
                if metrics:
                    log_dict["server/train_loss"] = metrics.get("train_loss", 0)

                # Log individual client metrics and progress
                for reply in valid_replies:
                    client_metrics: MetricRecord = reply.content["metrics"]
                    client_id = client_metrics['client_id']
                    node_id = reply.metadata.src_node_id
                    worker_state = self.progress_tracker.worker_states[node_id]

                    client_prefix = f"client_{client_id}"
                    log_dict.update({
                        f"{client_prefix}/train_loss": client_metrics["train_loss"],
                        f"{client_prefix}/batches_processed": client_metrics["batches_processed"],
                        f"{client_prefix}/epoch": worker_state["epoch"],
                        f"{client_prefix}/seed": worker_state["seed"],
                    })

                wandb.log(log_dict, step=server_round)

        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        return []
        node_ids = list(grid.get_node_ids())
        log(INFO, "configure_evaluate: Evaluating on all %s nodes", len(node_ids))

        config["server_round"] = server_round
        record = RecordDict({"arrays": arrays, "config": config})
        messages = [Message(content=record, message_type=MessageType.EVALUATE, dst_node_id=node_id)
                    for node_id in node_ids]
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=False)
        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)  # can be customized
        return metrics

    def _build_deadline_iter(self, anchor: pd.Timestamp) -> Iterator[pd.Timestamp]:
        next_boundary = anchor.floor(self._interval_delta) + self._interval_delta
        while True:
            yield next_boundary
            next_boundary = next_boundary + self._interval_delta

    def _check_and_log_replies(
        self, replies: Iterable[Message], is_train: bool, validate: bool = True
    ) -> tuple[list[Message], list[Message]]:
        """Check replies for errors and log them.

        Parameters
        ----------
        replies : Iterable[Message]
            Iterable of reply Messages.
        is_train : bool
            Set to True if the replies are from a training round; False otherwise.
            This impacts logging and validation behavior.
        validate : bool (default: True)
            Whether to validate the reply contents for consistency.

        Returns
        -------
        tuple[list[Message], list[Message]]
            A tuple containing two lists:
            - Messages with valid contents.
            - Messages with errors.
        """
        if not replies:
            return [], []

        # Filter messages that carry content
        valid_replies: list[Message] = []
        error_replies: list[Message] = []
        for msg in replies:
            if msg.has_error():
                error_replies.append(msg)
            else:
                valid_replies.append(msg)

        log(INFO, "%s: Received %s results and %s failures",
            "aggregate_train" if is_train else "aggregate_evaluate", len(valid_replies), len(error_replies))

        # Log errors
        for msg in error_replies:
            log(INFO, "\t> Received error in reply from node %d: %s", msg.metadata.src_node_id, msg.error.reason)

        # Ensure expected ArrayRecords and MetricRecords are received
        if validate and valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=is_train,
            )

        return valid_replies, error_replies


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    num_rounds: int = context.run_config["num_rounds"]
    lr: float = context.run_config["lr"]
    dataset_name: str = context.run_config["dataset_name"]
    dataset_size: int = context.run_config.get("dataset_size", 50000)  # CIFAR-10 train size
    batch_size: int = context.run_config["batch_size"]
    model_type: str = context.run_config["model_type"]

    server_debug_port: int = context.run_config.get("server_debug_port", None)
    if server_debug_port:
        print("[Server] Debug mode enabled...")
        import pydevd_pycharm
        pydevd_pycharm.settrace('localhost', port=server_debug_port, stdout_to_server=True, stderr_to_server=True)

    # Initialize wandb if enabled
    wandb_enabled: bool = context.run_config.get("wandb_enabled", False)
    if wandb_enabled:
        wandb_project: str = context.run_config["wandb_project"]
        wandb_entity: str | None = context.run_config.get("wandb_entity")

        # Get number of supernodes from grid
        num_supernodes = len(list(grid.get_node_ids()))

        # Create run name with hyperparameters
        run_name = f"nodes{num_supernodes},bs{batch_size},rounds{num_rounds}"

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={
                "num_supernodes": num_supernodes,
                "learning_rate": lr,
                "batch_size": batch_size,
                "dataset_size": dataset_size,
                "num_rounds": num_rounds,
                "model_type": model_type,
                "dataset_name": dataset_name,
            }
        )
        log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)

    # Load global model
    global_model = get_model(model_type)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with epoch management
    strategy = PilotAvg(
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        batch_size=batch_size,
        client_debug_port=context.run_config.get("client_debug_port", None),
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(dict(lr=lr)),
        num_rounds=num_rounds,
        evaluate_fn=lambda round, arrays: global_evaluate(round, arrays, model_type, dataset_name),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    # Finish wandb run
    if wandb_enabled:
        wandb.finish()
        log(INFO, "Wandb run finished")


def global_evaluate(server_round: int, arrays: ArrayRecord, model_type: str, dataset_name: str) -> MetricRecord:
    """Evaluate model on central test data."""

    # Load the model and initialize it with the received weights
    model = get_model(model_type)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test set
    test_dataloader = get_test_loader(dataset_name=dataset_name, batch_size=128)

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Log to wandb if enabled
    if wandb.run is not None:
        wandb.log({
            "server/eval_accuracy": test_acc,
            "server/eval_loss": test_loss,
        }, step=server_round)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
