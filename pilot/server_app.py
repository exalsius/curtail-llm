import time
from collections.abc import Iterator
from typing import Iterable

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    validate_message_reply_consistency,
)

from pilot.models import get_model
from pilot.data import get_test_loader, QueueManager
from pilot.train_test import test

app = ServerApp()


class NewStrategy(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for both ArrayRecords and MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
    ) -> None:
        self.weighted_by_key = weighted_by_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t└──> Keys in records:")
        log(INFO, "\t\t├── Weighted by: '%s'", self.weighted_by_key)
        log(INFO, "\t\t├── ArrayRecord key: '%s'", self.arrayrecord_key)
        log(INFO, "\t\t└── ConfigRecord key: '%s'", self.configrecord_key)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        node_ids = list(grid.get_node_ids())
        log(INFO, "configure_train: Training on all %s nodes", len(node_ids))

        config["server-round"] = server_round

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)  # can be customized
        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        node_ids = list(grid.get_node_ids())
        log(INFO, "configure_evaluate: Evaluating on all %s nodes", len(node_ids))

        config["server-round"] = server_round

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

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

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> Iterable[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = [Message(content=record, message_type=message_type, dst_node_id=node_id) for node_id in node_ids]
        return messages

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


class PilotAvg(FedAvg):
    """FedAvg with time-based scheduling and queue-based data management."""

    def __init__(
        self,
        *,
        round_interval_seconds: float,
        schedule_anchor_ts: float,
        dataset_name: str,
        num_queues: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.round_interval_seconds = round_interval_seconds
        self._interval_delta = pd.to_timedelta(round_interval_seconds, unit="s")
        anchor = pd.Timestamp.utcfromtimestamp(schedule_anchor_ts)
        self._deadline_iter: Iterator[pd.Timestamp] = self._build_deadline_iter(anchor)

        # Queue management
        self.queue_manager = QueueManager(num_queues=num_queues)
        print(f"\n[Server] Initialized {self.queue_manager}")
        self.dataset_name = dataset_name

    def _build_deadline_iter(self, anchor: pd.Timestamp) -> Iterator[pd.Timestamp]:
        next_boundary = anchor.floor(self._interval_delta) + self._interval_delta
        while True:
            yield next_boundary
            next_boundary = next_boundary + self._interval_delta

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure train round with queue assignments and timing metadata."""

        # Get worker assignments from queue manager
        num_available = grid.num_nodes()
        assignments = self.queue_manager.assign_workers(num_available)

        print(f"\n[Round {server_round}] Assigning {len(assignments)} workers to queues")
        print(f"[Round {server_round}] Queue states: {self.queue_manager.queue_states}")

        # Inject queue assignments into config
        config["num-queues"] = self.queue_manager.num_queues
        config["dataset-name"] = self.dataset_name
        config["assignments"] = assignments  # List of (queue_id, epoch, batch_idx)

        # Add round deadline
        config["round-deadline"] = next(self._deadline_iter).timestamp()

        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, results: Iterable[Message], grid: Grid
    ) -> tuple[ArrayRecord, ConfigRecord]:
        """Aggregate results and update queue states."""

        # Update queue states from worker results
        results_list = list(results)
        for result in results_list:
            metrics = result.content["metrics"]
            queue_id = metrics.get("queue-id")

            if queue_id is not None:
                final_batch_idx = metrics["final-batch-idx"]
                final_epoch = metrics["final-epoch"]
                self.queue_manager.update(queue_id, final_batch_idx, final_epoch)

                print(f"[Round {server_round}] Updated Queue {queue_id}: "
                      f"epoch={final_epoch}, batch={final_batch_idx}")

        print(f"[Round {server_round}] Final queue states: {self.queue_manager.queue_states}")

        # Call parent aggregate_train with the original results iterator
        return super().aggregate_train(server_round, iter(results_list), grid)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    debug: bool = context.run_config["debug"]
    round_interval_seconds: float = context.run_config["round-interval-seconds"]
    num_queues: int = context.run_config["num-queues"]
    dataset_name: str = context.run_config["dataset-name"]

    # Establish the anchor timestamp for round scheduling
    schedule_anchor_ts = time.time()

    # Load global model
    model_type = context.run_config.get("model-type", "resnet18")
    global_model = get_model(model_type)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with queue management
    strategy = PilotAvg(
        fraction_evaluate=fraction_evaluate,
        round_interval_seconds=round_interval_seconds,
        schedule_anchor_ts=schedule_anchor_ts,
        dataset_name=dataset_name,
        num_queues=num_queues,
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

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
