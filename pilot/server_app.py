import time
from collections.abc import Iterator
from typing import Iterable

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg

from pilot.models import get_model
from pilot.data import get_data_loaders, QueueManager
from pilot.train_test import test

app = ServerApp()


class PilotAvgWithQueues(FedAvg):
    """FedAvg with time-based scheduling and queue-based data management."""

    def __init__(
        self,
        *,
        round_interval_seconds: float,
        schedule_anchor_ts: float,
        queue_manager: QueueManager,
        dataset_name: str,
        **kwargs,
    ) -> None:
        if "min_train_nodes" not in kwargs:
            kwargs["min_train_nodes"] = 1
        if "min_available_nodes" not in kwargs:
            kwargs["min_available_nodes"] = 1

        super().__init__(**kwargs)
        self.round_interval_seconds = round_interval_seconds
        self._interval_delta = pd.to_timedelta(round_interval_seconds, unit="s")
        anchor = pd.Timestamp.utcfromtimestamp(schedule_anchor_ts)
        self._deadline_iter: Iterator[pd.Timestamp] = self._build_deadline_iter(anchor)

        # Queue management
        self.queue_manager = queue_manager
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

    # Initialize queue manager
    queue_manager = QueueManager(num_queues=num_queues)
    print(f"\n[Server] Initialized {queue_manager}")

    # Load global model
    model_type = context.run_config.get("model-type", "resnet18")
    global_model = get_model(model_type)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with queue management
    strategy = PilotAvgWithQueues(
        fraction_evaluate=fraction_evaluate,
        round_interval_seconds=round_interval_seconds,
        schedule_anchor_ts=schedule_anchor_ts,
        queue_manager=queue_manager,
        dataset_name=dataset_name,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(dict(lr=lr, debug=debug)),
        num_rounds=num_rounds,
        # evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord, model_type: str = "resnet18") -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = get_model(model_type)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set (num_shards=1 gives full dataset)
    test_dataloader = get_data_loaders(
        dataset_name="uoft-cs/cifar10",
        shard_id=0,
        num_shards=1,
        start_batch_idx=0,
        epoch=0,
        batch_size=128,
        data_format="image",
        model_type=model_type,
    )

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
