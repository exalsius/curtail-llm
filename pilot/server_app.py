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
from pilot.data import get_data_loaders
from pilot.train_test import test

app = ServerApp()


class PilotAvg(FedAvg):
    """FedAvg variant that annotates each round with absolute time windows."""

    def __init__(
        self,
        *,
        round_interval_seconds: float,
        schedule_anchor_ts: float,
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

    def _build_deadline_iter(self, anchor: pd.Timestamp) -> Iterator[pd.Timestamp]:
        next_boundary = anchor.floor(self._interval_delta) + self._interval_delta
        while True:
            yield next_boundary
            next_boundary = next_boundary + self._interval_delta

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure train round and inject timing metadata."""
        config["round-deadline"] = next(self._deadline_iter).timestamp()
        return super().configure_train(server_round, arrays, config, grid)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    debug: bool = context.run_config["debug"]
    round_interval_seconds: float = context.run_config["round-interval-seconds"]

    # Establish the anchor timestamp for round scheduling
    schedule_anchor_ts = time.time()

    # Load global model
    model_type = context.run_config.get("model-type", "resnet18")
    global_model = get_model(model_type)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = PilotAvg(
        fraction_evaluate=fraction_evaluate,
        round_interval_seconds=round_interval_seconds,
        schedule_anchor_ts=schedule_anchor_ts,
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

    # Load entire test set
    test_dataloader = get_data_loaders(
        data_type="cifar10",
        centralized=True,
        model_type=model_type,
        batch_size=128
    )

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
