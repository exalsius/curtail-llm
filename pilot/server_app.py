import io
import time
import threading
from logging import INFO
from typing import Iterable, Optional

import redis
import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.common import log, RecordDict, MessageType
from flwr.server.grid.grid import Grid
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import Strategy, Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    validate_message_reply_consistency,
    config_to_str,
)

from pilot.model import get_model
from pilot.data import ShardManager

app = ServerApp()


class PilotAvg(Strategy):
    """Custom Pilot Strategy based on FedAvg."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        dataset_name: str,
        num_shards: int,
        debug_port_client: bool,
        redis_url: str,
        redis_client: redis.Redis,
        min_round_duration: int,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_group: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.num_shards = num_shards
        self.shard_manager = ShardManager(num_shards=num_shards)
        self.debug_port_client = debug_port_client
        self.redis_url = redis_url
        self.redis_client = redis_client
        self.min_round_duration = min_round_duration
        self.weighted_by_key = "batches_processed"
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_group = wandb_run_group

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t└──> Dataset: '%s'", self.dataset_name)
        log(INFO, "\t└──> Shard: '%s'", self.shard_manager)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, base_config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        node_ids = list(grid.get_node_ids())
        log(INFO, "configure_train: Training on all %s nodes", len(node_ids))

        # Get worker assignments from shard manager
        assignments = self.shard_manager.assign_workers(node_ids)
        progress = self.shard_manager.get_progress_summary()
        log(INFO, f"Shard progress: {progress['progress']:.1%} ({progress['num_complete']}/{progress['num_total']} complete)")

        for node_id, shard_list in assignments.items():
            if shard_list:
                shard_summary = ", ".join(f"shard_{sid}@{start}" for sid, start in shard_list)
                log(INFO, f"└──> Client {node_id}: {len(shard_list)} shards [{shard_summary}]")
            else:
                log(INFO, f"└──> Client {node_id}: No shards (training complete)")

        base_config["server_round"] = server_round
        base_config["dataset_name"] = self.dataset_name
        base_config["num_shards"] = self.num_shards
        base_config["redis_url"] = self.redis_url
        if self.debug_port_client:
            base_config["debug_port_client"] = self.debug_port_client

        # Pass W&B configuration to clients
        base_config["wandb_project"] = self.wandb_project
        base_config["wandb_group"] = self.wandb_run_group
        if self.wandb_entity:
            base_config["wandb_entity"] = self.wandb_entity

        messages = []
        for node_id in node_ids:
            shard_list = assignments[node_id]
            # Convert list of tuples to two parallel lists for serialization
            shard_ids = [sid for sid, _ in shard_list]
            shard_starts = [start for _, start in shard_list]
            config = ConfigRecord({
                **base_config,
                "shard_ids": shard_ids,
                "shard_starts": shard_starts,
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
        # Update shard states from worker results
        for reply in replies:
            metrics: MetricRecord = reply.content["metrics"]
            # Reconstruct shard_updates from two parallel lists
            shard_ids = metrics["shard_ids"]
            shard_rows = metrics["shard_rows"]
            shard_updates = list(zip(shard_ids, shard_rows))
            self.shard_manager.update(shard_updates)

        progress = self.shard_manager.get_progress_summary()
        log(INFO, f"[Round {server_round}] Shard progress: {progress['progress']:.1%} "
                  f"({progress['processed_rows']:,}/{progress['total_rows']:,} rows, "
                  f"{progress['num_complete']}/{progress['num_total']} complete)")

        # Default FedAvg aggregation
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)  # can be customized

            # Log to wandb
            log_dict = {
                "server/num_clients": len(valid_replies),
                "data/progress": progress["progress"],
                "data/total_rows_processed": progress["processed_rows"],
                "data/shards_complete": progress["num_complete"],
            }

            # Log aggregated server metrics
            if metrics:
                log_dict["server/train_loss"] = metrics.get("train_loss", 0)
                if "train_ppl" in metrics:
                    log_dict["server/train_ppl"] = metrics.get("train_ppl", 0)
                if "actual_train_time" in metrics:
                    log_dict["server/actual_train_time"] = metrics.get("actual_train_time", 0)

            # Log individual client metrics
            for reply in valid_replies:
                client_metrics: MetricRecord = reply.content["metrics"]
                client_prefix = f"client_{client_metrics['client_id']}"
                # Always present
                log_dict[f"{client_prefix}/total_rows_processed"] = client_metrics["total_rows_processed"]
                log_dict[f"{client_prefix}/batches_processed"] = client_metrics["batches_processed"]
                # Optional metrics, add if available
                if "train_loss" in client_metrics:
                    log_dict[f"{client_prefix}/train_loss"] = client_metrics["train_loss"]
                if "train_ppl" in client_metrics:
                    log_dict[f"{client_prefix}/train_ppl"] = client_metrics["train_ppl"]
                if "actual_train_time" in client_metrics:
                    log_dict[f"{client_prefix}/actual_train_time"] = client_metrics["actual_train_time"]

            # Add individual shard states (processed rows for each shard)
            for shard_id, state in self.shard_manager.shard_states.items():
                log_dict[f"data/shard_{shard_id}_rows"] = state["processed_rows"]

            wandb.log(log_dict, step=server_round)

        return arrays, metrics

    def configure_evaluate(self, server_round, arrays, base_config, grid) -> Iterable[Message]:
        return []  # not used, server-side eval only

    def aggregate_evaluate(self, server_round, replies) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        return None, None  # not used, server-side eval only

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 1e100,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[callable] = None,
    ):
        """Run time-based federated learning."""
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log(INFO, "\t├── ArrayRecord (%.2f MB)", sum(len(array.data) for array in initial_arrays.values()) / (1024 ** 2))
        log(INFO, "\t├── ConfigRecord (train): %s", config_to_str(train_config))

        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        result = Result()

        t_start = time.time()
        # Note: No evaluation

        arrays = initial_arrays

        for current_round in range(1, int(num_rounds + 1)):
            log(INFO, f"\n[ROUND {current_round}]")

            # TRAINING
            # Start monitor thread to signal stop when conditions are met
            def _monitor():
                time.sleep(self.min_round_duration)
                while len(list(grid.get_node_ids())) <= 1:
                    time.sleep(1)
                self.redis_client.publish(f"round:{current_round}:stop", "stop")

            threading.Thread(target=_monitor, daemon=True).start()

            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )
            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # Note: No evaluation

        log(INFO, "\nStrategy execution finished in %.2fs\n", time.time() - t_start)
        log(INFO, "Final results:\n")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result

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
    lr: float = context.run_config["lr"]
    dataset_name: str = context.run_config["dataset_name"]
    num_shards: int = context.run_config["num_shards"]
    batch_size: int = context.run_config["batch_size"]
    model_type: str = context.run_config["model_type"]
    redis_url: str = context.run_config["redis_url"]
    min_round_duration: int = context.run_config.get("round_min_duration", 300)

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

    strategy = PilotAvg(
        dataset_name=dataset_name,
        num_shards=num_shards,
        debug_port_client=context.run_config.get("debug_port_client", None),
        redis_url=redis_url,
        redis_client=redis_client,
        min_round_duration=min_round_duration,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_group=run_name,
    )

    # Load global model
    max_length = context.run_config.get("max_length", 2048)
    global_model = get_model(model_type, max_length)

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord(dict(lr=lr)),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    # Finish wandb run
    wandb.finish()
    log(INFO, "Wandb run finished")
