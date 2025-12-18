import asyncio
import time
from dataclasses import dataclass
from logging import INFO
from typing import Optional, Literal, Iterable

import redis
import torch
import wandb

from flwr.common import log, ArrayRecord, ConfigRecord, Message, RecordDict, MessageType, MetricRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords, aggregate_metricrecords, config_to_str, \
    validate_message_reply_consistency
from redis import Redis

from pilot.data import ShardManager


@dataclass
class Client:
    name: str
    host: str  # TODO currently unused
    node_id: Optional[int] = None  # TODO
    _state: Literal["OFF", "STARTING", "IDLE", "TRAINING"] = "OFF"    # , "STOPPING"

    def update_provisioning(self, redis_client):
        # Check whether the client should be provisioned
        if self._state == "OFF":
            # TODO query forecast
            below_threshold = False  # TODO define and compare to thresholds
            if below_threshold:
                self._state = "STARTING"
                # TODO: Call provisioning API
                # State transitions to IDLE happens in the server loop once client connects to grid
        # Check whether the client should be deprovisioned
        else:
            # TODO query forecast
            below_threshold = False  # TODO define and compare to thresholds
            if below_threshold:  # TODO should also only deprovision if end of billing interval
                asyncio.create_task(self._deprovision(redis_client))

    def start_training(self):
        assert self._state == "IDLE", f"Cannot start training: client '{self.name}' is {self._state}, expected IDLE"
        self._state = "TRAINING"

    def stop_training(self):
        assert self._state == "TRAINING", f"Cannot stop training: client '{self.name}' is {self._state}, expected TRAINING"
        self._state = "IDLE"

    async def _deprovision(self, redis_client: Redis):
        """Gracefully deprovision a client by signaling stop and waiting for completion."""
        # Wrap up current round in case the client is still training
        if self._state == "TRAINING":
            log(INFO, f"Signaling client '{self.name}' to stop training")
            await redis_client.publish(f"{self.name}:stop", "DEPROVISION {self.name}")

            # Wait for client to become IDLE (set by train loop after results have been returned)
            wait_start = time.time()
            while self._state == "TRAINING" and (time.time() - wait_start) < 60:
                await asyncio.sleep(1)
            if self._state == "TRAINING":
                log(INFO, f"Client '{self.name}' did not respond within timeout")

        assert self._state == "IDLE"
        # TODO: Call deprovisioning API
        log(INFO, f"Deprovisioning client '{self.name}'")
        self._state = "OFF"


class PilotAvg(Strategy):
    """Custom Pilot Strategy based on FedAvg."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        clients: dict[str, Client],
        dataset_name: str,
        num_shards: int,
        debug_port_client: bool,
        redis_url: str,
        redis_client: redis.Redis,
        round_min_duration: int,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_group: Optional[str] = None,
    ) -> None:
        self.clients: dict[str, Client] = clients
        self.dataset_name = dataset_name
        self.num_shards = num_shards
        self.shard_manager = ShardManager(num_shards=num_shards)
        self.debug_port_client = debug_port_client
        self.redis_url = redis_url
        self.redis_client = redis_client
        self.round_min_duration = round_min_duration
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
        while len(node_ids := list(grid.get_node_ids())) < 1:
            log(INFO, "Waiting for a client to connect")
            time.sleep(1)


        #################################################
        log(INFO, f"Querying all {len(node_ids)} connected nodes for their names...")
        messages = []
        for node_id in node_ids:
            content = RecordDict({"config": ConfigRecord({})})
            messages.append(Message(content=content, message_type=MessageType.QUERY, dst_node_id=node_id))
        query_replies = grid.send_and_receive(messages=messages, timeout=10)
        for reply in query_replies:
            client_name: str = reply.content["config"]["name"]
            log(INFO, f" - Node {reply.metadata.src_node_id}: '{client_name}'")
            self.clients[client_name]._state = "IDLE"
            self.clients[client_name].node_id = reply.metadata.src_node_id
        #################################################


        log(INFO, "configure_train: Training on all %s nodes", len(node_ids))

        # Get worker assignments from shard manager
        assignments = self.shard_manager.assign_workers(node_ids)
        progress = self.shard_manager.get_progress_summary()
        log(INFO, f"Shard progress: {progress['progress']:.1%} ({progress['num_complete']}/{progress['num_total']} complete)")

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
            config = ConfigRecord({
                **base_config,
                **assignments[node_id],
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
        valid_replies, _ = self._check_and_log_replies(replies)

        arrays, metrics = None, None
        if valid_replies:
            for reply in valid_replies:
                # Update shard states from worker results
                metrics: MetricRecord = reply.content["metrics"]
                self.shard_manager.update(metrics["shard_ids"], metrics["shard_rows"])

            progress = self.shard_manager.get_progress_summary()
            log(INFO, f"[Round {server_round}] Shard progress: {progress['progress']:.1%} "
                      f"({progress['processed_rows']:,}/{progress['total_rows']:,} rows, "
                      f"{progress['num_complete']}/{progress['num_total']} complete)")

            # Default FedAvg aggregation
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

    async def start(
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
        log(INFO, f"Starting {self.__class__.__name__} strategy:")
        log(INFO, f"\t├── ArrayRecord ({sum(len(array.data) for array in initial_arrays.values()) / (1024 ** 2):.2f} MB)")
        log(INFO, f"\t├── ConfigRecord (train): {config_to_str(train_config)}")
        self.summary()
        train_config = ConfigRecord() if train_config is None else train_config

        provisioning_task = asyncio.create_task(_provisioning_task(self.clients, self.redis_client))

        start = time.time()
        arrays = initial_arrays
        for current_round in range(1, int(num_rounds + 1)):
            log(INFO, f"\n[ROUND {current_round}]")

            messages = self.configure_train(current_round, arrays, train_config, grid)
            round_controller = asyncio.create_task(_round_controller(grid, self.redis_client, self.round_min_duration, current_round))
            train_replies = grid.send_and_receive(messages=messages, timeout=timeout)
            round_controller.cancel()  # once grid.send_and_receive() has returned, the monitor is obsolete

            arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)

        provisioning_task.cancel()
        log(INFO, f"\nStrategy execution finished in {time.time() - start:.2f}s")
        log(INFO, "\nSaving final model to disk...")
        torch.save(arrays.to_torch_state_dict(), "final_model.pt")

    def _check_and_log_replies(self, replies: Iterable[Message]) -> tuple[list[Message], list[Message]]:
        """Copied from FedAvg"""
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

        log(INFO, "aggregate_train: Received %s results and %s failures", len(valid_replies), len(error_replies))

        # Log errors
        for msg in error_replies:
            log(INFO, "\t> Received error in reply from node %d: %s", msg.metadata.src_node_id, msg.error.reason)

        # Ensure expected ArrayRecords and MetricRecords are received
        if valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=True,
            )

        return valid_replies, error_replies


async def _provisioning_task(clients: dict[str, Client], redis_client: Redis):
    """Monitor clients for provisioning and deprovisioning decisions."""
    while True:
        for client in clients.values():
            client.update_provisioning(redis_client)
        await asyncio.sleep(5)


async def _round_controller(grid: Grid, redis_client: Redis, round_min_duration: float, current_round: int):
    log(INFO, f"Round monitor started for ROUND {current_round}")
    start = time.time()
    await asyncio.sleep(round_min_duration)
    while active_nodes := len(list(grid.get_node_ids())) <= 1:
        log(INFO, f"Round active since {int(time.time() - start)}s, waiting for more clients to join...")
        await asyncio.sleep(10)
    await redis_client.publish(f"round:{current_round}:stop", f"END ROUND {current_round}")
    log(INFO, f"Signaled ROUND END after {int(time.time() - start)}s with {active_nodes} connected clients.")
