import asyncio
import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from logging import INFO
from typing import Optional, Literal, Iterable

import redis
import redis.asyncio
import requests
import torch
import wandb

from flwr.common import (
    log,
    ArrayRecord,
    ConfigRecord,
    Message,
    RecordDict,
    MessageType,
    MetricRecord,
)
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    config_to_str,
    validate_message_reply_consistency,
)
from redis import Redis

from pilot.data import ShardManager
from pilot.provisioner import ExlsProvisioner

FlwrNodeId = int


@dataclass
class Client:
    name: str
    provision_threshold: int
    deprovision_threshold: int
    redis_url: str
    flwr_node_id: Optional[FlwrNodeId] = None  # Set when client connects to Flower Grid
    _state: Literal["OFF", "STARTING", "IDLE", "TRAINING"] = "OFF"

    def update_provisioning(
        self, redis_client: Redis, provisioner: ExlsProvisioner, carbon_intensity: float
    ):
        # Check whether the client should be provisioned
        if self._state == "OFF":
            if carbon_intensity < self.provision_threshold:
                self._state = "STARTING"
                provisioner.add_node(self.name)
                # State transitions to IDLE happens in the server loop once client connects to grid
        # Check whether the client should be deprovisioned
        else:
            if carbon_intensity > self.deprovision_threshold:
                threading.Thread(target=self._deprovision, args=(provisioner,)).start()

    def start_training(self):
        assert (
            self._state == "IDLE"
        ), f"Cannot start training: client '{self.name}' is {self._state}, expected IDLE"
        self._state = "TRAINING"

    def stop_training(self):
        assert (
            self._state == "TRAINING"
        ), f"Cannot stop training: client '{self.name}' is {self._state}, expected TRAINING"
        self._state = "IDLE"

    def _deprovision(self, provisioner: ExlsProvisioner):
        """Gracefully deprovision a client by signaling stop and waiting for completion."""
        redis_client = redis.from_url(self.redis_url)
        # Wrap up current round in case the client is still training
        if self._state == "TRAINING":
            log(INFO, f"Signaling client '{self.name}' to stop training")
            redis_client.publish(f"{self.name}:stop", "DEPROVISION {self.name}")

            # Wait for client to become IDLE (set by train loop after results have been returned)
            wait_start = time.time()
            while self._state == "TRAINING" and (time.time() - wait_start) < 60:
                time.sleep(1)
            if self._state == "TRAINING":
                log(INFO, f"Client '{self.name}' did not respond within timeout")

        assert self._state == "IDLE"
        provisioner.remove_node(self.name)
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
        round_min_duration: int,
        provisioner: Optional[ExlsProvisioner],
        forecast_api_url: str,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ) -> None:
        self.clients: dict[str, Client] = clients
        self.dataset_name = dataset_name
        self.num_shards = num_shards
        self.debug_port_client = debug_port_client
        self.redis_url = redis_url
        self.round_min_duration = round_min_duration
        self.provisioner = provisioner
        self.forecast_api_url = forecast_api_url
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        self.redis_client = redis.from_url(redis_url)
        self.shard_manager = ShardManager(num_shards=num_shards)
        self.weighted_by_key = "batches_processed"
        self.global_tokens_processed = 0

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t└──> Dataset: '%s'", self.dataset_name)
        log(INFO, "\t└──> Shard: '%s'", self.shard_manager)

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        base_config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        while len(flwr_node_ids := list(grid.get_node_ids())) < 1:
            log(INFO, "Waiting for a client to connect")
            time.sleep(1)

        # --- Server-side Scheduling ---
        # Work on a copy to preserve baseline LRs in base_config
        round_config = dict(base_config)

        total_batch_size = int(round_config["total_batch_size"])
        self.total_batch_size = total_batch_size
        current_step = self.global_tokens_processed // total_batch_size

        # Scheduler helpers
        num_iterations = int(round_config["num_iterations"])
        warmup_ratio = float(round_config["warmup_ratio"])
        warmdown_ratio = float(round_config["warmdown_ratio"])
        final_lr_frac = float(round_config["final_lr_frac"])

        def get_lr_multiplier(step):
            warmup_iters = round(warmup_ratio * num_iterations)
            warmdown_iters = round(warmdown_ratio * num_iterations)
            if step < warmup_iters:
                return (step + 1) / warmup_iters
            elif step <= num_iterations - warmdown_iters:
                return 1.0
            else:
                progress = (num_iterations - step) / warmdown_iters
                return progress * 1.0 + (1 - progress) * final_lr_frac

        def get_muon_momentum(step):
            frac = min(step / 300, 1)
            return (1 - frac) * 0.85 + frac * 0.95

        lrm = get_lr_multiplier(current_step)
        momentum = get_muon_momentum(current_step)

        # Apply schedule
        # Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
        batch_lr_scale = 1.0
        reference_batch_size = 2**19
        batch_ratio = total_batch_size / reference_batch_size
        if batch_ratio != 1.0:
            batch_lr_scale = batch_ratio**0.5
            log(
                INFO,
                f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}",
            )

        # Weight decay is tuned at d12 and its scaling seems to be \propto 1/channels^2 (or equivalently, \propto 1/depth^2 due to constant aspect ratio)
        weight_decay = float(round_config.get("weight_decay", 0.0))
        depth = int(round_config["n_layer"])
        weight_decay_scaled = weight_decay * (12 / depth) ** 2
        if depth != 12:
            log(
                INFO,
                f"Scaling weight decay from {weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {depth}",
            )

        round_config["matrix_lr"] = (
            float(round_config["matrix_lr"]) * lrm * batch_lr_scale
        )
        round_config["embedding_lr"] = (
            float(round_config["embedding_lr"]) * lrm * batch_lr_scale
        )
        round_config["unembedding_lr"] = (
            float(round_config["unembedding_lr"]) * lrm * batch_lr_scale
        )
        round_config["scalar_lr"] = (
            float(round_config.get("scalar_lr", 0.5)) * lrm * batch_lr_scale
        )
        round_config["muon_momentum"] = momentum
        round_config["weight_decay"] = weight_decay_scaled

        log(
            INFO,
            f"Round {server_round} schedule: Step {current_step}, LRM {lrm:.4f}, Momentum {momentum:.4f}",
        )
        # ------------------------------

        #################################################
        log(
            INFO,
            f"Querying all {len(flwr_node_ids)} connected Flower nodes for their names...",
        )
        messages = []
        for flwr_node_id in flwr_node_ids:
            content = RecordDict({"config": ConfigRecord({})})
            messages.append(
                Message(
                    content=content,
                    message_type=MessageType.QUERY,
                    dst_node_id=flwr_node_id,
                )
            )
        query_replies = grid.send_and_receive(messages=messages, timeout=10)
        for reply in query_replies:
            client_name: str = reply.content["config"]["name"]
            log(INFO, f" - Flower node {reply.metadata.src_node_id}: '{client_name}'")
            self.clients[client_name]._state = "IDLE"
            self.clients[client_name]._flwr_node_id = reply.metadata.src_node_id
        #################################################

        log(
            INFO, "configure_train: Training on all %s Flower nodes", len(flwr_node_ids)
        )

        # Get worker assignments from shard manager
        assignments = self.shard_manager.assign_workers(flwr_node_ids)
        progress = self.shard_manager.get_progress_summary()
        log(
            INFO,
            f"Shard progress: {progress['progress']:.1%} ({progress['num_complete']}/{progress['num_total']} complete)",
        )

        # Set clients to TRAINING state
        node_id_to_client: dict[FlwrNodeId, Client] = {
            client.flwr_node_id: client
            for client in self.clients.values()
            if client.flwr_node_id is not None
        }
        for flwr_node_id in flwr_node_ids:
            if flwr_node_id in node_id_to_client:
                node_id_to_client[flwr_node_id].start_training()

        round_config["server_round"] = server_round
        round_config["dataset_name"] = self.dataset_name
        round_config["num_shards"] = self.num_shards
        round_config["redis_url"] = self.redis_url
        round_config["global_tokens_processed_start"] = self.global_tokens_processed

        if self.debug_port_client:
            round_config["debug_port_client"] = self.debug_port_client

        messages = []
        for flwr_node_id in flwr_node_ids:
            config = ConfigRecord(
                {
                    **round_config,
                    **assignments[flwr_node_id],
                }
            )
            record = RecordDict({"arrays": arrays, "config": config})
            message = Message(
                content=record, message_type=MessageType.TRAIN, dst_node_id=flwr_node_id
            )
            messages.append(message)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies)

        node_id_to_client: dict[FlwrNodeId, Client] = {
            client.flwr_node_id: client
            for client in self.clients.values()
            if client.flwr_node_id is not None
        }

        arrays, metrics = None, None
        if valid_replies:
            for reply in valid_replies:
                # Update client state
                if reply.metadata.src_node_id in node_id_to_client:
                    node_id_to_client[reply.metadata.src_node_id].stop_training()

                # Update shard states from worker results
                metrics: MetricRecord = reply.content["metrics"]
                self.shard_manager.update(metrics["shard_ids"], metrics["shard_rows"])

                # Accumulate global tokens
                if "num_tokens_processed" in metrics:
                    self.global_tokens_processed += int(metrics["num_tokens_processed"])

            progress = self.shard_manager.get_progress_summary()
            log(
                INFO,
                f"[Round {server_round}] Shard progress: {progress['progress']:.1%} "
                f"({progress['processed_rows']:,}/{progress['total_rows']:,} rows, "
                f"{progress['num_complete']}/{progress['num_total']} complete)",
            )

            # Default FedAvg aggregation
            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = aggregate_metricrecords(
                reply_contents, self.weighted_by_key
            )  # can be customized

            # Log to wandb
            log_dict = {
                "server/num_clients": len(valid_replies),
                "data/progress": progress["progress"],
                "data/shards_complete": progress["num_complete"],
                "server/global_tokens_processed": self.global_tokens_processed,
            }

            # Log aggregated server metrics
            if metrics:
                log_dict["server/train_loss"] = metrics.get("train_loss", 0)
                if "train_ppl" in metrics:
                    log_dict["server/train_ppl"] = metrics.get("train_ppl", 0)
                if "actual_train_time" in metrics:
                    log_dict["server/actual_train_time"] = metrics.get(
                        "actual_train_time", 0
                    )

            # Log individual client metrics
            for reply in valid_replies:
                client_metrics: MetricRecord = reply.content["metrics"]
                client_prefix = f"client_{client_metrics['client_id']}"
                # Always present
                log_dict[f"{client_prefix}/batches_processed"] = client_metrics[
                    "batches_processed"
                ]
                # Optional metrics, add if available
                if "train_loss" in client_metrics:
                    log_dict[f"{client_prefix}/train_loss"] = client_metrics[
                        "train_loss"
                    ]
                if "train_ppl" in client_metrics:
                    log_dict[f"{client_prefix}/train_ppl"] = client_metrics["train_ppl"]
                if "actual_train_time" in client_metrics:
                    log_dict[f"{client_prefix}/actual_train_time"] = client_metrics[
                        "actual_train_time"
                    ]

            # Use global step for summary
            summary_step = (
                self.global_tokens_processed // self.total_batch_size
                if hasattr(self, "total_batch_size")
                else server_round
            )
            wandb.log(log_dict, step=summary_step)

        return arrays, metrics

    def configure_evaluate(
        self, server_round, arrays, base_config, grid
    ) -> Iterable[Message]:
        return []  # not used, server-side eval only

    def aggregate_evaluate(
        self, server_round, replies
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
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
        log(INFO, f"Starting {self.__class__.__name__} strategy:")
        log(
            INFO,
            f"\t├── ArrayRecord ({sum(len(array.data) for array in initial_arrays.values()) / (1024 ** 2):.2f} MB)",
        )
        log(INFO, f"\t├── ConfigRecord (train): {config_to_str(train_config)}")
        self.summary()
        train_config = ConfigRecord() if train_config is None else train_config

        stop_event = threading.Event()
        threads = []

        if self.provisioner:
            provisioning_thread = threading.Thread(
                target=_provisioning_task,
                args=(
                    stop_event,
                    self.clients,
                    self.redis_client,
                    self.provisioner,
                    self.forecast_api_url,
                ),
                daemon=True,
            )
            provisioning_thread.start()
            threads.append(provisioning_thread)

        polling_thread = threading.Thread(
            target=_poll_logs,
            args=(stop_event, self.redis_url, self.clients),
            daemon=True,
        )
        polling_thread.start()
        threads.append(polling_thread)

        start = time.time()
        arrays = initial_arrays
        for current_round in range(1, int(num_rounds + 1)):
            log(INFO, f"\n[ROUND {current_round}]")

            if self.shard_manager.is_complete():
                log(INFO, "All data shards processed. Terminating training.")
                break

            round_controller_thread = threading.Thread(
                target=_round_controller,
                args=(
                    stop_event,
                    grid,
                    self.redis_client,
                    self.round_min_duration,
                    current_round,
                ),
                daemon=True,
            )
            round_controller_thread.start()

            messages = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=messages, timeout=timeout)
            
            # The round controller's job is done once send_and_receive returns, but we can't easily kill it.
            # It will terminate on its own with the main thread, or after its own timeout.

            arrays, agg_train_metrics = self.aggregate_train(
                current_round, train_replies
            )

            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)

        stop_event.set()
        for thread in threads:
            thread.join(timeout=5)
            if thread.is_alive():
                log(INFO, f"Thread {thread.name} did not terminate in time.")

        log(INFO, f"\nStrategy execution finished in {time.time() - start:.2f}s")
        log(INFO, "\nSaving final model to disk...")
        torch.save(arrays.to_torch_state_dict(), "final_model.pt")

    def _check_and_log_replies(
        self, replies: Iterable[Message]
    ) -> tuple[list[Message], list[Message]]:
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

        log(
            INFO,
            "aggregate_train: Received %s results and %s failures",
            len(valid_replies),
            len(error_replies),
        )

        # Log errors
        for msg in error_replies:
            log(
                INFO,
                "\t> Received error in reply from node %d: %s",
                msg.metadata.src_node_id,
                msg.error.reason,
            )

        # Ensure expected ArrayRecords and MetricRecords are received
        if valid_replies:
            validate_message_reply_consistency(
                replies=[msg.content for msg in valid_replies],
                weighted_by_key=self.weighted_by_key,
                check_arrayrecord=True,
            )

        return valid_replies, error_replies


def _poll_logs(stop_event: threading.Event, redis_url: str, clients: dict[str, Client]):
    """Poll client logs from Redis and log them to wandb."""
    log(INFO, "Starting Redis log polling task...")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    while not stop_event.is_set():
        try:
            for client_name in clients.keys():
                log_key = f"logs:{client_name}"
                
                pipe = redis_client.pipeline()
                pipe.lrange(log_key, 0, -1)
                pipe.ltrim(log_key, 1, 0)
                log_entries_str = pipe.execute()[0]

                if log_entries_str:
                    log(
                        INFO,
                        f"Fetched {len(log_entries_str)} log entries for {client_name} from Redis.",
                    )
                    for log_entry_str in log_entries_str:
                        try:
                            log_entry = json.loads(log_entry_str)
                            wandb.log(log_entry, step=log_entry["step"])
                        except json.JSONDecodeError:
                            log(INFO, f"Could not decode log entry: {log_entry_str}")
            
            time.sleep(10)
        except Exception as e:
            log(INFO, f"Error in log polling task: {e}")
            time.sleep(30)


def _provisioning_task(
    stop_event: threading.Event,
    clients: dict[str, Client],
    redis_client: Redis,
    provisioner: ExlsProvisioner,
    forecast_api_url: str,
):
    """Monitor clients for provisioning and deprovisioning decisions."""
    while not stop_event.is_set():
        for client in clients.values():
            mci = get_mci(forecast_api_url, client.name)
            client.update_provisioning(redis_client, provisioner, mci)
        time.sleep(5)


def _round_controller(
    stop_event: threading.Event, grid: Grid, redis_client: Redis, round_min_duration: float, current_round: int
):
    log(INFO, f"Round monitor started for ROUND {current_round}")
    start_time = time.time()
    
    # Wait for the minimum round duration
    while not stop_event.is_set() and time.time() - start_time < round_min_duration:
        time.sleep(1)

    # Wait for more clients to join
    while not stop_event.is_set() and len(list(grid.get_node_ids())) <= 1:
        log(
            INFO,
            f"Round active since {int(time.time() - start_time)}s, waiting for more clients to join...",
        )
        time.sleep(10)

    if not stop_event.is_set():
        redis_client.publish(f"round:{current_round}:stop", f"END ROUND {current_round}")
        log(
            INFO,
            f"Signaled ROUND END after {int(time.time() - start_time)}s with {len(list(grid.get_node_ids()))} connected Flower clients.",
        )

def get_mci(base_url: str, client_name: str) -> float:
    """Fetch current MCI index for a client's microgrid."""
    url = f"{base_url}/microgrids/{client_name}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()["grid_signals"]["mci_index"]
