import json
import threading
import time
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
    flwr_node_id: Optional[FlwrNodeId] = None
    _state: Literal["OFF", "STARTING", "IDLE", "TRAINING"] = "OFF"

    def update_provisioning(self, redis_client: Redis, provisioner: ExlsProvisioner, carbon_intensity: float):
        # Check whether the client should be provisioned
        if self._state == "OFF" and carbon_intensity < self.provision_threshold:
            self._state = "STARTING"
            provisioner.add_node(self.name)
        # Check whether the client should be deprovisioned
        elif self._state != "OFF" and carbon_intensity > self.deprovision_threshold:
            threading.Thread(target=self._deprovision, args=(provisioner,)).start()

    def start_training(self):
        assert self._state == "IDLE", f"Cannot start training: client '{self.name}' is {self._state}"
        self._state = "TRAINING"

    def stop_training(self):
        assert self._state == "TRAINING", f"Cannot stop training: client '{self.name}' is {self._state}"
        self._state = "IDLE"

    def _deprovision(self, provisioner: ExlsProvisioner):
        """Gracefully deprovision a client by signaling stop and waiting for completion."""
        redis_client = redis.from_url(self.redis_url)
        # Wrap up current round in case the client is still training
        if self._state == "TRAINING":
            log(INFO, f"Signaling client '{self.name}' to stop training")
            redis_client.publish(f"{self.name}:stop", f"DEPROVISION {self.name}")
            
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

    def __init__(
        self,
        clients: dict[str, Client],
        dataset_name: str,
        num_shards: int,
        debug_port_client: bool,
        redis_url: str,
        round_min_duration: int,
        provisioner: Optional[ExlsProvisioner],
        mci_api_url: str,
        wandb_project: Optional[str] = None,
    ):
        self.clients = clients
        self.dataset_name = dataset_name
        self.num_shards = num_shards
        self.debug_port_client = debug_port_client
        self.redis_url = redis_url
        self.round_min_duration = round_min_duration
        self.provisioner = provisioner
        self.mci_api_url = mci_api_url
        self.wandb_project = wandb_project
        self.redis_client = redis.from_url(redis_url)
        self.shard_manager = ShardManager(num_shards=num_shards)
        self.weighted_by_key = "batches_processed"
        self.global_tokens_processed = 0


    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t└──> Dataset: '%s'", self.dataset_name)
        log(INFO, "\t└──> Shard: '%s'", self.shard_manager)

    def _query_clients(self, grid: Grid, flwr_node_ids: list[FlwrNodeId]):
        log(INFO, f"Querying all {len(flwr_node_ids)} connected Flower nodes for their names...")
        messages = [
            Message(content=RecordDict({"config": ConfigRecord({})}), message_type=MessageType.QUERY, dst_node_id=flwr_node_id)
            for flwr_node_id in flwr_node_ids
        ]
        query_replies = grid.send_and_receive(messages=messages, timeout=10)
        for reply in query_replies:
            client_name = reply.content["config"]["name"]
            log(INFO, f" - Flower node {reply.metadata.src_node_id}: '{client_name}'")
            self.clients[client_name]._state = "IDLE"
            self.clients[client_name]._flwr_node_id = reply.metadata.src_node_id

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, base_config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        while not (flwr_node_ids := list(grid.get_node_ids())):
            log(INFO, "Waiting for a client to connect")
            time.sleep(1)

        # --- Server-side Scheduling ---
        # Work on a copy to preserve baseline LRs in base_config
        round_config = dict(base_config)
        total_batch_size = int(round_config["total_batch_size"])
        self.total_batch_size = total_batch_size
        current_step = self.global_tokens_processed // total_batch_size
        
        # The client will handle per-step scheduling. The server just passes the raw parameters.
        # num_iterations is already in base_config, derived from num_shards in server_app.py
        
        log(INFO, f"Round {server_round}: Step {current_step}, Passing scheduling parameters to clients.")
        # ------------------------------

        self._query_clients(grid, flwr_node_ids)
        log(INFO, "configure_train: Training on all %s Flower nodes", len(flwr_node_ids))

        # Get worker assignments from shard manager
        assignments = self.shard_manager.assign_workers(flwr_node_ids)
        progress = self.shard_manager.get_progress_summary()
        log(
            INFO,
            f"Shard progress: {progress['progress']:.1%} ({progress['processed_rows']:,}/{progress['total_rows']:,} rows, "
            f"{progress['num_complete']}/{progress['num_total']} complete)",
        )

        # Set clients to TRAINING state
        node_id_to_client = {c.flwr_node_id: c for c in self.clients.values() if c.flwr_node_id is not None}
        for flwr_node_id in flwr_node_ids:
            if flwr_node_id in node_id_to_client:
                node_id_to_client[flwr_node_id].start_training()

        round_config.update({
            "server_round": server_round,
            "dataset_name": self.dataset_name,
            "num_shards": self.num_shards,
            "redis_url": self.redis_url,
            "global_tokens_processed_start": self.global_tokens_processed,
        })
        if self.debug_port_client:
            round_config["debug_port_client"] = self.debug_port_client

        return [
            Message(
                content=RecordDict({"arrays": arrays, "config": ConfigRecord({**round_config, **assignments[flwr_node_id]})}),
                message_type=MessageType.TRAIN,
                dst_node_id=flwr_node_id,
            )
            for flwr_node_id in flwr_node_ids
        ]

    def _log_to_wandb(self, server_round: int, progress: dict, valid_replies: list[Message], metrics: MetricRecord | None):
        log_dict = {
            "server/num_clients": len(valid_replies),
            "data/progress": progress["progress"],
            "data/shards_complete": progress["num_complete"],
            "server/global_tokens_processed": self.global_tokens_processed,
        }
        if metrics:
            log_dict["server/train_loss"] = metrics.get("train_loss", 0)
            if "train_ppl" in metrics:
                log_dict["server/train_ppl"] = metrics.get("train_ppl", 0)
            if "actual_train_time" in metrics:
                log_dict["server/actual_train_time"] = metrics.get("actual_train_time", 0)

        for reply in valid_replies:
            client_metrics = reply.content["metrics"]
            client_prefix = f"client_{client_metrics['client_id']}"
            log_dict[f"{client_prefix}/batches_processed"] = client_metrics["batches_processed"]
            if "train_loss" in client_metrics:
                log_dict[f"{client_prefix}/train_loss"] = client_metrics["train_loss"]
            if "train_ppl" in client_metrics:
                log_dict[f"{client_prefix}/train_ppl"] = client_metrics["train_ppl"]
            if "actual_train_time" in client_metrics:
                log_dict[f"{client_prefix}/actual_train_time"] = client_metrics["actual_train_time"]

        summary_step = self.global_tokens_processed // self.total_batch_size if hasattr(self, "total_batch_size") else server_round
        log_dict["server/global_step"] = summary_step
        wandb.log(log_dict)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies)
        node_id_to_client = {c.flwr_node_id: c for c in self.clients.values() if c.flwr_node_id is not None}
        arrays, metrics = None, None
        if valid_replies:
            for reply in valid_replies:
                if reply.metadata.src_node_id in node_id_to_client:
                    node_id_to_client[reply.metadata.src_node_id].stop_training()
                
                reply_metrics = reply.content["metrics"]
                self.shard_manager.update(
                    shard_ids=reply_metrics.get("shard_ids", []),
                    shard_rows=reply_metrics.get("shard_rows", []),
                    shard_totals=reply_metrics.get("shard_totals"),
                )
                if "num_tokens_processed" in reply_metrics:
                    self.global_tokens_processed += int(reply_metrics["num_tokens_processed"])

            progress = self.shard_manager.get_progress_summary()
            log(
                INFO,
                f"[Round {server_round}] Shard progress: {progress['progress']:.1%} "
                f"({progress['processed_rows']:,}/{progress['total_rows']:,} rows, "
                f"{progress['num_complete']}/{progress['num_total']} complete)",
            )

            reply_contents = [msg.content for msg in valid_replies]
            arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
            metrics = aggregate_metricrecords(reply_contents, self.weighted_by_key)

            self._log_to_wandb(server_round, progress, valid_replies, metrics)

        if arrays is None:
            log(INFO, "aggregate_train: No valid results to aggregate.")
            log(INFO, replies)

        return arrays, metrics

    def configure_evaluate(
        self, server_round, arrays, base_config, grid
    ) -> Iterable[Message]:
        return []  # not used, server-side eval only

    def aggregate_evaluate(
        self, server_round, replies
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        return None, None  # not used, server-side eval only

    def _start_threads(self):
        if self.provisioner:
            threading.Thread(
                target=_provisioning_task,
                args=(self.clients, self.redis_client, self.provisioner, self.mci_api_url),
                daemon=True,
            ).start()
        threading.Thread(target=_poll_logs, args=(self.redis_url, self.clients), daemon=True).start()

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 1e100,
        timeout: float = None,
        train_config: Optional[ConfigRecord] = None,
    ):
        """Run time-based federated learning."""
        log(INFO, f"Starting {self.__class__.__name__} strategy:")
        log(INFO, f"\t├── ArrayRecord ({sum(len(a.data) for a in initial_arrays.values()) / (1024 ** 2):.2f} MB)")
        log(INFO, f"\t├── ConfigRecord (train): {config_to_str(train_config)}")
        self.summary()
        
        train_config = train_config or ConfigRecord()
        train_config["experiment_start_time"] = int(time.time())

        self._start_threads()

        start_time = time.time()
        arrays = initial_arrays
        for current_round in range(1, int(num_rounds + 1)):
            log(INFO, f"\n[ROUND {current_round}]")
            if self.shard_manager.is_complete():
                log(INFO, "All data shards processed. Terminating training.")
                break

            round_controller_thread = threading.Thread(
                target=_determine_round_end,
                args=(grid, self.redis_client, self.round_min_duration, current_round),
                daemon=True,
            )
            round_controller_thread.start()

            messages = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=messages, timeout=timeout)

            arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

            if agg_train_metrics:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)

        log(INFO, f"\nStrategy execution finished in {time.time() - start_time:.2f}s")
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


def _poll_logs(redis_url: str, clients: dict[str, Client]):
    """Poll client logs from Redis and push them to a queue."""
    log(INFO, "Starting Redis log polling task...")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    while True:
        try:
            for client_name in clients:
                log_key = f"logs:{client_name}"
                pipe = redis_client.pipeline()
                pipe.lrange(log_key, 0, -1)
                pipe.ltrim(log_key, 1, 0)
                log_entries_str = pipe.execute()[0]
                if not log_entries_str:
                    continue
                
                log(INFO, f"Fetched {len(log_entries_str)} log entries for {client_name} from Redis.")
                for log_entry_str in log_entries_str:
                    log_entry = json.loads(log_entry_str)
                    step = log_entry.pop("step", None)
                    if step is not None:
                        log_entry["client_step"] = step
                        wandb.log(log_entry)
            time.sleep(10)
        except Exception as e:
            log(INFO, f"Error in log polling task: {e}")
            time.sleep(30)


def _provisioning_task(
    clients: dict[str, Client], redis_client: Redis, provisioner: ExlsProvisioner, mci_api_url: str
):
    """Monitor clients for provisioning and deprovisioning decisions."""
    while True:
        for client in clients.values():
            mci = get_mci(mci_api_url, client.name)
            client.update_provisioning(redis_client, provisioner, mci)
        time.sleep(5)


def _determine_round_end(
    grid: Grid, redis_client: Redis, round_min_duration: float, current_round: int
):
    log(INFO, f"Round monitor started for ROUND {current_round}")
    start_time = time.time()
    time.sleep(round_min_duration)

    while len(list(grid.get_node_ids())) <= 1:
        log(INFO, f"Round active since {int(time.time() - start_time)}s, waiting for more clients to join...")
        time.sleep(10)

    redis_client.publish(f"round:{current_round}:stop", f"END ROUND {current_round}")
    log(INFO, f"🛑 Signaled END ROUND after {int(time.time() - start_time)}s with {len(list(grid.get_node_ids()))} connected Flower clients.")



def get_mci(base_url: str, client_name: str) -> float:
    """Fetch current MCI index for a client's microgrid."""
    url = f"{base_url}/microgrids/{client_name}"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()["grid_signals"]["mci_index"]
