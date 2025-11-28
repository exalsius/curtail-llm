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
    log_strategy_start_info,
    aggregate_arrayrecords,
    aggregate_metricrecords,
    validate_message_reply_consistency,
)

from pilot.model import get_model
from pilot.data import ShardManager

app = ServerApp()

# Time-based round duration handled via configuration and Redis stop signals


class RoundStopMonitor:
    def __init__(self, grid: Grid, redis_client: redis.Redis, min_duration: int):
        self.grid = grid
        self.redis_client = redis_client
        self.min_duration = min_duration
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, server_round: int) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(server_round,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _run(self, server_round: int) -> None:
        start_time = time.time()
        channel = f"round:{server_round}:stop"
        while not self._stop_event.is_set():
            node_count = len(list(self.grid.get_node_ids()))
            elapsed = time.time() - start_time
            if node_count > 1 and elapsed >= self.min_duration:
                self.redis_client.publish(channel, "stop")
                break
            time.sleep(1)


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
        wandb_run_id: Optional[str] = None,
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
        self.wandb_run_id = wandb_run_id
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
        base_config["wandb_run_id"] = self.wandb_run_id
        base_config["wandb_project"] = self.wandb_project
        base_config["wandb_group"] = self.wandb_run_group
        if self.wandb_entity:
            base_config["wandb_entity"] = self.wandb_entity

        messages = []
        for node_id in node_ids:
            shard_list = assignments[node_id]
            # Convert list of tuples to lists for serialization
            config = ConfigRecord({
                **base_config,
                "shard_assignments": [[sid, start] for sid, start in shard_list]
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
            # shard_updates is a list of [shard_id, rows_processed] pairs
            shard_updates = [(sid, rows) for sid, rows in metrics["shard_updates"]]
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
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[callable] = None,
    ):
        """Run time-based federated learning for num_rounds."""
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        round_monitor = RoundStopMonitor(grid, self.redis_client, self.min_round_duration)

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # TRAINING
            round_monitor.start(current_round)
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )
            round_monitor.stop()
            agg_arrays, agg_train_metrics = self.aggregate_train(current_round, train_replies)

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # Note: No client-side evaluation

            # EVALUATION
            if evaluate_fn:
                log(INFO, "Global evaluation")
                eval_start = time.time()
                res = evaluate_fn(current_round, arrays)
                eval_time = time.time() - eval_start
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res
                if wandb.run:
                    wandb.log({"server/eval_time": eval_time}, step=current_round)

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
    num_rounds: int = context.run_config["num_rounds"]
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

    # Get number of supernodes from grid
    num_supernodes = len(list(grid.get_node_ids()))

    # Create run name with hyperparameters
    run_name = f"nodes{num_supernodes},sh{num_shards},bs{batch_size},rounds{num_rounds}"

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        group=run_name,  # Group all runs together
        config={
            "num_supernodes": num_supernodes,
            "learning_rate": lr,
            "batch_size": batch_size,
            "num_shards": num_shards,
            "num_rounds": num_rounds,
            "model_type": model_type,
            "dataset_name": dataset_name,
        }
    )
    log(INFO, "Wandb initialized with run_id: %s", wandb.run.id)

    wandb_run_id = wandb.run.id
    wandb_run_group = run_name

    # Load global model
    max_length = context.run_config.get("max_length", 2048)
    global_model = get_model(model_type, max_length)

    arrays = ArrayRecord(global_model.state_dict())

    redis_client = redis.from_url(redis_url)

    strategy = PilotAvg(
        dataset_name=dataset_name,
        num_shards=num_shards,
        debug_port_client=context.run_config.get("debug_port_client", None),
        redis_url=redis_url,
        redis_client=redis_client,
        min_round_duration=min_round_duration,
        wandb_run_id=wandb_run_id,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_group=wandb_run_group,
    )

    # Get max_length from run_config
    max_length = context.run_config.get("max_length", 2048)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(dict(lr=lr)),
        num_rounds=num_rounds,
        evaluate_fn=lambda round, arrays: global_evaluate(
            round, arrays, model_type, dataset_name, batch_size, max_length
        ),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

    # Finish wandb run
    wandb.finish()
    log(INFO, "Wandb run finished")


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    model_type: str,
    dataset_name: str,
    batch_size: int = 32,
    max_length: int = 2048,
) -> MetricRecord:
    """Evaluate nanochat model on validation data.

    Args:
        server_round: Current training round
        arrays: Model weights to evaluate
        model_type: Type of model (nanochat)
        dataset_name: Name of the dataset
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        MetricRecord with bits-per-byte metric
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    log(INFO, f"Server evaluation on {dataset_name}")

    # Load model
    model = get_model(model_type, max_length)
    model.load_state_dict(arrays.to_torch_state_dict(), strict=True)
    model.to(device)

    # Load tokenizer and create token_bytes tensor
    from nanochat.tokenizer import get_tokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state
    from nanochat.loss_eval import evaluate_bpb

    tokenizer = get_tokenizer()

    # Create token_bytes tensor: maps token ID to byte length
    # Special tokens (BOS, EOS, etc.) get 0 bytes to exclude them from metric
    vocab_size = tokenizer.get_vocab_size()
    token_bytes_list = []
    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            # Exclude special tokens by setting their byte count to 0
            if token_str in ["<|endoftext|>", "<|bos|>", "<|eos|>"]:
                token_bytes_list.append(0)
            else:
                token_bytes_list.append(len(token_str.encode('utf-8')))
        except:
            token_bytes_list.append(0)  # Invalid tokens get 0

    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int64, device=device)

    # Create eval dataloader using validation split
    eval_loader = tokenizing_distributed_data_loader_with_state(
        B=batch_size,
        T=max_length,
        split="val",
        tokenizer_threads=4,
        tokenizer_batch_size=128,
        device=device.type,
    )

    # Evaluate using bits-per-byte metric
    eval_steps = 100  # Number of batches to evaluate
    bpb = evaluate_bpb(model, eval_loader, eval_steps, token_bytes)

    log(INFO, f"Evaluation complete: bits-per-byte = {bpb:.4f}")

    wandb.log({"server/eval_bpb": bpb,}, step=server_round)

    return MetricRecord({"bits_per_byte": bpb})
