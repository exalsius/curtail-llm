"""Core data sharding infrastructure for federated learning."""

import os
from collections import deque
from logging import INFO

import torch
import pyarrow.parquet as pq
from flwr.common import log

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


class ShardManager:
    """Manages multiple data shards, tracking progress through the dataset."""

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        # Initialize shard states by reading parquet files to get total rows
        base_dir = get_base_dir()
        data_dir = os.path.join(base_dir, "base_data")

        self.shard_states = {}
        for shard_id in range(num_shards):
            filepath = os.path.join(data_dir, f"shard_{shard_id:05d}.parquet")
            if os.path.exists(filepath):
                pf = pq.ParquetFile(filepath)
                total_rows = pf.metadata.num_rows
            else:
                # File doesn't exist yet, assume typical size
                total_rows = 53248  # typical size based on shard_00000.parquet

            self.shard_states[shard_id] = {
                "total_rows": total_rows,
                "processed_rows": 0,
            }

    def assign_workers(self, node_ids: list[int]):
        """Assign workers to shards with least progress.

        Distributes incomplete shards evenly among workers, prioritizing
        in-progress shards first, then unstarted ones.

        Returns:
            dict: {node_id: [(shard_id, start_row), ...]}
        """
        # Get incomplete shards (processed_rows < total_rows)
        incomplete_shards = [
            (shard_id, state["processed_rows"])
            for shard_id, state in self.shard_states.items()
            if state["processed_rows"] < state["total_rows"]
        ]

        if not incomplete_shards:
            # All shards complete, return empty assignments
            return {node_id: [] for node_id in node_ids}

        # Sort by processed_rows ascending (in-progress first)
        incomplete_shards.sort(key=lambda x: x[1])

        # Distribute shards evenly among workers (round-robin)
        assignments = {node_id: [] for node_id in node_ids}
        for i, (shard_id, start_row) in enumerate(incomplete_shards):
            node_id = node_ids[i % len(node_ids)]
            assignments[node_id].append((shard_id, start_row))

        return assignments

    def update(self, shard_updates: list[tuple[int, int]]):
        """Update multiple shard states after training.

        Args:
            shard_updates: List of (shard_id, rows_processed) tuples
        """
        for shard_id, rows_processed in shard_updates:
            self.shard_states[shard_id]["processed_rows"] += rows_processed

    def is_complete(self) -> bool:
        """Check if all shards are complete."""
        return all(
            state["processed_rows"] >= state["total_rows"]
            for state in self.shard_states.values()
        )

    def get_progress_summary(self) -> dict:
        """Get summary of shard progress."""
        total_rows = sum(s["total_rows"] for s in self.shard_states.values())
        processed_rows = sum(s["processed_rows"] for s in self.shard_states.values())
        return {
            "total_rows": total_rows,
            "processed_rows": processed_rows,
            "progress": processed_rows / total_rows if total_rows > 0 else 0,
            "num_complete": sum(1 for s in self.shard_states.values() if s["processed_rows"] >= s["total_rows"]),
            "num_total": self.num_shards,
        }

    def __repr__(self):
        summary = self.get_progress_summary()
        return (f"ShardManager(num_shards={self.num_shards}, "
                f"progress={summary['progress']:.1%}, "
                f"complete={summary['num_complete']}/{summary['num_total']})")


def fl_shard_dataloader(shard_assignments, B, T, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    """Process multiple parquet shards sequentially for FL training.

    Args:
        shard_assignments: List of (shard_id, start_row) tuples
        B: Batch size
        T: Sequence length
        tokenizer_threads: Tokenizer threads
        tokenizer_batch_size: Tokenization batch size
        device: Device type

    Yields:
        (inputs, targets, shard_id, rows_processed_in_shard)
    """
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "base_data")
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    needed_tokens = B * T + 1
    token_buffer = deque()

    for shard_id, start_row in shard_assignments:
        filepath = os.path.join(data_dir, f"shard_{shard_id:05d}.parquet")
        if not os.path.exists(filepath):
            log(INFO, f"Shard {shard_id} not found, skipping")
            continue

        pf = pq.ParquetFile(filepath)
        total_rows = pf.metadata.num_rows
        current_row = start_row
        rows_processed = 0

        log(INFO, f"Processing shard {shard_id}: rows {start_row}-{total_rows}")

        # Read entire shard, skip rows before start_row
        table = pf.read()
        texts = table.column('text').to_pylist()[start_row:]

        # Process in batches
        for i in range(0, len(texts), tokenizer_batch_size):
            batch = texts[i:i+tokenizer_batch_size]
            token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)

            for tokens in token_lists:
                token_buffer.extend(tokens)
                current_row += 1
                rows_processed += 1

                # Yield batches
                while len(token_buffer) >= needed_tokens:
                    tokens_list = [token_buffer.popleft() for _ in range(needed_tokens)]
                    use_cuda = device == "cuda"
                    scratch = torch.tensor(tokens_list, dtype=torch.long, pin_memory=use_cuda)
                    inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
                    targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
                    yield inputs, targets, shard_id, rows_processed

                if current_row >= total_rows:
                    break

        # Clear buffer between shards
        if token_buffer:
            log(INFO, f"Discarding {len(token_buffer)} tokens from shard {shard_id}")
            token_buffer.clear()
