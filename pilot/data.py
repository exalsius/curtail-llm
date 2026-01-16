"""Core data sharding infrastructure for federated learning."""

import os
from collections import deque
from logging import INFO

import torch
import pyarrow.parquet as pq
from flwr.common import log

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

FlwrNodeId = int


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

    def assign_workers(self, flwr_node_ids: list[FlwrNodeId]):
        """Assign workers to shards with least progress.

        Distributes incomplete shards evenly among workers, prioritizing
        in-progress shards first, then unstarted ones.

        Returns:
            dict: {flwr_node_id: {"shard_ids": [...], "shard_starts": [...]}}
        """
        # Get incomplete shards (processed_rows < total_rows)
        incomplete_shards = [
            (shard_id, state["processed_rows"])
            for shard_id, state in self.shard_states.items()
            if state["processed_rows"] < state["total_rows"]
        ]

        if not incomplete_shards:
            # All shards complete, return empty assignments
            return {flwr_node_id: {"shard_ids": [], "shard_starts": []} for flwr_node_id in flwr_node_ids}

        # Sort by processed_rows ascending (in-progress first)
        incomplete_shards.sort(key=lambda x: x[1], reverse=True)

        # Distribute shards evenly among workers (round-robin) in gRPC format
        assignments = {flwr_node_id: {"shard_ids": [], "shard_starts": []} for flwr_node_id in flwr_node_ids}
        for i, (shard_id, start_row) in enumerate(incomplete_shards):
            flwr_node_id = flwr_node_ids[i % len(flwr_node_ids)]
            assignments[flwr_node_id]["shard_ids"].append(shard_id)
            assignments[flwr_node_id]["shard_starts"].append(start_row)

        return assignments

    def update(self, shard_ids: list[int], shard_rows: list[int]):
        """Update multiple shard states after training.

        Args:
            shard_ids: List of shard IDs
            shard_rows: List of absolute row positions for each shard
        """
        for shard_id, new_row in zip(shard_ids, shard_rows):
            # The client reports the 0-indexed row it has processed. To get the
            # number of rows processed so far, we use `new_row + 1`.
            num_processed = new_row + 1
            if num_processed > self.shard_states[shard_id].get("processed_rows", 0):
                self.shard_states[shard_id]["processed_rows"] = num_processed

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


def fl_shard_dataloader(shard_assignments, B, T, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", rank=0, world_size=1):
    """Process multiple parquet shards sequentially for FL training.

    Args:
        shard_assignments: List of (shard_id, start_row) tuples
        B: Batch size
        T: Sequence length
        tokenizer_threads: Tokenizer threads
        tokenizer_batch_size: Tokenization batch size
        device: Device type
        rank: Worker rank (for DDP)
        world_size: Total workers (for DDP)

    Yields:
        (inputs, targets, shard_id, current_row)
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

        log(INFO, f"Processing shard {shard_id}: rows {start_row}-{total_rows} (rank {rank}/{world_size})")

        # Read entire shard, skip rows before start_row
        table = pf.read()
        texts = table.column('text').to_pylist()[start_row:]
        
        # This rank processes all texts in its assigned shard (no further subsampling)
        my_texts = texts

        # Process in batches
        for i in range(0, len(my_texts), tokenizer_batch_size):
            batch = my_texts[i:i+tokenizer_batch_size]
            token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)

            for j, tokens in enumerate(token_lists):
                token_buffer.extend(tokens)
                
                # Calculate global current_row relative to the start of this shard
                local_idx = i + j
                current_row = start_row + local_idx 

                # Yield batches
                while len(token_buffer) >= needed_tokens:
                    tokens_list = [token_buffer.popleft() for _ in range(needed_tokens)]
                    use_cuda = device == "cuda"
                    scratch = torch.tensor(tokens_list, dtype=torch.long, pin_memory=use_cuda)
                    inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
                    targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
                    yield inputs, targets, shard_id, current_row, current_row / total_rows

                # current_row will exceed total_rows if the last batch goes past the end
                # The condition check should be on the original shard's total_rows, not current_row
                # The loop for (i,j) will naturally stop when my_texts is exhausted.
                # The "if current_row >= total_rows" check here is redundant/misleading now.

        # Clear buffer between shards
        if token_buffer:
            log(INFO, f"Discarding {len(token_buffer)} tokens from shard {shard_id}")
            token_buffer.clear()
