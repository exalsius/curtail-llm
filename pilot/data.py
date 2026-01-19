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
        for s in self.shard_states.values():
            print(f"Shard progress: {s['processed_rows']}/{s['total_rows']}")
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

    This implementation is inspired by nanochat's `tokenizing_distributed_data_loader_with_state`
    but adapted for the Flower sharding mechanism. It reads data row-group by row-group
    to avoid loading entire large files into memory and correctly distributes the workload
    across DDP ranks.

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
        (inputs, targets, shard_id, current_row, progress_fraction)
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
            log(INFO, f"Rank {rank}: Shard {shard_id} not found, skipping.")
            continue

        pf = pq.ParquetFile(filepath)
        total_rows = pf.metadata.num_rows
        # Find the row group containing the start_row
        start_rg_idx = 0
        cumulative_rows = 0
        for i in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(i).num_rows
            if cumulative_rows + rg_rows > start_row:
                start_rg_idx = i
                break
            cumulative_rows += rg_rows
        
        # Each rank starts processing from a different row group offset
        rg_idx = start_rg_idx + rank
        
        log(INFO, f"Rank {rank}: Processing shard {shard_id} from row group {rg_idx} (starts at row {start_row}/{total_rows})")

        while rg_idx < pf.num_row_groups:
            rg = pf.read_row_group(rg_idx)
            # This is the starting row number of the current row group
            rg_start_row = sum(pf.metadata.row_group(i).num_rows for i in range(rg_idx))

            texts = rg.column('text').to_pylist()
            
            # If this is the first row group, we need to skip rows to get to the actual start_row
            if rg_idx == start_rg_idx:
                row_offset = start_row - rg_start_row
                texts = texts[row_offset:]
            else:
                row_offset = 0

            # Process texts in smaller batches for tokenization
            for i in range(0, len(texts), tokenizer_batch_size):
                batch = texts[i:i + tokenizer_batch_size]
                token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)

                for j, tokens in enumerate(token_lists):
                    token_buffer.extend(tokens)
                    
                    # Calculate the absolute row index within the shard
                    current_row = rg_start_row + row_offset + i + j

                    # Yield batches as they become available
                    while len(token_buffer) >= needed_tokens:
                        tokens_list = [token_buffer.popleft() for _ in range(needed_tokens)]
                        use_cuda = "cuda" in device
                        scratch = torch.tensor(tokens_list, dtype=torch.long, pin_memory=use_cuda)
                        
                        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
                        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
                        
                        progress_frac = (current_row + 1) / total_rows if total_rows > 0 else 0
                        yield inputs, targets, shard_id, current_row, progress_frac
            
            rg_idx += world_size # Move to the next row group for this rank

        # Clear buffer between shards to prevent data leakage
        if token_buffer:
            log(INFO, f"Rank {rank}: Discarding {len(token_buffer)} tokens from shard {shard_id}")
            token_buffer.clear()
