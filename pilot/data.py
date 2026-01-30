import os
from collections import deque
from logging import INFO

import torch
import pyarrow.parquet as pq
from pilot.logger import log

from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

FlwrNodeId = int


class ShardManager:
    """Manages multiple data shards, tracking progress through the dataset."""

    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        base_dir = get_base_dir()
        data_dir = os.path.join(base_dir, "base_data")
        self.shard_states = {}
        for shard_id in range(num_shards):
            filepath = os.path.join(data_dir, f"shard_{shard_id:05d}.parquet")
            total_rows = 53248  # Default typical size
            if os.path.exists(filepath):
                pf = pq.ParquetFile(filepath)
                total_rows = pf.metadata.num_rows
            self.shard_states[shard_id] = {"total_rows": total_rows, "processed_rows": 0}

    def assign_workers(self, flwr_node_ids: list[FlwrNodeId]):
        """Assign workers to shards with least progress."""
        incomplete_shards = [
            (s_id, s["processed_rows"])
            for s_id, s in self.shard_states.items()
            if s["processed_rows"] < s["total_rows"]
        ]
        if not incomplete_shards:
            return {nid: {"shard_ids": [], "shard_starts": []} for nid in flwr_node_ids}

        incomplete_shards.sort(key=lambda x: x[1], reverse=True)
        assignments = {nid: {"shard_ids": [], "shard_starts": []} for nid in flwr_node_ids}
        for i, (shard_id, start_row) in enumerate(incomplete_shards):
            flwr_node_id = flwr_node_ids[i % len(flwr_node_ids)]
            assignments[flwr_node_id]["shard_ids"].append(shard_id)
            assignments[flwr_node_id]["shard_starts"].append(start_row)
        return assignments

    def update(
        self,
        shard_ids: list[int],
        shard_rows: list[int],
        shard_totals: list[int] | None = None,
    ):
        """Update multiple shard states after training."""
        if shard_totals is None:
            shard_totals = [0] * len(shard_ids)
        for shard_id, new_row, total_rows in zip(shard_ids, shard_rows, shard_totals):
            if total_rows > 0:
                self.shard_states[shard_id]["total_rows"] = total_rows
            if new_row > self.shard_states[shard_id].get("processed_rows", 0):
                self.shard_states[shard_id]["processed_rows"] = new_row

    def is_complete(self) -> bool:
        """Check if all shards are complete."""
        return all(s["processed_rows"] >= s["total_rows"] for s in self.shard_states.values())

    def get_progress_summary(self) -> dict:
        """Get summary of shard progress."""
        total = sum(s["total_rows"] for s in self.shard_states.values())
        processed = sum(s["processed_rows"] for s in self.shard_states.values())
        return {
            "total_rows": total,
            "processed_rows": processed,
            "progress": processed / total if total > 0 else 0,
            "num_complete": sum(
                1 for s in self.shard_states.values() if s["processed_rows"] >= s["total_rows"]
            ),
            "num_total": self.num_shards,
        }

    def __repr__(self):
        summary = self.get_progress_summary()
        return (
            f"ShardManager(num_shards={self.num_shards}, "
            f"progress={summary['progress']:.1%}, "
            f"complete={summary['num_complete']}/{summary['num_total']})"
        )


def fl_shard_dataloader(
    shard_assignments,
    batch_size,
    sequence_length,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    rank=0,
    world_size=1,
):
    """Process multiple parquet shards sequentially for FL training."""
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "base_data")
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    needed_tokens = batch_size * sequence_length + 1
    token_buffer = deque()

    for shard_id, start_row in shard_assignments:
        filepath = os.path.join(data_dir, f"shard_{shard_id:05d}.parquet")
        if not os.path.exists(filepath):
            log(INFO, f"Rank {rank}: Shard {shard_id} not found, skipping.")
            continue

        pf = pq.ParquetFile(filepath)
        total_rows = pf.metadata.num_rows
        
        start_rg_idx, cumulative_rows = 0, 0
        for i in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(i).num_rows
            if cumulative_rows + rg_rows > start_row:
                start_rg_idx = i
                break
            cumulative_rows += rg_rows
        
        rg_idx = start_rg_idx
        if rank == 0:
            log(INFO, f"Processing shard {shard_id} starting at row {start_row}/{total_rows}")

        current_row = start_row - 1
        while rg_idx < pf.num_row_groups:
            if (rg_idx - start_rg_idx) % world_size == rank:
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()

                row_offset = 0
                if rg_idx == start_rg_idx:
                    row_offset = start_row - cumulative_rows
                    texts = texts[row_offset:]

                for i in range(0, len(texts), tokenizer_batch_size):
                    batch = texts[i : i + tokenizer_batch_size]
                    token_lists = tokenizer.encode(
                        batch, prepend=bos_token, num_threads=tokenizer_threads
                    )
                    for j, tokens in enumerate(token_lists):
                        token_buffer.extend(tokens)
                        current_row = cumulative_rows + row_offset + i + j
                        while len(token_buffer) >= needed_tokens:
                            tokens_list = [token_buffer.popleft() for _ in range(needed_tokens)]
                            use_cuda = "cuda" in device
                            scratch = torch.tensor(tokens_list, dtype=torch.long, pin_memory=use_cuda)
                            inputs = scratch[:-1].view(batch_size, sequence_length).to(device=device, non_blocking=use_cuda)
                            targets = scratch[1:].view(batch_size, sequence_length).to(device=device, non_blocking=use_cuda)
                            yield inputs, targets, shard_id, current_row, total_rows
            
            cumulative_rows += pf.metadata.row_group(rg_idx).num_rows
            rg_idx += 1
