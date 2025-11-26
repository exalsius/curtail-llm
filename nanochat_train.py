"""
Standalone nanochat training script (vanilla PyTorch, no Flower).

This script reuses the nanochat modules shipped with the federated stack but
trains in the traditional single-node / DDP fashion so it can be compared
with the upstream reference implementation.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from pilot.nanochat.common import (
    DummyWandb,
    compute_cleanup,
    compute_init,
    print0,
    print_banner,
)
from pilot.nanochat.dataloader import tokenizing_distributed_data_loader_with_state
from pilot.nanochat.gpt import GPT, GPTConfig
from pilot.nanochat_fl import get_nanochat_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vanilla nanochat training")
    parser.add_argument("--model-type", default="nanochat_d20", choices=["nanochat_d20", "nanochat_d32"])
    parser.add_argument("--max-length", type=int, default=2048, help="Sequence length (T)")
    parser.add_argument("--batch-size", type=int, default=2, help="Sequences per device (B)")
    parser.add_argument("--num-steps", type=int, default=100, help="Optimiser steps to run")
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=0, help="Run evaluation every N steps (0 = skip)")
    parser.add_argument("--eval-batches", type=int, default=10, help="Batches per eval run")
    parser.add_argument("--output-dir", type=Path, default=Path("nanochat_runs"))
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Save checkpoint every N steps (0 = final only)")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--run-name", default="nanochat-vanilla")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--tokenizer-threads", type=int, default=4)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    return parser.parse_args()


def init_wandb(args: argparse.Namespace):
    if args.wandb_project is None:
        return DummyWandb()

    import wandb

    mode = "online"
    if os.environ.get("WANDB_DISABLED") == "true":
        mode = "disabled"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args),
        mode=mode,
    )
    return wandb


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[int, Optional[Dict]]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    loader_state = checkpoint.get("loader_state")
    print0(f"Resumed from checkpoint {path} at step {step}")
    return step, loader_state


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loader_state: Optional[Dict],
    extra: Optional[Dict] = None,
) -> None:
    state_dict = {
        "step": step,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loader_state": loader_state,
    }
    if extra:
        state_dict.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)
    print0(f"Saved checkpoint to {path}")


def format_tokens(num_tokens: int) -> str:
    if num_tokens >= 1e9:
        return f"{num_tokens / 1e9:.2f}B"
    if num_tokens >= 1e6:
        return f"{num_tokens / 1e6:.2f}M"
    if num_tokens >= 1e3:
        return f"{num_tokens / 1e3:.2f}K"
    return str(num_tokens)


def run_eval(model: torch.nn.Module, eval_loader, steps: int, device_type: str, precision: str) -> float:
    model.eval()
    autocast = torch.autocast if device_type in {"cuda", "mps"} and precision != "fp32" else None
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(steps):
            inputs, targets, _ = next(eval_loader)
            ctx = (
                autocast(device_type=device_type, dtype=dtype, enabled=True)
                if autocast
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with ctx:
                loss = model(inputs, targets=targets)
            total_loss += loss.detach().float().item()
    model.train()
    return total_loss / max(1, steps)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.expanduser()

    ddp, rank, local_rank, world_size, device = compute_init(args.device)
    print_banner()
    print0(f"Starting vanilla nanochat training ({args.model_type}) on {world_size} rank(s)")

    config: GPTConfig = get_nanochat_config(args.model_type, args.max_length)
    model = GPT(config)
    model.init_weights()
    model.to(device)

    if ddp and device.type == "cuda":
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    start_step = 0
    resume_state = None
    if args.resume_from:
        start_step, resume_state = load_checkpoint(args.resume_from, model, optimizer)
        start_step += 1

    train_loader = tokenizing_distributed_data_loader_with_state(
        B=args.batch_size,
        T=args.max_length,
        split="train",
        tokenizer_threads=args.tokenizer_threads,
        tokenizer_batch_size=args.tokenizer_batch_size,
        device=device.type,
        resume_state_dict=resume_state,
    )

    eval_loader = None
    if args.eval_interval > 0:
        eval_loader = tokenizing_distributed_data_loader_with_state(
            B=args.batch_size,
            T=args.max_length,
            split="val",
            tokenizer_threads=args.tokenizer_threads,
            tokenizer_batch_size=args.tokenizer_batch_size,
            device=device.type,
        )

    autocast_enabled = device.type in {"cuda", "mps"} and args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    wandb = init_wandb(args)

    total_tokens_per_step = args.batch_size * args.max_length * world_size * args.gradient_accumulation_steps
    print0(f"Tokens per optimizer step: {format_tokens(total_tokens_per_step)}")

    model.train()
    last_save = time.time()

    progress = tqdm(
        range(start_step, args.num_steps),
        disable=rank != 0,
        desc="Training",
        initial=start_step,
        total=args.num_steps,
    )

    loader_state = resume_state
    optimizer.zero_grad(set_to_none=True)

    for step in progress:
        for accum in range(args.gradient_accumulation_steps):
            inputs, targets, loader_state = next(train_loader)
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled)
            with ctx:
                loss = model(inputs, targets=targets)
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step = step + 1
        if rank == 0 and args.log_interval and global_step % args.log_interval == 0:
            tokens_seen = format_tokens(global_step * total_tokens_per_step)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            wandb.log(
                {
                    "train_loss": loss.detach().float().item(),
                    "tokens": global_step * total_tokens_per_step,
                    "tokens_readable": tokens_seen,
                    "step": global_step,
                },
                step=global_step,
            )

        if (
            rank == 0
            and args.checkpoint_interval
            and global_step % args.checkpoint_interval == 0
        ):
            ckpt_path = args.output_dir / f"{args.run_name}_step{global_step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, global_step, loader_state)
            last_save = time.time()

        if (
            rank == 0
            and args.eval_interval > 0
            and eval_loader is not None
            and global_step % args.eval_interval == 0
        ):
            eval_loss = run_eval(model, eval_loader, args.eval_batches, device.type, args.precision)
            wandb.log({"eval_loss": eval_loss, "step": global_step}, step=global_step)
            print0(f"[eval] step {global_step}: loss={eval_loss:.4f}")

    if rank == 0:
        final_path = args.output_dir / f"{args.run_name}_final.pt"
        save_checkpoint(final_path, model, optimizer, args.num_steps, loader_state)
        print0(f"Finished training in {(time.time() - last_save):.1f}s since last checkpoint")

    if hasattr(wandb, "finish"):
        wandb.finish()

    compute_cleanup()


if __name__ == "__main__":
    main()
