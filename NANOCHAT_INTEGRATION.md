# Nanochat Integration Guide

This document explains how to use the nanochat GPT model integration for federated learning.

## Overview

The nanochat integration allows you to:
1. **Vanilla Training**: Train the nanochat d20 or d32 model using standard PyTorch DDP
2. **Federated Training**: Train the same model using Flower's federated learning framework with sporadic client availability and 5-minute aggregation rounds

## Quick Start

### Vanilla Training

Run the standard (non-FL) trainer to sanity-check the nanochat stack or benchmark against Flower. This script supports single GPU runs as well as multi-GPU DDP via `torchrun`.

```bash
# Single GPU
python nanochat_train.py --num-steps 50 --batch-size 2 --max-length 512 --output-dir nanochat_runs/test

# Multi-GPU (2 GPUs)
torchrun --standalone --nproc_per_node=2 nanochat_train.py \
  --num-steps 200 --batch-size 2 --max-length 2048 \
  --wandb-project nanochat_vanilla --run-name d20_ddp
```

Before running full training, download a few FineWeb-EDU shards:

```bash
python -m pilot.nanochat.dataset -n 10 -w 4
```

The trainer streams data directly from the parquet files and saves checkpoints to `nanochat_runs/` by default. Resume a run via `--resume-from path/to/checkpoint.pt`.

### Federated Training (Current Implementation)

The federated training mode is now implemented and ready to test:

```bash
# Run federated training with GPU
flwr run . local-simulation-gpu

# Or run with CPU (slower)
flwr run .
```

### Configuration

Edit `pyproject.toml` to configure the nanochat training:

```toml
[tool.flwr.app.config]
task_type = "nanochat"
model_type = "nanochat_d20"  # or "nanochat_d32"
dataset_name = "karpathy/fineweb-edu-100b-shuffle"
num_shards = 2  # Number of data shards for federated learning
num_rounds = 3  # Number of aggregation rounds
lr = 0.0006  # Learning rate (nanochat's original)
weight_decay = 0.01
batch_size = 2  # Sequences per GPU
max_length = 512  # Sequence length (original: 2048)
gradient_accumulation_steps = 1  # Set to 8 to simulate 8 GPUs on 1 GPU
```

### Model Variants

- **nanochat_d20**: 20 layers, 560M parameters (original speedrun config)
- **nanochat_d32**: 32 layers, 1.9B parameters (larger variant)

## Architecture

### Model Architecture (d20)
- Vocabulary size: 65,536 tokens
- Layers: 20
- Hidden size: 1,280
- Attention heads: 10
- KV heads: 10 (Multi-Query Attention)
- Context length: 2,048 tokens
- Parameters: ~560M

### Key Features
- **RoPE** (Rotary Position Embeddings)
- **RMSNorm** (no learnable parameters)
- **ReLU²** activation in MLP
- **Untied** token embedding and lm_head weights
- **BF16** mixed precision training
- **Group-Query Attention (GQA)** for efficient inference

## Current Status

### ✅ Implemented
1. **Core Model**: GPT model ported from karpathy/nanochat
2. **Optimizers**: Muon and DistAdamW optimizers
3. **Utilities**: Common utilities (logging, distributed setup, etc.)
4. **Tokenizer**: HuggingFace BPE tokenizer with GPT-2 fallback
5. **Data Loading**: Streaming parquet loader (structure in place)
6. **FL Integration**: Complete Flower integration for federated training
7. **Server**: nanochat model detection, initialization, and evaluation hooks
8. **Client**: nanochat task routing and training
9. **Vanilla Trainer**: `nanochat_train.py` for local/torchrun DDP runs

### 🚧 Pending
1. **Streaming Data**: Full integration of streaming parquet loader (currently using dummy data)
2. **Tokenizer Training**: Train custom nanochat tokenizer (currently using GPT-2 fallback)
3. **Server Evaluation**: Implement proper evaluation on streaming data

### 📝 TODO
1. Download FineWeb-EDU parquet files:
   ```bash
   python -m pilot.nanochat.dataset -n 10 -w 4  # Download first 10 shards
   ```

2. Train nanochat tokenizer (or use GPT-2 fallback):
   ```bash
   # TODO: Add tokenizer training script
   ```

3. Replace dummy data with streaming parquet loader in `pilot/nanochat_fl.py`

4. Implement server-side evaluation in `pilot/server_app.py`

## Testing

### Test Import
```bash
python -c "import pilot.nanochat.gpt as gpt; print('Success!')"
```

### Test Federated Training (Short Run)
```bash
# Edit pyproject.toml:
# - num_rounds = 3 (short test)
# - batch_size = 2 (small batch)
# - max_length = 512 (reduced context)

flwr run . local-simulation-gpu
```

## Original Nanochat Performance

From karpathy/nanochat:
- **Hardware**: 8×H100 GPUs
- **Training time**: ~4 hours (3h51m)
- **Tokens processed**: 11.2B tokens
- **Cost**: ~$92-100
- **Dataset**: FineWeb-EDU (100B tokens total)

## Federated Training Advantages

1. **Sporadic Availability**: Clients can join/leave between rounds
2. **Time-Based Rounds**: Fixed 5-minute rounds (configurable via `ROUND_DURATION` in `pilot/server_app.py`)
3. **Data Sharding**: Each client trains on different data shard
4. **Shard Management**: Automatic assignment to least-progressed shards
5. **Distributed Scaling**: Can run on 1-3 nodes with arbitrary GPU counts

## File Structure

```
pilot/
├── nanochat/                    # Core nanochat module (ported from karpathy/nanochat)
│   ├── __init__.py
│   ├── gpt.py                  # GPT model implementation
│   ├── common.py               # Utilities (logging, distributed setup)
│   ├── muon.py                 # Muon optimizer
│   ├── adamw.py                # Distributed AdamW optimizer
│   ├── tokenizer.py            # BPE tokenizer (HuggingFace + RustBPE)
│   ├── dataloader.py           # Streaming data loader
│   └── dataset.py              # Parquet file utilities
├── nanochat_fl.py              # Flower integration (federated learning)
├── server_app.py               # Updated with nanochat support
└── client_app.py               # Updated with nanochat routing
```

## Dependencies

Added to `pyproject.toml`:
- `filelock`: For thread-safe file downloads
- `pyarrow`: For parquet file reading
- `requests`: For downloading parquet files
- `tokenizers`: HuggingFace tokenizers
- `rustbpe` (optional): Rust-based BPE tokenizer
- `tiktoken` (optional): Efficient inference tokenizer

## References

- Original nanochat: https://github.com/karpathy/nanochat
- FineWeb-EDU dataset: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
- Flower documentation: https://flower.ai/docs/

## Next Steps

1. Test the basic federated training with dummy data
2. Download FineWeb-EDU parquet files
3. Integrate streaming parquet loader
4. Train custom tokenizer (or continue with GPT-2 fallback)
5. Implement server-side evaluation
6. Create vanilla DDP training script for comparison
7. Run full training experiments comparing vanilla vs federated performance
