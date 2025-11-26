# Nanochat Integration Guide

This document explains how to use the nanochat GPT model integration for federated learning.

## Overview

The nanochat integration allows you to:
1. **Vanilla Training**: Train the nanochat d20 or d32 model using standard PyTorch DDP
2. **Federated Training**: Train the same model using Flower's federated learning framework with sporadic client availability and 5-minute aggregation rounds

## Setup (One-Time)

```bash
# Download data for tokenizer training and start downloading rest for training
python -m nanochat.dataset -n 16      # ~1.5GB for tokenizer (blocks until done)
python -m nanochat.dataset -n 100 &   # ~10GB total dataset (runs in background)

# Train tokenizer on ~4B characters
cd nanochat/scripts
python tok_train.py --max_chars=4000000000 --vocab_size=65536

# Verify setup
python -c "from nanochat.tokenizer import get_tokenizer; print(f'Vocab: {get_tokenizer().get_vocab_size()}')"
```

**Tokenizer location:** `~/.cache/nanochat/tokenizer/` (shared across all training runs)
**Custom location:** `export NANOCHAT_BASE_DIR=/path/to/shared/storage`

## Quick Start

### Vanilla Training

```bash
# Single GPU
python nanochat_train.py --num-steps 50 --batch-size 2 --max-length 512

# Multi-GPU DDP (2 GPUs)
torchrun --standalone --nproc_per_node=2 nanochat_train.py \
  --num-steps 200 --batch-size 2 --max-length 2048 \
  --wandb-project nanochat_vanilla --run-name d20_ddp
```

Resume via `--resume-from path/to/checkpoint.pt`. Checkpoints saved to `nanochat_runs/`.

### Federated Training

```bash
# Configure pyproject.toml with task_type="nanochat"
flwr run . local-simulation-gpu
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

### ✅ Completed
1. Streaming data with OG parquet loader
2. Bits-per-byte evaluation metrics
3. Direct OG nanochat imports (removed pilot/nanochat copy)

### ℹ️ Data Sharding

**Full dataset on all nodes** - Time-based rounds (5 min) naturally partition data across clients. Simple and efficient for 1-3 nodes. No explicit shard management needed.

## Testing

### Test Import
```bash
python -c "import nanochat.gpt as gpt; print('Success!')"
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
