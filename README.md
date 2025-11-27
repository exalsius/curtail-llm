# Pilot ✈

Federated GPT pretraining with nanochat - distributed training framework for large language models with sporadic client availability and time-based rounds.

## Features

- **Federated GPT Pretraining**: Train GPT models across multiple nodes using Flower framework
- **Time-Based Rounds**: 5-minute training rounds for predictable runtime and sporadic client handling
- **Efficient Tokenization**: Custom Rust BPE tokenizer with 65K vocab (GPT-4 style)
- **Server-Side Evaluation**: Bits-per-byte metric on validation data
- **Memory Optimized**: Aggressive cleanup for multi-client simulation on single GPU

## Installation

```bash
# Create and activate virtual environment
uv venv
uv sync
source .venv/bin/activate
```

### Build Rust Tokenizer

The nanochat tokenizer uses a high-performance Rust BPE implementation:

```bash
# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build rustbpe tokenizer extension
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Setup (One-Time)

Before running federated training, you need to prepare the tokenizer:

```bash
# 1. Download dataset for tokenizer training (~22GB of FineWeb-EDU data)
python -m nanochat.dataset -n 240

# 2. Train BPE tokenizer on ~4B characters (creates 65K vocab)
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536

# 3. Verify tokenizer is ready
python -c "from nanochat.tokenizer import get_tokenizer; print(f'Vocab size: {get_tokenizer().get_vocab_size()}')"
```

**Tokenizer Storage:**
- Default: `~/.cache/nanochat/tokenizer/`
- Custom: Set `export NANOCHAT_BASE_DIR=/path/to/shared/storage`

## Quick Start

### Federated Training

Configure `pyproject.toml`:

```toml
[tool.flwr.app.config]
task_type = "nanochat"
model_type = "nanochat_d20"  # 560M params (or "nanochat_d32" for 1.9B params)
dataset_name = "karpathy/fineweb-edu-100b-shuffle"
num_shards = 2              # Number of data shards for FL
num_rounds = 50             # Number of training rounds
batch_size = 2              # Per-device batch size
max_length = 512            # Sequence length (2048 for full context)
gradient_accumulation_steps = 1  # Effective batch = batch_size * accum_steps
lr = 0.0006                 # Learning rate
weight_decay = 0.01
wandb_project = "pilot_flwr"  # Weights & Biases project name
```

Run federated training:

```bash
# Local simulation with GPU
flwr run . local-simulation-gpu

# With live output streaming
flwr run . local-simulation-gpu --stream
```

### Vanilla Training (Non-Federated)

For comparison or single-node training:

```bash
# Single GPU
python nanochat_train.py --num-steps 50 --batch-size 2 --max-length 512

# Multi-GPU with DDP
torchrun --standalone --nproc_per_node=2 nanochat_train.py \
  --num-steps 200 --batch-size 2 --max-length 2048 \
  --wandb-project nanochat_vanilla
```

### Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/base_eval.py --checkpoint final_model.pt
```

## Architecture

### Model Configurations

**nanochat_d20** (560M parameters):
- Vocabulary: 65,536 tokens (BPE)
- Layers: 20
- Hidden size: 1,280
- Attention heads: 10 (MQA with RoPE)
- Context length: 2,048 tokens
- Features: RMSNorm, ReLU² MLP, untied embeddings

**nanochat_d32** (1.9B parameters):
- Same architecture with 32 layers
- Hidden size: 2,048
- Attention heads: 16

### Training Features

- **Mixed Precision**: BFloat16 for efficient GPU utilization
- **Gradient Accumulation**: Simulate larger batch sizes
- **Time-Based Rounds**: Fixed 5-minute rounds for predictable runtime
- **Distributed Sharding**: ShardManager assigns clients to shards with least progress

### Data Strategy

**Full dataset on all nodes** - Each client has access to the full FineWeb-EDU dataset. Time-based rounds (5 minutes) naturally partition data across clients. This approach is simple and efficient for 1-3 sporadic nodes.

### Evaluation Metric

**Bits-Per-Byte (BPB)**: Vocabulary-size independent metric that measures compression quality. Lower is better (typical GPT-3 class models: ~0.7-0.9 BPB on web data).

## Deployment

### Local Simulation

Best for development and testing:

```bash
flwr run . local-simulation-gpu --stream
```

### Production Deployment

Deploy across multiple physical nodes:

**1. Start Flower SuperLink (coordinator):**
```bash
flower-superlink --insecure
```

**2. Start SuperNodes (workers) on each GPU:**
```bash
# Node 0
CUDA_VISIBLE_DEVICES=0 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0"

# Node 1
CUDA_VISIBLE_DEVICES=1 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config "partition-id=1"
```

**3. Run the server app:**
```bash
flwr run . local-deployment
```

## Configuration Override

You can override config values from command line:

```bash
flwr run . local-simulation-gpu --run-config "num-rounds=20 batch-size=4 lr=0.0005"
```

**Note**: Use hyphens (`num-rounds`, `batch-size`) in CLI, underscores in `pyproject.toml`.

## Project Structure

```
pilot/
├── nanochat_fl.py          # Nanochat FL integration
├── nanochat/               # Nanochat subpackage (core training logic)
│   ├── gpt.py              # GPT model architecture
│   ├── tokenizer.py        # BPE tokenizer interface
│   ├── dataloader.py       # Streaming data loader
│   ├── dataset.py          # Dataset utilities
│   ├── common.py           # Common utilities
│   ├── adamw.py            # AdamW optimizer
│   ├── muon.py             # Muon optimizer
│   ├── loss_eval.py        # Bits-per-byte evaluation
│   ├── checkpoint_manager.py  # Checkpoint management
│   ├── engine.py           # Training engine
│   ├── configurator.py     # Configuration
│   └── report.py           # Reporting
├── data.py                 # Shard management (ShardManager)
├── client_app.py           # Flower client app
└── server_app.py           # Flower server app (PilotAvg strategy)

rustbpe/                    # Rust BPE tokenizer extension
├── Cargo.toml
└── src/
    └── lib.rs

scripts/
├── base_train.py           # Standalone training
├── base_eval.py            # Model evaluation
└── tok_train.py            # Tokenizer training

nanochat_train.py           # Vanilla nanochat training (non-FL)
```

## How It Works

1. **Server Initialization**: Server loads nanochat model, extracts weights, then frees memory
2. **Shard Assignment**: ShardManager assigns each client to shard with least progress
3. **Client Training**: Each client trains for fixed time (5 min), tracks batches processed
4. **Aggregation**: Server aggregates weights using FedAvg weighted by batches processed
5. **Evaluation**: Server loads model fresh, evaluates on validation split, logs BPB to W&B
6. **Cleanup**: Aggressive memory cleanup between rounds for simulation mode

## Troubleshooting

**Rust compiler not found:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

**Tokenizer not found:**
```bash
# Ensure tokenizer training completed successfully
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536
```

**Out of memory:**
- Reduce `batch_size` or `max_length`
- Increase `gradient_accumulation_steps`
- Use `nanochat_d20` instead of `nanochat_d32`

**Negative time budget on clients:**
- Model loading took longer than `ROUND_DURATION` (5 min)
- Use smaller model or increase `ROUND_DURATION` in `pilot/server_app.py`

## Performance

**Single H100 GPU (nanochat_d20):**
- ~57 hours for full training run
- Sequence length: 2048
- Batch size: 2-4 depending on memory

**Multi-GPU Federated:**
- Linear speedup with additional nodes (time-based rounds)
- Efficient for 1-3 sporadic clients

## References

- **Flower Framework**: https://flower.ai/docs/
- **Nanochat (Karpathy)**: https://github.com/karpathy/nanochat
- **FineWeb-EDU Dataset**: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
- **Weights & Biases**: https://wandb.ai/

## License

MIT
