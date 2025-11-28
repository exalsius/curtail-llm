# Pilot ✈

Federated GPT pretraining with nanochat - distributed training framework for large language models with sporadic client availability and time-based rounds.

## Features

- **Federated GPT Pretraining**: Train GPT models across multiple nodes using Flower framework
- **Redis Coordination**: Server-controlled stopping and shard assignment via Redis
- **Server-Controlled Rounds**: Dynamic round control based on client availability

## Installation & Setup

```bash
uv venv
uv sync
source .venv/bin/activate
```

### Redis Setup

The system uses Redis for coordination between server and clients:

```bash
docker run -d -p 6379:6379 redis:latest
```

For distributed deployment, use managed Redis (AWS ElastiCache, Redis Cloud) and configure `redis_url` in `pyproject.toml`.

### Rust Tokenizer

The nanochat tokenizer uses a high-performance Rust BPE implementation.
To install Rust / Cargo and build the rustbpe tokenizer extension, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Tokenizer Training

Before running federated training, prepare the tokenizer:

```bash
python -m nanochat.dataset -n 240  # Download dataset for tokenizer training (~22GB of FineWeb-EDU data)
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536  # Train BPE tokenizer on ~4B characters
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
- **Server-Controlled Stopping**: Redis pub/sub for dynamic round control
- **Distributed Sharding**: Redis-based coordination for shard assignments

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
