# Pilot ✈

Federated nanochat pretraining on volatile FL clients.


## Setup

### Installation (all nodes)

Clone the repository and install dependencies:

```bash
uv venv
uv sync
source .venv/bin/activate
```

### Data & Tokenizer Setup (all nodes)

The nanochat tokenizer uses a Rust BPE implementation.
To install Rust / Cargo and build the tokenizer extension, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

Next, prepare the tokenizer:

```bash
python -m nanochat.dataset -n 240  # Download dataset for tokenizer training (~22GB of FineWeb-EDU data)
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536  # Train BPE tokenizer on ~4B characters
python -c "from nanochat.tokenizer import get_tokenizer; print(f'Vocab size: {get_tokenizer().get_vocab_size()}')"
```

### Redis Setup (head node only)

The system uses Redis for coordination between server and clients. 
Redis must be accessible by all nodes, run via:

```bash
apt-get update
apt-get install redis-server
redis-server --daemonize yes
```

## Quick Start

### Deployment

Deploy across multiple physical nodes:

1. Make sure Redis is running.
```bash
redis-server --daemonize yes
# docker run -d -p 6379:6379 redis:latest

redis-cli ping
```

2. Start Flower SuperLink (coordinator):
```bash
flower-superlink --insecure
```

3. Start SuperNodes (workers) on each GPU:
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
4. Run the Flower app:
```bash
flwr run . local-deployment
```

You can override config values from command line:

```bash
flwr run . local-deployment --run-config "lr=0.0005"
```

### Vanilla nanochat Training

For comparison:

```bash
# Single GPU
python nanochat_train.py --num-steps 50 --batch-size 2 --max-length 512

# Multi-GPU with DDP
torchrun --standalone --nproc_per_node=2 nanochat_train.py \
  --num-steps 200 --batch-size 2 --max-length 2048 \
  --wandb-project nanochat_vanilla
```

### Local Simulation

Best for development and testing:

```bash
flwr run . local-simulation-gpu --stream
```


## Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/base_eval.py --checkpoint final_model.pt
```
