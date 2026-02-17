# curtail-llm

Distributed LLM pretraining during renewable curtailment windows.

This prototype trains a 561M-parameter transformer ([nanochat](https://github.com/KellerJordan/modded-nanogpt) d20) across geographically distributed GPU clusters, scheduling training only when regional renewable curtailment is detected. 
- Training is coordinated via the [Flower](https://flower.ai/) federated learning framework
- Nodes are elastically added/removed using [Exalsius](https://www.exalsius.ai/) and a custom [Kubernetes operator](https://github.com/exalsius/flower-operator)
- Energy system dynamics are simulated by [Vessim](https://vessim.readthedocs.io/en/latest/), with curtailment periods derived from real-world marginal carbon intensity traces provided by [WattTime](https://watttime.org/).

Depending on how many sites experience curtailment at any given time, the system operates in one of three regimes:
- **No curtailment** -- Training is paused.
- **One site curtailed** -- That site trains locally using standard data-parallel training.
- **Multiple sites curtailed** -- Sites train concurrently and synchronize periodically via federated aggregation.

Even when curtailment windows are rare and sporadic, spreading training across multiple regions recovers enough compute to match single-site training quality while cutting operational carbon emissions to 5--12% of baseline levels.


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
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/
python -m nanochat.dataset -n 240  # Download dataset for tokenizer training (~22GB of FineWeb-EDU data)
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536  # Train BPE tokenizer on ~4B characters, takes about 3min
python -c "from nanochat.tokenizer import get_tokenizer; print(f'Vocab size: {get_tokenizer().get_vocab_size()}')"
```

### Redis Setup (head node only)

The system uses Redis for coordination between server and clients.
Redis must be accessible by all nodes, run via:

```bash
sudo apt-get update
sudo apt-get install redis-server
redis-server --daemonize yes
redis-cli ping
```

## Deployment

Deploy across multiple physical nodes:

0. Make sure Redis is running.
```bash
redis-server --daemonize yes
redis-cli ping
```

1. Start the energy simulation (Vessim):
```bash
tmux new -A -s vessim

cd /workspace/pilot
source .venv/bin/activate

python energy_simulation.py
```

2. Start Flower SuperLink (coordinator):
```bash
tmux new -A -s superlink

cd /workspace/pilot
source .venv/bin/activate
export WANDB_API_KEY=<your-key>

flower-superlink --insecure
```

3. Start SuperNodes (workers) on each GPU:
```bash
# Client 0
tmux new -A -s client_0

cd /workspace/pilot
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/

CUDA_VISIBLE_DEVICES=0,1 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config 'name="client_0" partition-id=0'
```

```bash
# Client 1
tmux new -A -s client_1

cd /workspace/pilot
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/

CUDA_VISIBLE_DEVICES=4,5,6,7 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config 'name="client_1" partition-id=1'
```

4. Run the Flower app:
```bash
cd /workspace/pilot
source .venv/bin/activate
flwr run . local-deployment --stream
```

You can override config values from the command line:

```bash
flwr run . local-deployment --run-config "lr=0.0005" --stream
```

### Local Simulation

Best for development and testing:

```bash
flwr run . local-simulation-gpu --stream
```

### Vanilla nanochat Baseline

```bash
tmux new -A -s baseline

cd /workspace/pilot
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m scripts.base_train -- --depth 20 --target_param_data_ratio 20 \
  --device_batch_size 8 --run baseline
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python scripts/base_eval.py --checkpoint final_model.pt
```
