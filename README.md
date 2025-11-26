# Pilot ✈

Federated learning framework supporting vision models, LLMs, and nanochat GPT models with sporadic client availability and time-based rounds.

## Installation

```bash
# Install pilot package
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,test]"
```

## Setup (One-Time for Nanochat)

```bash
# Download dataset for tokenizer training (~22GB)
python -m pilot.nanochat.dataset -n 240

# Train tokenizer on ~4B characters
python scripts/tok_train.py --max_chars=4000000000 --vocab_size=65536

# Verify setup
python -c "from pilot.nanochat.tokenizer import get_tokenizer; print(f'Vocab: {get_tokenizer().get_vocab_size()}')"
```

**Tokenizer location:** `~/.cache/nanochat/tokenizer/` (shared across all runs)
**Custom location:** `export NANOCHAT_BASE_DIR=/path/to/shared/storage`

## Quick Start

### Vision Task (CIFAR-10, ImageNet)

```bash
# Configure pyproject.toml
[tool.flwr.app.config]
task_type = "vision"
model_type = "efficientnet_b0"  # or "simple_cnn"
dataset_name = "uoft-cs/cifar10"
num_rounds = 100
batch_size = 128

# Run
flwr run . local-simulation-gpu
```

### LLM Task (Qwen2.5-7B + LoRA)

```bash
[tool.flwr.app.config]
task_type = "llm"
model_type = "Qwen/Qwen2.5-7B"
dataset_name = "HuggingFaceH4/ultrachat_200k"
num_rounds = 50
batch_size = 4
max_length = 512
gradient_accumulation_steps = 4

# Run
flwr run . local-simulation-gpu
```

### Medical Task (Alpaca-13B + LoRA)

```bash
[tool.flwr.app.config]
task_type = "medical"
model_type = "chavinlo/alpaca-13b"
dataset_name = "medalpaca/medical_meadow_curated"
num_rounds = 10
batch_size = 4
max_length = 512

# Run
flwr run . local-simulation-gpu
```

### Nanochat Task (GPT Base Training)

```bash
[tool.flwr.app.config]
task_type = "nanochat"
model_type = "nanochat_d20"  # 560M params, or "nanochat_d32" (1.9B)
dataset_name = "karpathy/fineweb-edu-100b-shuffle"
num_rounds = 50
batch_size = 2
max_length = 512
gradient_accumulation_steps = 1
lr = 0.0006
weight_decay = 0.01

# Run
flwr run . local-simulation-gpu
```

**Vanilla Training (Non-FL):**
```bash
# Single GPU
python nanochat_train.py --num-steps 50 --batch-size 2 --max-length 512

# Multi-GPU DDP
torchrun --standalone --nproc_per_node=2 nanochat_train.py \
  --num-steps 200 --batch-size 2 --max-length 2048 \
  --wandb-project nanochat_vanilla
```

## Nanochat Architecture

### Model (d20 - 560M params)
- **Vocabulary**: 65,536 tokens (BPE, GPT-4 style)
- **Layers**: 20
- **Hidden size**: 1,280
- **Attention heads**: 10 (with RoPE)
- **Context**: 2,048 tokens
- **Features**: RMSNorm, ReLU² MLP, untied embeddings, BF16 training

### Model (d32 - 1.9B params)
- Same architecture, 32 layers, 2,048 hidden size, 16 heads

### Data Sharding Strategy

**Full dataset on all nodes** - Time-based rounds (5 min) naturally partition data across clients. Simple and efficient for 1-3 nodes.

### Evaluation

- **Vision**: Standard test split accuracy
- **LLM/Medical**: Holdout data (last 10% of training), perplexity
- **Nanochat**: Bits-per-byte (vocab-size independent metric)

## Deployment

### Local Simulation
```bash
flwr run . local-simulation-gpu --stream
```

### Production Deployment
```bash
# Start superlink
flower-superlink --insecure

# Start supernodes
CUDA_VISIBLE_DEVICES=0 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config "partition-id=0"

CUDA_VISIBLE_DEVICES=1 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9095 \
  --node-config "partition-id=1"
```

## Project Structure

```
pilot/
├── vision.py           # Vision models (EfficientNet, CNN)
├── llm.py              # LLM models (Qwen + LoRA)
├── medical.py          # Medical models (Alpaca + LoRA)
├── nanochat_fl.py      # Nanochat FL integration
├── nanochat/           # Nanochat subpackage
│   ├── gpt.py          # GPT model
│   ├── tokenizer.py    # BPE tokenizer
│   ├── dataloader.py   # Streaming loader
│   ├── dataset.py      # Dataset utilities
│   ├── common.py       # Utilities
│   ├── adamw.py        # AdamW optimizer
│   ├── muon.py         # Muon optimizer
│   ├── loss_eval.py    # Bits-per-byte eval
│   └── checkpoint_manager.py
├── data.py             # Shard management
├── client_app.py       # Client routing
└── server_app.py       # Server orchestration

rustbpe/                # Rust tokenizer extension
scripts/
└── tok_train.py        # Tokenizer training

nanochat_train.py       # Vanilla nanochat training
```

## References

- **Flower**: https://flower.ai/docs/
- **Nanochat**: https://github.com/karpathy/nanochat
- **FineWeb-EDU**: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
- **Medical Meadow**: https://github.com/kbressem/medAlpaca
