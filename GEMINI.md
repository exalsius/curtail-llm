# Exalsius Pilot - Federated Nanochat Pretraining

This project implements a Federated Learning (FL) system for pretraining GPT-style Large Language Models ("nanochat") on volatile clients using the [Flower](https://flower.dev/) framework.

## Project Overview

*   **Goal:** Pretrain LLMs in a federated manner, handling client volatility and resource constraints.
*   **Architecture:**
    *   **Federated Learning:** Uses Flower (`flwr`) Next-Gen architecture (ClientApp/ServerApp).
    *   **Model:** "Nanochat" (a GPT-style transformer), inspired by Karpathy's `nanogpt`.
    *   **Tokenizer:** Custom Byte-Pair Encoding (BPE) implemented in **Rust** (`rustbpe`) for performance, exposed to Python via `pyo3` and `maturin`.
    *   **Coordination:** Redis is used for state management and coordination in distributed deployments.
    *   **Monitoring:** Weights & Biases (WandB) for experiment tracking.

## Directory Structure

*   `pilot/`: Contains the Flower application logic.
    *   `client_app.py`: The `ClientApp` definition (local training logic).
    *   `server_app.py`: The `ServerApp` definition (federated strategy and aggregation).
    *   `model.py`: Model definition/wrapping for FL.
    *   `strategy.py`: Custom FL strategies.
*   `nanochat/`: The core LLM implementation.
    *   `gpt.py`: Transformer architecture definition.
    *   `engine.py`: Training engine.
    *   `tokenizer.py`: Python wrapper for the Rust tokenizer.
*   `rustbpe/`: Rust source code for the high-performance BPE tokenizer.
*   `scripts/`: Standalone scripts for training and evaluation.
    *   `base_train.py`: Standalone (centralized) training script (comparable to `nanochat_train.py` in older docs).
    *   `tok_train.py`: Script to train the BPE tokenizer on a dataset.
    *   `base_eval.py`: Evaluation script.
*   `energy_simulation.py`: Likely used for simulating energy/volatility constraints.

## Setup & Dependencies

The project uses `uv` for Python dependency management and `cargo` (Rust) for the tokenizer.

1.  **Python Environment:**
    ```bash
    uv venv
    uv sync
    source .venv/bin/activate
    ```

2.  **Rust Tokenizer:**
    *   Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
    *   Build extension:
        ```bash
        uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
        ```

3.  **Redis:** Required for distributed runs (Head node).
    ```bash
    redis-server --daemonize yes
    ```

## Development & Usage

### 1. Data Preparation
Before training, you need to download data and train the tokenizer:
```bash
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/ # or your preferred path
python -m nanochat.dataset -n 240
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536
```

### 2. Federated Simulation (Local)
Run a simulation on your local machine (using Flower Simulation):

*   **GPU:**
    ```bash
    flwr run . local-simulation-gpu --stream
    ```
*   **CPU:**
    ```bash
    flwr run . local-simulation --stream
    ```

### 3. Federated Deployment (Distributed)
Requires starting `redis-server`, `flower-superlink`, and `flower-supernode` processes manually (see `README.md` for details), then running:
```bash
flwr run . local-deployment --stream
```

### 4. Standalone Training
For baseline comparison or debugging model code without FL overhead:
```bash
# Single GPU/CPU
python scripts/base_train.py --depth=4 --max_seq_len=512 --num_iterations=20

# Distributed (DDP)
torchrun --nproc_per_node=2 scripts/base_train.py ...
```

## Configuration

*   **Project Config:** `pyproject.toml` contains `[tool.flwr.app.config]` which defines default hyperparameters (learning rate, batch size, model type) and FL settings.
*   **Overrides:** You can override config values via the CLI when running `flwr`:
    ```bash
    flwr run . local-simulation-gpu --run-config "lr=0.0001 batch_size=4"
    ```

## Conventions

*   **Code Style:** Standard Python conventions.
*   **Logging:** WandB is heavily used. Ensure `WANDB_API_KEY` is set if tracking is enabled.
*   **Dependencies:** Always use `uv add` or `uv sync` to manage python packages.
