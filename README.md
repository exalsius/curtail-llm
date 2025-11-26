# Pilot ✈

## Installation

```
# uv venv
# source .venv/bin/activate
# uv sync --dev --extra test

python -m venv venv
source venv/bin/activate
pip install -e ".[dev,test]"
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0 --upgrade --force-reinstall
```


### Cluster Setup

```
apt update && apt install git -y
git config --global user.name "Philipp Wiesner"
git config --global user.email "philipp.wiesner@logsight.ai"
ssh-keygen -t ed25519 -C "philipp.wiesner@logsight.ai" -N "" -f ~/.ssh/id_ed25519
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

git clone git@github.com:exalsius-dsv-collab/pilot.git
git clone git@github.com:exalsius/cold-start-hackathon.git
```


## Run Flower simulation

### Single Run
```bash
# Run with default configuration
flwr run . local-simulation --stream

# Run with custom parameters
flwr run . --run-config "num_rounds=50 lr=0.05 batch_size=64" --stream

# Run with GPU support
flwr run . local-simulation-gpu --stream
```

## Deployment
```
flower-superlink --insecure

CUDA_VISIBLE_DEVICES=0 flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9094 --node-config "partition-id=0"

CUDA_VISIBLE_DEVICES=1 flower-supernode --insecure --superlink 127.0.0.1:9092 \
    --clientappio-api-address 127.0.0.1:9095 --node-config "partition-id=1"
```

### Parameter Sweeps

Run multiple experiments with different numbers of supernodes:

```bash
python run_sweep.py
```

The script uses Flower's programmatic API to run simulations with different `num-supernodes` values.
Customize parameters at the top of the script:

```python
NUM_SUPERNODES = [2, 3, 5, 10]  # Supernodes to test
RUN_CONFIG = {
    "lr": 0.1,
    "batch_size": 32,
    "num_rounds": 100,
    # ... other parameters
}
```


## medAlpaca

```
cd medAlpaca-main
python medalpaca/train.py --model chavinlo/alpaca-13b --output_dir 'output' --wandb_run_name "chavinlo/alpaca-13b"
```