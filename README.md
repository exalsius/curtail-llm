# Pilot ✈

## Installation

```
# uv venv
# source .venv/bin/activate
# uv sync --dev --extra test

python -m venv venv
source venv/bin/activate
pip install -e ".[dev,test]"
```


### Cluster Setup

```
apt update && apt install git -y
git config --global user.name "Philipp Wiesner"
git config --global user.email "philipp.wiesner@logsight.ai"
ssh-keygen -t ed25519 -C "philipp.wiesner@logsight.ai" -N ""
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:exalsius/pilot.git
```


## Run Flower simulation

```
flwr run . local-simulation --stream
```


## medAlpaca

```
cd medAlpaca-main
python medalpaca/train.py --model chavinlo/alpaca-13b --output_dir 'output' --wandb_run_name "chavinlo/alpaca-13b"
```