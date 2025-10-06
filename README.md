# Pilot ✈

## Installation

```
uv venv
source .venv/bin/activate
uv sync --dev --extra test
```


### Cluster Setup

```
apt update && apt install git -y
git config --global user.name "Philipp Wiesner"
git config --global user.email "philipp.wiesner@logsight.ai"
ssh-keygen -t ed25519 -C "philipp.wiesner@logsight.ai"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:exalsius/pilot.git
```


## medAlpaca

```
python medAlpaca-main/medalpaca/train.py --model chavinlo/alpaca-native --output_dir 'output'
```