# Pilot 🧑‍✈️

## Installation

```
uv venv
source .venv/bin/activate
uv sync --dev --extra test
```


### Remote

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
python medAlpaca-main/medalpaca/train.py --model chavinlo/alpaca-native --data_path medical_meadow_small.json --output_dir 'output' --train_in_8bit False \ --use_lora True --bf16 True --tf32 False --fp16 False --global_batch_size 128 --per_device_batch_size 8
```