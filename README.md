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
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/
python -m nanochat.dataset -n 240  # Download dataset for tokenizer training (~22GB of FineWeb-EDU data)
python -m scripts.tok_train --max_chars=4000000000 --vocab_size=65536  # Train BPE tokenizer on ~4B characters, takes about 3min
python -c "from nanochat.tokenizer import get_tokenizer; print(f'Vocab size: {get_tokenizer().get_vocab_size()}')"
```

### Redis Setup (head node only)

The system uses Redis for coordination between server and clients. 
Redis must be accessible by all nodes, run via:

```bash
apt-get update
apt-get install redis-server
redis-server --daemonize yes
redis-cli ping
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
tmux new -A -s superlink

cd /workspace/pilot
source .venv/bin/activate
export WANDB_API_KEY=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

flower-superlink --insecure
```

3. Start SuperNodes (workers) on each GPU:
```bash
# Client 0
tmux new -A -s client_0

cd /workspace/pilot
source .venv/bin/activate
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat/

CUDA_VISIBLE_DEVICES=0,1,2,3 flower-supernode --insecure \
  --superlink 127.0.0.1:9092 \
  --clientappio-api-address 127.0.0.1:9094 \
  --node-config 'name="client_0" partition-id=0' 


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

You can override config values from command line:

```bash
flwr run . local-deployment --run-config "lr=0.0005" --stream
```

### Vanilla nanochat Training

Baseline:

```bash
# Single GPU
python nanochat_train.py --num-steps 50 --batch-size 2 --max-length 512

# Multi-GPU with DDP
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m scripts.base_train -- --depth 20 --target-param-data-ratio 20 \
  --batch-size 8 --wandb-project pilot_flwr --run baseline
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



[ROUND 2]
INFO :      Round monitor started for ROUND 2
INFO :      Scaling weight decay from 0.200000 to 0.072000 for depth 20
INFO :      Round 2 schedule: Step 0, LRM 1.0000, Momentum 0.8500
INFO :      Querying all 2 connected Flower nodes for their names...
INFO :       - Flower node 15551475861331552659: 'client_1'
INFO :       - Flower node 16296237454998711291: 'client_0'
INFO :      configure_train: Training on all 2 Flower nodes
INFO :      Shard progress: 0.0% (0/240 complete)
ERROR :     ServerApp raised an exception
Traceback (most recent call last):
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/server/serverapp/app.py", line 239, in run_serverapp
    updated_context = run_(
                      ^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/server/run_serverapp.py", line 62, in run
    server_app(grid=grid, context=context)
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/server/server_app.py", line 176, in __call__
    self._main(grid, context)
  File "/home/dev/.flwr/apps/exalsius.pilot.0.1.0.e7f48d5d/pilot/server_app.py", line 97, in main
    strategy.start(
  File "/home/dev/.flwr/apps/exalsius.pilot.0.1.0.e7f48d5d/pilot/strategy.py", line 454, in start
    messages = self.configure_train(current_round, arrays, train_config, grid)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dev/.flwr/apps/exalsius.pilot.0.1.0.e7f48d5d/pilot/strategy.py", line 282, in configure_train
    record = RecordDict({"arrays": arrays, "config": config})
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/recorddict.py", line 194, in __init__
    self[key] = record
    ~~~~^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/recorddict.py", line 233, in __setitem__
    super().__setitem__(key, value)
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/typeddict.py", line 41, in __setitem__
    cast(Callable[[V], None], self.__dict__["_check_value_fn"])(value)
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/recorddict.py", line 63, in _check_value
    raise TypeError(
TypeError: Expected `ArrayRecord`, `MetricRecord`, or `ConfigRecord` but received `NoneType` for the value.
ERROR :     Exit Code: 201

An unhandled exception occurred in the ServerApp.

For more information, visit: <https://flower.ai/docs/framework/v1.23.0/en/ref-exit-codes/201.html>
/workspace/pilot/.venv/lib/python3.12/site-packages/google/protobuf/internal/well_known_types.py:174: DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects to represent datetimes in UTC: datetime.datetime.now(datetime.UTC).
  self.FromDatetime(datetime.datetime.utcnow())



INFO :      Rank 2: Stop signal received (END ROUND 1) after batch 511
INFO :      Rank 1: Stop signal received (END ROUND 1) after batch 511
INFO :      Rank 0: Stop signal received (END ROUND 1) after batch 511
Training: 510it [03:53,  2.19it/s, loss=5.34, shard=0, shard_progress=64.7%]
INFO :      Client 0: 511 batches
ERROR :     ClientApp raised an exception
Traceback (most recent call last):
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/supernode/runtime/run_clientapp.py", line 117, in run_clientapp
    reply_message = client_app(message=message, context=context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/clientapp/client_app.py", line 161, in __call__
    return self._registered_funcs[full_name](message, context)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/dev/.flwr/apps/exalsius.pilot.0.1.0.14486aac/pilot/client_app.py", line 321, in train
    mp.spawn(
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 364, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 320, in start_processes
    while not context.join():
              ^^^^^^^^^^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 220, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/torch/multiprocessing/spawn.py", line 95, in _wrap
    fn(i, *args)
  File "/home/dev/.flwr/apps/exalsius.pilot.0.1.0.14486aac/pilot/client_app.py", line 291, in run_training_process
    "metrics": MetricRecord(
               ^^^^^^^^^^^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/metricrecord.py", line 135, in __init__
    self[k] = metric_dict[k]
    ~~~~^^^
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/typeddict.py", line 41, in __setitem__
    cast(Callable[[V], None], self.__dict__["_check_value_fn"])(value)
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/metricrecord.py", line 68, in _check_value
    is_valid(value)
  File "/workspace/pilot/.venv/lib/python3.12/site-packages/flwr/common/record/metricrecord.py", line 46, in is_valid
    raise TypeError(
TypeError: Not all values are of valid type. Expected `typing.Union[int, float, list[int], list[float]]` but `<class 'str'>` was passed.






Bug: clients don't stop of signal gets sent before they are active. This case thei train forever