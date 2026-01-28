# FL Round-Trip Timing Analysis

## Overview

Analysis of the ~107s "network overhead" observed during federated learning rounds between clients and server running on the same physical machine.

## Measured Timings

| Metric | Value | Description |
|--------|-------|-------------|
| timing/round_trip | 645s | Total time for send_and_receive |
| timing/client_total | 536s | load + compile + train + serialize |
| timing/network_overhead | 107s | round_trip - client_total |
| timing/train | 525s | Training loop inside subprocess |
| timing/ddp_spawn_time | 580s | Wall-clock time for mp.spawn() |
| timing/dispatch_latency | 32s | Server send to client entry |
| timing/return_latency | 27s | Client exit to server receive |

## Key Finding: The Bug

The original calculation was:
```
unknown_overhead = network - (dispatch + return + ddp_spawn_time)
unknown_overhead = 107 - (32 + 27 + 580) = -532s  # WRONG!
```

The problem: `ddp_spawn_time` (580s) **includes** the training time (525s). We were double-counting.

## Corrected Calculation

```
ddp_overhead = ddp_spawn_time - train = 580 - 525 = 55s
unknown_overhead = network - (dispatch + return + ddp_overhead)
unknown_overhead = 107 - (32 + 27 + 55) = -7s  # ~0, as expected
```

## Real Breakdown of "Network Overhead"

The 107s is **not network overhead** (same physical machine = negligible network latency). It consists of:

| Component | Time | What it is |
|-----------|------|------------|
| dispatch_latency | ~32s | Flower protobuf serialization (server->client) |
| return_latency | ~27s | Flower protobuf serialization (client->server) |
| ddp_overhead | ~55s | mp.spawn() setup/teardown overhead |
| **Total** | ~114s | Accounts for the 107s (small variance expected) |

## Root Causes

### 1. Flower Serialization (~59s total)

Flower uses protobuf/gRPC to serialize the ~582MB model. Even on localhost, this serialization overhead is significant:
- Server encodes ArrayRecord to protobuf (~32s)
- Client decodes, then re-encodes response (~27s)

### 2. DDP Spawn Overhead (~55s)

`mp.spawn()` creates fresh processes each round:
- Process creation
- CUDA context initialization per GPU
- NCCL distributed group setup
- Process teardown after training

## Potential Optimizations

### Reduce Serialization Overhead
- Use shared memory for same-machine deployments instead of gRPC
- Investigate Flower's experimental shared memory transport
- Consider direct tensor sharing via `torch.multiprocessing.Queue`

### Reduce DDP Spawn Overhead
- Keep worker processes alive between rounds (process pool pattern)
- Use persistent CUDA contexts
- Pre-initialize DDP groups that persist across rounds

## Timing Instrumentation

The following metrics are now logged to wandb:

- `timing/round_trip` - Total send_and_receive time
- `timing/client_total` - Sum of client-side work
- `timing/network_overhead` - round_trip - client_total
- `timing/dispatch_latency` - Server to client transfer
- `timing/return_latency` - Client to server transfer
- `timing/ddp_spawn_time` - Full mp.spawn() wall-clock time
- `timing/ddp_overhead` - ddp_spawn_time - train (pure overhead)
- `timing/unknown_overhead` - Any remaining unaccounted time
