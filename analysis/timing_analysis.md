# Breakdown of "Network Overhead"

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
