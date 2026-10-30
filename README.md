---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: vllm
tags:
- serving-recipe
- vllm
- dgx-spark
- tensor-parallel
- long-context
---

# DeepSeek-V4.1-Flash at TP=3 across 3× NVIDIA DGX Spark — capacity-first

**TL;DR** — Serve DeepSeek-V4.1-Flash (552B total, native FP4+FP8 MoE
with a 196B engram conditional-memory block) tensor-parallel across three
DGX Sparks on a 200G CX-7 RoCE triangle, using the eugr b12x vLLM fork
on a public Docker Hub image. Capacity-optimized: a **1.6M-token KV
pool** at **512K context** with **6.2× concurrency** at 262K — the
largest published TP3 KV pool for this model. ~29 tok/s single-stream
(prose, no spec decode). Engram n-gram tables are served from NVMe,
freeing 31 GiB/rank of host memory that would otherwise be consumed.
The bring-up maps every memory trap on GB10 unified memory: NCCL buffer
pinning, engram host staging, loader bounce buffers, and the startup
free-memory check — each with its measured fix.

**The only published vLLM TP3 recipe for V4.1-Flash using a publicly
available image.** Other V4.1-on-Spark recipes either use SGLang (a
different serving engine), require building a custom vLLM fork from
source, or target TP4 with a 4th box. This one is four commands with
a pre-built image.

As far as we can find: this is the first TP=3 vLLM serving of
V4.1-Flash on the eugr b12x lineage, and the first recipe of any engine
to prioritize KV capacity and context length over throughput on this
hardware (the trade is documented below). The speed-optimized SGLang
recipe from the community achieves higher tok/s but at half the KV
capacity and half the context — see **Comparison** below.

Prerequisites: three DGX Sparks wired as a full CX-7 RoCE triangle (one
/24 per leg, MTU 9000) plus a management LAN, the ~510 GB checkpoint
and the digest-pinned image staged on every box, and Docker with GPU
support.

## Quick start

After staging the checkpoint and image on all three boxes:

```sh
./launch/ds41-flash-tp3-up preflight
./launch/ds41-flash-tp3-up stage-env
./launch/ds41-flash-tp3-up start --yes    # workers→head; :8600 in ~10–12 min
```

## Served numbers

Measured at the shipped configuration (gmu 0.85, 4 GiB KV, 512K context,
no spec decode, engram on NVMe):

| Metric | Value |
|---|---|
| Single-stream decode (prose) | ~29 tok/s |
| TTFT (short prompt) | 1–2 s |
| Weights per rank | 95.4 GiB |
| KV pool | **1,617,645 tokens** |
| Context length | **524,288 (512K)** |
| Concurrency at 262K ctx | **6.2×** |
| Concurrency at 512K ctx | **~3×** |
| Cold boot (load→health) | ~12 min (200 s load) |
| Temp-0 determinism | identical across runs |
| Tool calling (deepseek_v41) | working |
| Reasoning parser (deepseek_v41) | working |

## Why capacity over speed

The community SGLang recipe for this model on the same hardware
achieves ~51 tok/s but with a 750K KV pool and 262K context. This
recipe achieves 1.6M KV tokens at 512K context — **more than double the
capacity** — at ~29 tok/s.

For workloads that are context-bound (RAG with large knowledge bases,
long legal or research documents, multi-turn conversations with massive
history), the KV pool and context length are the binding constraints,
not generation speed. A 237K-token prompt that overflows a 262K context
costs you the entire request; a 22 tok/s difference in generation speed
is invisible when streaming at reading speed.

The trade-off table:

| | This recipe (vLLM) | Community SGLang |
|---|---|---|
| KV pool | **1.6M tokens** | 750K tokens |
| Context | **512K** | 262K |
| Concurrency @ 262K | **6.2×** | ~2.9× |
| Speed | 29 tok/s | **51 tok/s** |
| Spec decode | off (capacity) | on (speed) |
| Engine | vLLM (eugr b12x) | SGLang (custom) |

## The bring-up: six walls and their fixes

Getting V4.1-Flash to serve on 3× DGX Spark with vLLM required
navigating six distinct failure modes. Each is documented here so you
don't have to rediscover them:

### 1. NCCL connection buffers pin 4.7 GiB of host RAM

The default NCCL buffer sizing on a 3-node RoCE ring reserves ~4.7 GiB
of **pinned (unreclaimable) host memory per node** for connection
buffers. On GB10 unified memory this silently consumes KV capacity. Fix:

```sh
NCCL_BUFFSIZE=1048576        # 1 MiB (default 4 MiB)
NCCL_LL128_BUFFSIZE=262144   # 256 KiB
NCCL_PROTO="^LL128"
NCCL_MAX_NCHANNELS=8
```

This cut the pinned footprint from 4.7 GiB to 139 MiB per node —
freeing enough memory for the head node to boot.

### 2. Engram tables stage 31 GiB per rank in host memory

V4.1-Flash ships with a 196B-parameter engram (n-gram conditional
memory) block. Without explicit configuration, the loader stages the
engram tables into host memory — consuming ~31 GiB per rank on a 121 GiB
box that already holds 95 GiB of weights. Fix:

```sh
--engram-config '{"cpu_offload":false,"table_memory":"disk"}'
```

The tables stay on NVMe; rows are read on demand. This is the single
setting that makes V4.1 bootable at TP3 on this hardware.

### 3. The b12x loader's bounce buffers OOM on busy head nodes

The b12x O_DIRECT loader uses pinned bounce buffers that add ~10–16 GiB
of unreclaimable host memory during weight loading. On a head node that
also runs the API server, relay, and system daemons, this tips the box
into kernel OOM during load. Fix:

```sh
--load-format safetensors    # instead of b12x
```

Plain safetensors streams through kernel-evictable page cache instead
of pinned bounce buffers. The image's value is in its serving kernels
and memory management, not its loader — the loader is swappable.

### 4. The startup free-memory check

The fork checks available GPU memory at startup against the requested
utilization fraction. On a busy head node with even 2 GiB of extra
daemons, the default 0.92 target fails. Fix: set gmu to 0.85 and run
`echo 3 > /proc/sys/vm/drop_caches` before boot.

### 5. Vocabulary divisibility at TP3

The fork divides the raw vocabulary size (129,280) by the TP world size
at load. 129,280 % 3 ≠ 0. The fork's runtime TP3 padding handles
attention heads, o_groups, and drafter experts — but the vocabulary
check requires the `--served-model-name` to be set (the alias mechanism
triggers a different code path). Using a single served name (e.g.
`main`) avoids the issue.

### 6. io_uring requires seccomp=unconfined in Docker

The b12x image's I/O infrastructure uses io_uring, which Docker's
default seccomp profile blocks. Fix:

```sh
--security-opt seccomp=unconfined
```

## Stack

| Component | Pin |
|---|---|
| Image | `eugr/spark-vllm-b12x:nightly-20261015` (digest `sha256:57c24fc397c6…`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (~510 GB) |
| vLLM | v0.1.dev21830+g8244d1c44 (fork build) |
| MoE backend | DEEPGEMM_MXFP4 |
| Dense GEMM | FlashInferCutlassMxfp8LinearKernel |
| Model runner | V2 |
| KV layout | BLHNC |
| CUDA graphs | FULL_AND_PIECEWISE |

## Configuration

The complete launch environment (`config/cluster.env`):

```sh
# Memory
GMU=0.85
CTX=524288
KV_CACHE_MEMORY_BYTES=4g

# Serving
PORT=8600
TP=3
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192
BLOCK_SIZE=256

# Loader
LOAD_FORMAT=safetensors
ENGRAM_CONFIG='{"cpu_offload":false,"table_memory":"disk"}'

# NCCL (the critical caps)
NCCL_BUFFSIZE=1048576
NCCL_LL128_BUFFSIZE=262144
NCCL_PROTO="^LL128"
NCCL_MAX_NCHANNELS=8

# Allocator
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# Parsers
TOOL_CALL_PARSER=deepseek_v41
REASONING_PARSER=deepseek_v41

# Model name (single alias; see wall #5)
SERVED_MODEL_NAME=main
```

## Hardware

Three NVIDIA DGX Spark (GB10, 121.69 GiB unified CPU/GPU memory each),
wired as a full CX-7 RoCE triangle (one /24 subnet per leg, 200G per
link, MTU 9000) plus a management LAN for SSH and API access.

Per-node memory budget at the shipped configuration:

| Component | Size |
|---|---|
| Model weights | 95.4 GiB |
| KV cache (4 GiB reserved) | 3.7 GiB |
| NCCL buffers (capped) | 0.14 GiB |
| CUDA context + framework | ~4 GiB |
| Host OS + daemons | ~8 GiB |
| **Free (serving)** | **~7 GiB** |

## Engram on NVMe

The two 101 GiB engram n-gram table shards (~203 GiB total) remain on
each node's local NVMe. They are referenced by the model's index but
are never loaded into host or GPU memory — the `table_memory:disk`
setting instructs the engine to read rows on demand during inference.
This is the same approach described in the community SGLang recipe and
the openzaka TP4 deployment guide.

## Speculative decode: tested and rejected

DSpark speculative decoding (k=5) was tested and rejected on capacity
grounds. The draft model adds ~3 GiB per rank, which at TP3 leaves
insufficient memory for the KV pool and CUDA graphs — the boot OOMs at
every gmu and KV combination tried (0.83/0.85 × 1g/2g). The trade:
~51 tok/s (community SGLang number with spec decode) vs 29 tok/s, at
the cost of halving the KV pool and context. For context-bound
workloads, the capacity is worth more than the speed.

## Comparison with other V4.1-Flash recipes

| Recipe | Engine | TP | Image | KV pool | Context |
|---|---|---|---|---|---|
| **This one** | vLLM (eugr b12x) | **3** | **public Docker Hub** | **1.6M** | **512K** |
| MiaAI-Lab | SGLang | 3 | custom build | 750K | 262K |
| christopherowen/spark-ds41f | vLLM (custom fork) | 3 | build from source | ~800K | 512K |
| tonyd2wild | vLLM (patched nightly) | 2/4 | patched | varies | varies |
| openzeka | vLLM | 4 | custom | — | 300K |

## Credits

- The eugr/spark-vllm-docker project for the b12x images and the
  GB10-optimized kernels
- The MiaAI-Lab recipe for the SGLang comparison numbers and the NCCL
  buffer tuning discovery (their documentation of the 4.7 GiB → 139 MiB
  fix was the key that unlocked the head node)
- The MiniMax-M3 TP3 community thread for the `--load-format
  safetensors` OOM fix
- The openzaka TP4 deployment whitepaper for the engram-on-disk pattern

## License

MIT
