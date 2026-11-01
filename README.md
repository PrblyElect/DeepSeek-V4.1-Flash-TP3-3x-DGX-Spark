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
on a public Docker Hub image. Capacity-optimized: a **~2M-token KV
pool** at **512K context** — the pool is boot-labile, landing between
roughly 1.6M and 2.4M tokens depending on the boot — for **~6–9×
concurrency** at 262K and **~3–4.7×** at 512K. ~29 tok/s single-stream
(prose, no spec decode). Engram n-gram tables are served from NVMe,
freeing 31 GiB/rank of host memory that would otherwise be consumed.
The bring-up maps every memory trap on GB10 unified memory: NCCL buffer
pinning, engram host staging, loader bounce buffers, and the startup
free-memory check — each with its measured fix.

**A published vLLM TP3 recipe for V4.1-Flash on a publicly available
image.** This one is three commands with a pre-built image. The field moves
fast — check the community indexes before repeating the "only" claim.

**This is not the only three-node TP=3 recipe on the public image.** The
project that publishes the image now ships its own
`recipes/3x-spark-cluster/deepseek-v4.1-flash.yaml` — the same model, the
same image lineage, the same three-node shape. It is a recipe for their
cluster runner, with no acceptance gates of its own, and it asks for a
smaller KV pool and fewer concurrent sequences than this one, and a
context derived from that pool. The difference between the two is the trade this recipe makes;
the side-by-side is in **Upstream's own 3-node recipe** below, and where
they differ, that table is the reason this recipe reads the way it does.

The scope here is the public image, not the B12X kernel lineage: a
community TP=3 vLLM fork built from source reaches a larger pool and a
larger context still, and its figures are in the table below. The
speed-optimized SGLang recipe from the community achieves higher tok/s
but at about a third of the KV capacity and half the context — see
**Comparison** below.

Prerequisites: three DGX Sparks wired as a full CX-7 RoCE triangle (one
/24 per leg, MTU 9000) plus a management LAN, the ~510 GB checkpoint
and the digest-pinned image staged on every box, and Docker with GPU
support.

## Quick start

Edit `config/cluster.env` first: set the three box names, the three
management IPs, the checkpoint path (`MODEL_DIR` — it must be the path
**on the boxes**, and the shipped default is a placeholder), and the two
fabric selectors (`NCCL_IB_HCA` — the RoCE port pair, from `ibdev2netdev`
on each box — and `NCCL_SOCKET_IFNAME`, the management interface). The
preflight refuses to run while the fabric selectors still hold their
placeholder.

Then stage the checkpoint and the digest-pinned image on all three
boxes, and:

```sh
./launch/ds41-flash-tp3-up preflight    # image digest + content, checkpoint, fabric
./launch/ds41-flash-tp3-up stage-env     # copies cluster.env + launcher + tp3-kit
./launch/ds41-flash-tp3-up start --yes   # head first, then the workers; :8600 in ~10–12 min
```

`preflight` is read-only: it inspects the image that is already on the
box and never pulls. It checks three things per box — the image's
manifest digest matches `IMAGE_DIGEST`, the sha256 of the image's
RootFS layer list matches `IMAGE_CONTENT_SHA256`, and the checkpoint
directory is present — then reports the box's available memory.

`stage-env` copies `config/cluster.env`, `launch/start-tp3-fork.sh` and
the `launch/tp3-kit/` gate into `~/ds41-flash-tp3/` on every box. The
kit is the in-container engine-identity gate: the image is patched at
runtime by the bootstrap, so the manifest digest alone cannot prove what
executes. The gate asserts the engine's build stamp and the hashes of
the engine-critical files inside the container before the engine starts.
A newer upstream nightly will fail that gate until it is re-audited —
that is the intended adoption gate, not a bug.

Two files in this recipe are the load-bearing part and are worth
reading before you run anything: `launch/start-tp3-fork.sh` (what
actually boots a box) and `launch/tp3-kit/engine-identity.json` (what
the boot is allowed to execute).

### Reading the container command without starting it

`PRINT_CMD=1` prints the `docker run` command the boot script would run
and exits. No container is started, no GPU is touched, and the recipe's
own pre-boot assertions still run first — so what you read is a command
that would actually have been accepted.

Run it on the box you are about to start, in the staged directory, with
the profile exported first — the launcher reads its values from the
environment, and `config/cluster.env` is where they live:

```sh
cd ~/ds41-flash-tp3
set -a; . config/cluster.env; set +a
NODE_IP=192.168.0.10 NODE_RANK=0 ROLE=head PRINT_CMD=1 bash start-tp3-fork.sh
```

(`NODE_IP` is this box's own address, `NODE_RANK` its rank, `ROLE` its
role — the same values `ds41-flash-tp3-up start` sets. The placeholders
in `config/cluster.env` must already be filled in: the pre-boot
assertions reject an unset fabric, an unstaged model and an unpinned
image, which is why the printed command is one that would really have
been accepted.)

It prints the command one argument per line, and then again as a single
copy-pasteable line.

This is how you check that every value in `config/cluster.env` reached
the container. A knob that is read but never forwarded shows up here as
the engine's default where you expected your value — which is exactly
the defect class this recipe has already been bitten by.

## Served numbers

Measured at the shipped configuration (gmu 0.85, 4 GiB KV, 512K context,
no spec decode, engram on NVMe):

| Metric | Value |
|---|---|
| Single-stream decode (prose) | ~29 tok/s |
| TTFT (short prompt) | 1–2 s |
| Weights per rank | 95.4 GiB |
| KV pool | **~2M tokens** (boot-labile: ~1.6–2.4M) |
| Context length | **524,288 (512K)** |
| Concurrency at 262K ctx | **~6–9×** |
| Concurrency at 512K ctx | **~3–4.7×** |
| Cold boot (load→health) | ~12 min (200 s load) |
| Temp-0 determinism | identical across runs — short prompts (gate 2). The long-prompt path is a separate gate (9) and is not covered by gate 2. |
| Tool calling (deepseek_v41) | working |
| Reasoning parser (deepseek_v41) | working |

(All gate numbers refer to `docs/VALIDATION.md`, which ships with the recipe.)

### Where the boot time goes

The cold boot above is ~12 minutes, and the weight load is ~200 s of it.
The balance is the **b12x GEMM autotune** — estimated from those two
figures, not separately timed: the engine compiles a plan of
several thousand kernels for the shapes this model and this batch width
produce, and the boot log shows each shape starting from
`0 cached, 0 compilations`.

The compiled artifacts are cached under `/root/.cache/b12x` — **inside the
container's writable layer**, which is discarded when the container is
recreated. The cache therefore never survives a boot, and every boot pays
the full compile.

Two knobs in the pinned build can change that, and this recipe sets
neither: `VLLM_USE_AOT_COMPILE` and `VLLM_USE_MEGA_AOT_ARTIFACT` (both
default off; upstream's own vision recipe sets both). No AOT artifact is
baked into the image, so enabling them changes *what is compiled*, not
*what is reused* — persistence still has to come from a mount that
outlives the container.

What this recipe does **not** do is raise the compile parallelism. More
compile workers is the tempting answer and the wrong one: a wider batch
width has already been observed to kill the worker ranks during the
compile phase, and the compile-time memory spike on unified memory is
what makes that happen. A cache hit reduces both the boot time and that
risk; more workers increase the second.

Enabling any of this needs a measurement across two boots, and it needs
the cache keyed by the image build stamp — a stale artifact is a *wrong
kernel*, which is a worse failure than a slow boot. That is an operator
experiment on a lane, not a desk change, so the recipe documents the
knobs and leaves the default alone.

## Why capacity over speed

The community SGLang recipe for this model on the same hardware
achieves ~51 tok/s but with a 750K KV pool and 262K context. This
recipe achieves roughly 2M KV tokens at 512K context — **more than double
the capacity** — at ~29 tok/s.

For workloads that are context-bound (RAG with large knowledge bases,
long legal or research documents, multi-turn conversations with massive
history), the KV pool and context length are the binding constraints,
not generation speed. A 300K-token document is rejected outright at a
262K context; a 22 tok/s difference in generation speed is invisible when
streaming at reading speed.

The trade-off table:

| | This recipe (vLLM) | Community SGLang |
|---|---|---|
| KV pool | **~2M tokens** | 750K tokens |
| Context | **512K** | 262K (their "256k limit") |
| Concurrency @ 262K | **~6–9×** | ~2.9× (derived) |
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

### 2. Engram tables stage ~31 GiB per rank in host memory

V4.1-Flash ships with a 196B-parameter engram (n-gram conditional
memory) block: two ~94.5 GiB table shards, ~189 GiB in all, of which
each rank reads its ~63 GiB share from its node's local NVMe. Without
explicit configuration, the loader stages a large part of each rank's
rows into host memory — ~31 GiB per rank — on a 121 GiB box that already
holds 95 GiB of weights. Fix:

```sh
--engram-config '{"cpu_offload":false,"table_memory":"disk"}'
```

The tables stay on NVMe; rows are read on demand. This is the single
setting that makes V4.1 bootable at TP3 on this hardware.

### 3. The b12x loader's bounce buffers OOM on busy head nodes

The b12x O_DIRECT loader uses pinned bounce buffers that add ~10–16 GiB
of unreclaimable host memory during weight loading. On a head node that
also runs the API server and the usual system daemons, this tips the box
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
| Image | `eugr/spark-vllm-b12x:nightly-20261015` (digest `sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (~510 GB) |
| vLLM | v0.1.dev21830+g8244d1c44 (fork build) |
| MoE backend | DEEPGEMM_MXFP4 |
| Dense GEMM | FlashInferCutlassMxfp8LinearKernel |
| Model runner | V2 |
| KV layout | BLHNC |
| CUDA graphs | FULL_AND_PIECEWISE |

## Configuration

The launch environment (`config/cluster.env`). The bring-up tool
(`launch/ds41-flash-tp3-up`) reads the
box names and IPs, the checkpoint path (`MODEL_DIR` — it must exist on
every box), the image pins, the container name, the Docker security
options, and the serving knobs below; the remaining lines are plumbing
the launcher consumes — keep the two in step:

```sh
# Cluster (EDIT THESE — the bring-up tool reads them from here)
SPARK_HEAD="spark-head.local"      # head / rank 0
SPARK_A="spark-a.local"             # worker / rank 1
SPARK_B="spark-b.local"             # worker / rank 2
HEAD_IP="192.168.0.10"              # head's management LAN IP
WORKER_A_IP="192.168.0.11"
WORKER_B_IP="192.168.0.12"

# Image pin (the digest is what the preflight asserts)
IMAGE="eugr/spark-vllm-b12x:nightly-20261015"
IMAGE_DIGEST="sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"
IMAGE_CONTENT_SHA256="41e4f9a3005bca761eb883c6565c1a3667cfbb014d49ec3d97735c7cfe146616"

# Fabric (EDIT THESE — site-specific)
NCCL_IB_HCA="EDIT_ME_ROCE_PORT_PAIR"
NCCL_SOCKET_IFNAME="EDIT_ME_MGMT_IF"

# Memory
GMU=0.85
CTX=524288
KV_CACHE_MEMORY_BYTES=4g

# Serving
PORT=8600
TP=3
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192       # do not raise without a memory plan: 16384
                                  # OOM-killed both worker ranks during the b12x
                                  # compile phase (see FIXLOG failure mode 9)
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

Concurrency here means how many full-length requests fit in the KV pool.
It is a capacity figure, not a promise about simultaneous streams: the
shipped `--max-num-seqs 8` caps how many requests are served at once, so
the higher end of the 262K band describes pool depth, not concurrency.

### `BLOCK_SIZE=256` — what it is, and what `/metrics` will say

`BLOCK_SIZE=256` is not a free parameter: it is the block size the
DeepSeek-V4.1 sparse-SWA attention backend reports as its **preferred**
size on this hardware (`get_preferred_block_size() -> 256`; the engine's
own default is 16). Setting it explicitly pins the value against a future
backend that might prefer something else.

The value the engine *reports* is a different number, and the difference
is not a mistake:

- the engine logs `kv lcm block sizes 256` — the scheduler block size is
  the LCM of the KV groups' block sizes;
- `/metrics` reports `cache_config_info.block_size = 128`, because that
  field is the **minimum** across the prefix-cacheable KV groups, and
  DeepSeek-V4.1's sliding-window group is 128.

So `128` in the metrics is the sliding-window group's size, not the
requested size, and not an indication that the knob was ignored. The
acceptance gate checks all three numbers — the requested value (read from
the container's own environment, where the launcher forwards it), the LCM
the engine logs, and the metric label (see gate 8's kit).

The requested value cannot be read from the container's command line
alone, and the reason is worth stating: the serve line is a shell string,
so the command line holds the literal text `--block-size
"${BLOCK_SIZE:-256}"`, and bash expands it *inside* the container when the
bootstrap runs. That text is identical for every value the operator could
set, so matching it would assert nothing. The gate reads
`BLOCK_SIZE` from the container's environment instead — the launcher
forwards it with `-e BLOCK_SIZE="$BLOCK_SIZE"` — and falls back to the
command line only when it carries a *literal* number, which is what the
value is when a launcher bakes it in.

### What the engine actually resolves

The command line is not the behaviour. The MoE backend, the dense-linear
kernel and the attention backend are all chosen by the engine's own
policy, and a newer image can change them without changing a flag you
pass. On this configuration the engine resolves:

| | resolved to |
|---|---|
| MoE backend | `DEEPGEMM_MXFP4` (the MXFP4 experts) |
| dense linear | `FlashInferCutlassMxfp8LinearKernel` |
| attention | b12x (DSA indexer) |
| TP=3 head padding | 64 heads → 72, in 9 output groups |
| speculative decode | off |
| load format | `safetensors` |
| all-reduce transport | `PYNCCL` (see **The one-shot RoCE all-reduce is not engaged on this fabric**) |

Two of those are load-bearing and are checked, not assumed:

- the MoE backend must **not** be the b12x one — that path computes
  SwiGLU with no clamp on the small-batch decode step, so the model's
  clamped-SwiGLU limit would be silently dropped, with no error signal;
- the model's clamped-SwiGLU limit must still be what the engine
  resolves, because the chain that carries it from the config to the
  kernel is several layers deep.

Gate 8 checks all of that, along with the rest of the contract. Each
expectation is read from the engine's own startup log, its own `/metrics`
and its own resolved config; a missing marker is a failure, not a skip.
Every file that prints one of those markers is hash-pinned in the
engine-identity manifest, so a newer image cannot change what a marker
says without the gate noticing.

### The one-shot RoCE all-reduce is not engaged on this fabric

The launcher sets `VLLM_ENABLE_ROCE_ALLREDUCE=1` and
`VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB`, matching upstream's own three-node
recipe. The image also ships a one-shot RoCE collective
(`B12X_ROCENANTE`) that needs **both** RoCE rails of a link to reach the
peer. The topology this recipe ships has **one cable per leg**, so that
collective cannot form. The engine's own log says so and moves on:

```
RoCEnante initialization failed: RoCE all-reduce connect failed:
rank 0: … 1 of 2 RoCE rails have a link to rank 1 …
```

(abridged: the line continues with each rank's local and remote interface
list.)

and then reports what it engaged instead:

```
Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'
out of potential backends: ['FLASHINFER_PCIE_IPC', 'B12X_PCIE', ...]
```

(abridged: the engine lists every backend it could have used.)

`B12X_ROCENANTE` is not in that list. The reduction is carried by stock
NCCL over the same fabric, and it works — every number published in this
recipe was measured that way.

None of this is visible from the recipe's command line: the knob is set,
the collective is absent, and the fallback is silent. So the expectation
now lives in the engine-identity manifest and gate 8 asserts it, reading
the engine's own dispatch line for every group the engine reports. The
**expected** value on this topology is `PYNCCL`.

**If the transport ever changes**, gate 8 will **fail** — deliberately,
because the manifest still describes the old one. Read the transport out
of the engine's own dispatch line and update
`launch/tp3-kit/engine-identity.json`
(`resolved_backends.expect.allreduce_backend`) to match, and the gate
asserts the new one. A fabric change should be a recorded decision rather
than an unrecorded one.

Note that a second rail per leg is **off NVIDIA's documented topology**:
their cluster guidance says to use one cable per link, and that connecting
two devices with two cables will not improve performance. The collective
needs both rails of a link to reach the same peer, so engaging it means a
cabling NVIDIA does not recommend. That is a performance experiment to
run deliberately, with the numbers recorded — not a fix.

This is not a defect to fix in the recipe. The [DGX Spark release
notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html) record
that NCCL 2.30u1 "has updated support for connecting three DGX Spark
systems in a ring topology", and NVIDIA's
[cluster guidance](https://docs.nvidia.com/sync/0.97.6/cluster-assistant.html)
describes that topology as "three cables for a direct connection in which
each device connects to the other two" — which is exactly the triangle
this recipe ships. The stock path is therefore the supported one, and the
published figures here were measured on it. Enabling the one-shot
collective is a performance experiment, not a correctness fix.

### Re-audit triggers

- **A new upstream image.** The engine-identity gate refuses any image
  whose build stamp or engine-critical file hashes differ. That is the
  adoption gate: re-audit, then re-pin.
- **A rebase that touches TP=3.** Upstream vLLM rejects TP=3 for this
  model outright — 64 attention heads do not divide by 3 — so the
  head-padding this recipe depends on is a fork-local mechanism, not an
  upstream one, and an upstream change to the parallel-config checks or
  to this model's attention code can silently break it.
- **A 3-node TP mesh is an off-the-recommended shape.** Multi-node TP on
  DGX Spark has a documented upstream failure mode where the ranks sit in
  a collective that never completes, with no NCCL error and the nodes
  drawing idle power (upstream vLLM issue #46097, reported on a 4× Spark
  mesh). It is worth knowing before you debug a hang.

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
| Compile/CUDA-graph arenas | ~2 GiB |
| Host OS + daemons | ~8 GiB |
| **Free (serving, static estimate)** | **~8 GiB — see below** |

This table is an arithmetic estimate, not a measurement, and the
difference matters. Once the weights are resident, the *measured*
free memory on a loaded box can sit far below that estimate and moves
with the page cache and the engram staging path: observed readings on
a running triangle have ranged from a few tens of MiB to a few GiB,
and the engine has been OOM-killed by the kernel at the low end. Treat
the headroom as a budget to measure, not a constant:

- the bring-up drops the page cache before the load, which is what
  keeps the staging path from colliding with the loader;
- the loader's own gate compares its configured budget against
  *available* memory, which on this unified-memory kernel can read as
  zero even when memory is available — the boot probe therefore samples
  the minimum **reclaimable** headroom (free + reclaimable slab +
  reclaimable page cache) across the whole boot and prints it, so a thin
  run is visible in the log instead of at the first request;
- if a boot is killed, the kernel log says so (`journalctl -k | grep
  -i "killed process"`), and the fix is to free page cache before the
  load rather than to lower gmu.


## Engram on NVMe

The two ~94.5 GiB engram n-gram table shards (~189 GiB total, i.e.
~203 GB) remain on each node's local NVMe. They are referenced by the
model's index but are never loaded into host or GPU memory — the
`table_memory:disk` setting instructs the engine to read rows on demand
during inference. This is the same approach the community SGLang recipe
describes.

### Auto-prefetch is not available for this checkpoint

The engine has an optional auto-prefetch path for the engram tables, and
on this hardware it refuses it for two reasons of its own: the filesystem
is not a recognised network FS (NFS/Lustre), and the checkpoint is larger
than 90% of available RAM.

The second reason is the decisive one, and it is arithmetic rather than
configuration. The precondition is that the checkpoint fits inside 90% of
the memory **available at load time**. This model's checkpoint is larger
than 90% of the whole box's ~121 GiB of unified memory — ~109 GiB — so it
fails that test on an idle box with nothing else running, let alone on a
loaded one. Moving the checkpoint to a network filesystem fixes the first
reason and leaves the second exactly as it was.

So the posture this recipe already takes —
`--engram-config '{"cpu_offload":false,"table_memory":"disk"}'`, the same
setting upstream's own three-node recipe uses — is the answer, and there
is no further work to do here. If a smaller checkpoint is ever wanted, an
FP8 subset or a lower-bitpoint variant, that is a checkpoint question and
it would change the arithmetic; it is not a prefetch question.

## Speculative decode: tested and rejected

DSpark speculative decoding (k=5) was tested and rejected on capacity
grounds. The draft model adds ~3 GiB per rank, which at TP3 leaves
insufficient memory for the KV pool and CUDA graphs — the boot OOMs at
every gmu and KV combination tried (0.83/0.85 × 1g/2g). The trade:
~51 tok/s (community SGLang number with spec decode) vs 29 tok/s, at
the cost of the KV pool and the context length — the draft model does
not fit beside this pool. (That is a trade against *this* recipe's own
capacity with spec decode off; against the community's 750K pool this
recipe still carries more than double.) For context-bound
workloads, the capacity is worth more than the speed.

## Comparison with other V4.1-Flash recipes

Every figure in the other rows is **other people's published number**,
read from the repository linked in that row — not measured here, and not
comparable cell for cell: the checkpoint, the weight format, the engine
and the transport all differ. They are here to show the shape of the
landscape, not to claim a win.

| Recipe | Engine | TP | Image | KV pool | Context |
|---|---|---|---|---|---|
| **This one** | vLLM (b12x fork), native weights | **3** | **public Docker Hub, digest-pinned** | **~2M tokens** | **512K** |
| [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | SGLang | 3 (a 4-node line too) | SGLang base pinned by digest + a local overlay | 750K tokens (TP3) | 256K limit (TP3) |
| [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f) | vLLM (custom fork) | 3 or 4 | built from the repo; one image serves every shape | 1.3M–5.5M (TP3) – 11.6M (TP4) tokens | 256K (TP3, 4k) / 1M (TP3, TP4) |
| [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks) | TensorFold (a Zig engine); older vLLM lanes | 2 or 4 | engine built from source | 1.05M (TP2) – 29M positions | 300K – 1M |

Two things this recipe does differently from all three of them: the engine
image is **public** — pull it by digest and the boot is reproducible —
and the shape is three nodes on a RoCE triangle rather than a
switchless ring of four.

## Upstream's own 3-node recipe

The project that publishes the image ships a recipe system of its own —
`recipes/<model>.yaml`, plus `recipes/3x-spark-cluster/`,
`run-recipe.py`, `cluster_topology.py`, `autodiscover.sh` and a
`Dockerfile.mxfp4` — and among those recipes is
`recipes/3x-spark-cluster/deepseek-v4.1-flash.yaml`: the same model, the
same image lineage, the same three-node triangle.

It is worth reading, and it is not a competitor. It is a YAML for their
cluster runner on their cluster shape, with `cluster_only: true` and
no acceptance gates of its own. It
is also the best available evidence that a three-node V4.1-Flash lane is
reproducible from public parts.

Where the two differ:

| knob | this recipe | upstream 3-node |
|---|---|---|
| `--kv-cache-memory-bytes` | `4g` | `0.5g` |
| `--gpu-memory-utilization` | `0.85` | `0.87` |
| `--max-num-seqs` | `8` | `2` |
| `--max-num-batched-tokens` | `8192` | `4096` |
| `--max-model-len` | `524288` | `auto` |
| speculative decode | off | dspark, 7 tokens |
| `--load-format` | `safetensors` | `b12x` |
| linear / MoE / attention backend | unset — the engine's own policy | pinned `b12x` ×3 |

Two rows are the recipe's whole thesis, and they pull in opposite
directions from upstream's: upstream asks for **8× less KV cache** and **4×
fewer concurrent sequences**, with the context derived from that smaller
pool, which is a speed-shaped lane. This recipe asks for a **large pool and a pinned 512K
context**, which is a capacity-shaped lane, and accepts roughly 29 tok/s
for that.

Two rows are not a preference but a measured constraint on this hardware.
The `safetensors` loader instead of `b12x` is wall 3 in the bring-up
below: the b12x loader's pinned bounce buffers OOM a busy head node.
And leaving the backends unset is deliberate — the engine's own policy
is what gate 8 audits, so pinning them would remove the very thing the
gate exists to check.

## Upstream delta to watch

This recipe pins a specific nightly, so the useful question is what has
moved upstream since. The pinned build is `v0.1.dev21830+g8244d1c44` from
the 2026-10-15 nightly, which sits inside the upstream v0.32.0 window —
most of v0.32.0 is already here. The table below separates what the pin
already contains from what arrived after it, because a list that mixes the
two cannot be read as a delta.

| upstream | status | in the pin? | relevance here |
|---|---|---|---|
| [#64441](https://github.com/vllm-project/vllm/pull/64441) [1/N] top-k order | **open** | no | batch invariance for DeepSeek V4 — the upstream defect, see the note below |
| [#64870](https://github.com/vllm-project/vllm/pull/64870) [2/N] compressor | **open** | no | the same series |
| [#64884](https://github.com/vllm-project/vllm/pull/64884) [3/N] mHC | **open** | no | the same series |
| [#58720](https://github.com/vllm-project/vllm/pull/58720) indexed expert-mapping lookups | v0.32.0 | yes | −25 s weight load on DGX Spark; the load here takes ~200 s |
| [#54048](https://github.com/vllm-project/vllm/pull/54048) cuBLAS out_dtype router GEMM | v0.32.0 | yes | a GB10-specific GEMM path |
| [#58925](https://github.com/vllm-project/vllm/pull/58925) encoder CUDA-graph packing for V4.1 | v0.32.0 | yes | mean TTFT on vision workloads |
| [#58132](https://github.com/vllm-project/vllm/pull/58132) / [#59532](https://github.com/vllm-project/vllm/pull/59532) SWA bounded replay | v0.32.0 | yes | the SWA geometry the block-size story above depends on |
| [#58877](https://github.com/vllm-project/vllm/pull/58877) OAI Triton MXFP4 MoE on SM12x | v0.32.0 | yes | a second MXFP4 MoE path for sm_121 |
| [#46963](https://github.com/vllm-project/vllm/pull/46963) / [#60452](https://github.com/vllm-project/vllm/pull/60452) NVFP4 KV + XQA decode on SM12x | v0.32.0 | yes | halves KV bytes per token |
| [#59604](https://github.com/vllm-project/vllm/pull/59604) ModelOpt IQ2_XS/XXS/Q8_0 via b12x | Oct 13 | yes (pre-pin) | smaller weights → more pool, if a checkpoint exists |
| [#54163](https://github.com/vllm-project/vllm/pull/54163) DFlash/DSpark keep the last prefix-cache block | ~Oct 23 | no | upstream-validated on 4× Spark TP4 with GLM-5.3-Flash + DFlash2 — the closest prior art to this hardware |
| [#63421](https://github.com/vllm-project/vllm/pull/63421) AV draft-budget selector | Oct 26 | no | host-side latency under adaptive verification |
| [#63886](https://github.com/vllm-project/vllm/pull/63886) DSpark batch-sharded sequential block | ~Oct 30 | no | the closest upstream artefact to draft handling at TP>1 |
| [#62183](https://github.com/vllm-project/vllm/pull/62183) / [#62384](https://github.com/vllm-project/vllm/pull/62384) APC + SWA bounded replay under PCP | Oct 19 / 23 | no | prefix cache + SWA geometry, but PCP is a datacenter shape |

The table above is a selection, not the whole delta: roughly twenty pull
requests moved after the pin, and the eight marked `no` are the ones worth
watching. The only correctness-class work among them is the
batch-invariance series — three pull requests, all still open. Everything
else is performance or plumbing.

**The batch-invariance series is open.** All three parts of the upstream
DeepSeek-V4 batch-invariance work are still unmerged, the third filed
within days of the first. Any claim that upstream has fixed DS-V4.1
batch invariance would be wrong today, and would start being wrong in the
other direction the day it merges. This recipe does not make that claim,
and it does not claim to cover the property either: **gate 9 measures
repeat- and cache-state invariance of a single request, not batch
composition**, so it is adjacent to the upstream defect rather than a
substitute for the upstream fix. Batch-composition invariance needs the
engine patch; measuring it on a lane would need a second, concurrent
harness, which this recipe does not ship.

**The rest is watch, not adopt.** Most of this delta is performance or
plumbing. Two entries would change what the recipe *is* rather than how
fast it runs: the NVFP4 KV path — a different KV byte layout, so a
different pool figure and a different `/metrics` block size — and the
ModelOpt quantisation path, which is a different checkpoint. The ModelOpt
**code** is already inside this pin — what is missing is a quantised
checkpoint to use it with. Either way, moving onto either path is a
re-audit rather than an upgrade.

## Credits

- The eugr/spark-vllm-docker project for the b12x images and the
  GB10-optimized kernels
- The MiaAI-Lab recipe for the SGLang comparison numbers and the NCCL
  buffer tuning discovery (their documentation of the 4.7 GiB → 139 MiB
  fix was the key that unlocked the head node)
- The upstream vLLM `safetensors` loader path, which avoids the
  pinned-bounce-buffer OOM the b12x loader hits on a busy head node

## License

MIT
