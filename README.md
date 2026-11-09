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
on a public Docker Hub image. Capacity-optimized: a **5 GB (4.7 GiB) KV
reservation** at **1M context** — the reservation is fixed by a pin, and
the engine's token figure scales with the context value at the same
block count (roughly 5.1M tokens at 1M with the capacity overlay — one full
1M request fits by pool arithmetic; admission is verified to ~900K). **~35 tok/s single-stream** (prose,
no spec decode — the LM-head kernel, L2 set-aside and loader below are all
part of the shipped posture), a cold boot loads the weights in **~32 s**
(b12x O_DIRECT loader). A short request's first token lands in
**single-digit seconds even beside a deep multi-hundred-K prefill**
(parallel prefills + the decode-under-prefill share, measured — see
FIXLOG). And a short request *decoding* beside that same prefill runs
at **~25 tok/s** — the legacy scheduler gave it ~1.4. Engram n-gram tables are served from NVMe,
freeing 31 GiB/rank of host memory that would otherwise be consumed.
The bring-up maps every memory trap on GB10 unified memory: NCCL buffer
pinning, engram host staging, loader bounce buffers, and the startup
free-memory check — each with its measured fix.

(b12x is the kernel lineage of these images — it prefixes their env
vars, the MoE backend and the loader; "the b12x fork" below means this
image family.)

**Contents** — Quick start · Served numbers · The decode-step stack ·
One-shot RoCE transport (optional) · Why capacity over speed · The
bring-up: six walls · Troubleshooting · Stack · Configuration ·
Re-audit triggers · Hardware · Engram on NVMe · Vision · Speculative
decode · Comparison · Upstream's own 3-node recipe · Credits · License

**A published vLLM TP3 recipe for V4.1-Flash on a publicly available
image — three commands to a serving lane, with acceptance gates.** The
project that publishes the image also ships its own
`recipes/3x-spark-cluster/deepseek-v4.1-flash.yaml` — the same model,
the same image lineage, the same three-node shape, for their cluster
runner. The difference between the two is the trade this recipe makes;
the side-by-side is in **Upstream's own 3-node recipe** below, and where
they differ, that table is the reason this recipe reads the way it does.

The scope here is the public image, not the B12X kernel lineage: a
community TP=3 vLLM fork built from source reaches a slightly larger
pool (5.5M vs ~5.1M tokens) at the same 1M context when built with a
64 KiB-page kernel and the memory saver; on a stock 4 KiB-page box its
three-node shape is 256K. Its figures are in the table below. The
speed-optimized SGLang recipe from the community achieves higher tok/s
but at about a seventh of the KV capacity and a quarter of the context —
see **Comparison** below.

Prerequisites:

- three DGX Sparks wired as a full CX-7 RoCE triangle (three cables,
  each box connected to the other two, one /24 per pair, MTU 9000)
  plus a management LAN
- the ~510 GB checkpoint staged on every box (**Staging**, below)
- the digest-pinned image staged on every box (**Staging**, below)
- Docker with GPU support on every box

## Quick start

Edit `config/cluster.env` first: set the three box names, the three
management IPs, the checkpoint path (`MODEL_DIR` — it must be the path
**on the boxes**, and the shipped default is a placeholder), and the two
fabric selectors (`NCCL_IB_HCA` — the RoCE port pair, from `ibdev2netdev`
on each box — and `NCCL_SOCKET_IFNAME`, the management interface). The
preflight refuses to run while the fabric selectors still hold their
placeholder.

### Staging the checkpoint and the image (once per box)

```sh
# the image, pinned by digest (a tag can move; a digest cannot)
docker pull eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595

# the checkpoint (~510 GB) into MODEL_DIR on every box
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e10 --local-dir "$MODEL_DIR"
```

The boot gate proves the checkpoint directory is a complete snapshot: it
expects 48 shards at the root (engram included), the 12 engram index
entries, and a `_DOWNLOAD_COMPLETE.json` marker. `hf download` does not
write that marker — create it yourself **after** the download has fully
completed, so the gate is checking your honest claim:

```sh
echo '{"complete": true}' > "$MODEL_DIR/_DOWNLOAD_COMPLETE.json"
```

Then:

```sh
./launch/ds41-flash-tp3-up preflight    # image digest + content, checkpoint, fabric
./launch/ds41-flash-tp3-up stage-env     # copies cluster.env + launcher + tp3-kit
./launch/ds41-flash-tp3-up start --yes   # head first, then the workers; the API is on :8600 a few minutes later
```

`preflight` is read-only: it inspects the image that is already on the
box and never pulls. It checks three things per box — the image's
manifest digest matches `IMAGE_DIGEST`, the sha256 of the image's
RootFS layer list matches `IMAGE_CONTENT_SHA256`, and the checkpoint
directory is present — then reports the box's available memory.

### First request

When health is green, one request proves the lane answers (the full
acceptance suite — ten gates with expected outputs — is
`docs/VALIDATION.md`):

```sh
curl http://HEAD_IP:8600/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "main", "messages": [{"role": "user", "content": "Reply with the single word: ready"}], "max_tokens": 8, "temperature": 0}'
```

Expect HTTP 200, a short completion, and `finish_reason: "stop"`.

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

Measured at the shipped configuration (gmu — `--gpu-memory-utilization` — 0.85, 5 GB (4.7 GiB) KV + the capacity overlay, 1M context,
parallel prefills ×2 + decode-aware policy, decode-under-prefill share 0.2,
kernel autotune off, no spec decode, engram on NVMe):

| Metric | Value |
|---|---|
| Single-stream decode (prose) | ~35 tok/s (b12x MoE + LM-head kernel + L2 set-aside; ~31 without the 2026-11 kernel stack) |
| TTFT (short prompt, idle) | 1–2 s |
| TTFT (short prompt beside a deep prefill) | **≈5–6 s** at the shipped share 0.2 (≈4.6–4.8 s at 0.3–0.5; tens of seconds before parallel prefills; the deep prefill itself is unaffected) |
| Decode beside a deep prefill | ~1.4 tok/s on the legacy scheduler → **~25 tok/s** at the shipped share (the ladder and its TTFT price: failure mode 16) |
| Weights per rank | 95.4 GiB |
| KV reservation | **5 GB (4.7 GiB)** (fixed by `--kv-cache-memory-bytes`; with the capacity overlay the pool reads ~5.1M at both contexts) |
| Context length | **1,048,576 (1M)** — verified by deep-prefill admission to ~900K at the shipped batch width |
| Concurrency at 1M ctx | **~4.9×** — pool depth, not simultaneous streams (`MAX_NUM_SEQS=8` caps live sequences; see the note under Configuration) |
| Concurrency at 512K ctx | **~9.7×** — pool depth, as above |
| Cold boot (load→health) | a few minutes (weight load ~32 s with the b12x loader, ~190 s on the safetensors fallback; the first boot pays kernel compilation, later boots save ~60 s via the persistent cache) |
| Temp-0 determinism | identical across runs for most prompts (gate 2). A narrow tie class exists: prompts whose forced answer has an equally-likely variant (a trailing period, a formatting choice) can flip across boots and warm/cold states at temp 0 — measured on three postures, unchanged by context or scheduler knobs. Treat single-word temp-0 outputs as acceptance checks, not correctness proofs. The long-prompt path is a separate gate (9). |
| Tool calling (deepseek_v41) | working |
| Reasoning parser (deepseek_v41) | working |

(All gate numbers refer to `docs/VALIDATION.md`, which ships with the recipe.
Figures are single-boot point estimates; boot-to-boot variance measured ≈2%.)

**The capacity overlay** is a two-file patch the bootstrap installs before
the boot gate (FIXLOG failure mode 15): stock KV accounting charges every
request the worst-case span of all its sliding-window groups; the overlay
charges each request its window+lookahead and the unsettled batches once —
which is why the same 5 GB reservation reports ~5.1M tokens.

### Where the boot time goes

The cold boot above is a few minutes end to end; the weight load is ~32 s of
it. The shipped configuration runs the b12x kernel autotune **off**
(`B12X_AUTOTUNE=0`): the tuned alternative benchmarks kernel plans for
every shape this model and batch width produce, and its tuning arena
self-budgets at **half of free memory** — which has OOM-killed boots on
this unified-memory kernel, where `MemAvailable` can read as zero and
nothing self-limits. With autotune off, plans fall back to validated
heuristic defaults shipped as prebuilt cubins: boots run in roughly half
the tuned time, decode speed is unchanged (measured), and the cost is a
**one-time cold-prefill penalty of a few percent per new document**
(prefix-cache hits — the bulk of real traffic — are unaffected). Set
`B12X_AUTOTUNE=1` only with a specific reason.

A second tuner (the fork's FlashInfer autotune) is independent of that
switch and the shipped launcher disables it too
(`--no-enable-flashinfer-autotune`): on this model and pin it benchmarked
zero configurations (its log line reads `Saved 0 configs`), so removing
it is boot hygiene, not a speed lever. The launcher also drops the page
cache before the load (best-effort, requires passwordless sudo — the
manual command is wall 4) and caps the engine's glibc malloc arenas at 2.

What remains uncached is the compile of new *shapes*: raising
`MAX_NUM_BATCHED_TOKENS` (or any knob that creates new kernel shapes)
re-pays compilation on the next boot, and wider shapes have been
observed to kill worker ranks during the compile phase on this unified
memory. Two knobs in the pinned build touch compile persistence —
`VLLM_USE_AOT_COMPILE` and `VLLM_USE_MEGA_AOT_ARTIFACT` (on current
torch both default **on**; no AOT artifact is baked into the image, so
enabling them changes *what is compiled*, not *what is reused*).
Persistence would have to come from a mount keyed by the image build
stamp — a stale artifact is a *wrong kernel*, which is a worse failure
than a slow boot — so this recipe ships neither knob changed.

## The decode-step stack (all measured and kept, all one-line undo)

The ~35 tok/s figure is not one knob; it is the b12x MoE backend plus four
additions landed and A/B-measured one variable at a time (see FIXLOG for the
failure modes that did not survive):

1. **The LM-head kernel** — the engine hard-codes the `cutlass` MXFP8 GEMV
   for linear heads; the image already ships a b12x warp-MMA kernel ~2.1×
   faster at the decode shape with bit-identical output
   (`VLLM_MXFP8_LINEAR_BACKEND=auto`, +~6%).
2. **L2 behaviour** — the indexer's prefetcher stays ON (disabling it was
   measured at ~−5%), and a 12 MB persisting-L2 set-aside lands another
   ~1–2% (`VLLM_GLM53_L2_PREFETCH_PERSIST_MB=12`).
3. **The b12x loader** — not a decode win but a boot win: ~32 s loads.
4. **The persistent compile cache** — ~60 s off every cold start after the
   first.

Each of these is a launcher line or a kit overlay; removing the line
restores the prior behaviour. The capacity overlay and the parallel
prefill policy below are the same pattern.

## The one-shot RoCE transport (optional layer)

The stock configuration carries its TP all-reduce on NCCL over the triangle. The
image's own one-shot RoCE collective cannot form on a one-cable-per-pair
fabric (it demands two rails per peer pair — the bring-up gate was built on
exactly that silent fallback). `tp3-kit/sparknet/` ships an optional layer
that switches the small collectives to a per-peer-HCA-map one-shot transport
measured at **~+5% single-stream and +12–15% two-stream aggregate** on this
posture, with no memory cost: routing templates you edit for your fabric, a
node-map template, the fenced-NCCL build recipe, and the GID-index trap that
the first boot found the hard way. Off by default; the launcher refuses a
half-armed boot. See `launch/tp3-kit/sparknet/README.md`.

## Why capacity over speed

The community SGLang recipe for this model on the same hardware
achieves ~51 tok/s but with a 750K KV pool and 262K context. This
recipe carries roughly seven times the pool at 1M context — at ~35
tok/s, with short-request TTFT protected by the parallel-prefill
policy.

For workloads that are context-bound (RAG with large knowledge bases,
long legal or research documents, multi-turn conversations with massive
history), the KV pool and context length are the binding constraints,
not generation speed. A 300K-token document is rejected outright at a
262K context; a ~16 tok/s difference in generation speed is invisible when
streaming at reading speed.

The trade-off table:

| | This recipe (vLLM) | Community SGLang |
|---|---|---|
| KV pool | **~5.1M tokens** (5 GiB reservation + the capacity overlay) | 750K tokens |
| Context | **1M** | 262K (their "256k limit") |
| Concurrency @ 1M | **~4.9×** | — |
| Speed | ~35 tok/s | **51 tok/s** |
| Short TTFT beside deep prefills | **≈5–6 s** (parallel prefills + decode share) | not published |
| Spec decode | off (rejected — see below) | on (speed) |
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

### 3. Weight loading: the b12x loader ships (~32 s); safetensors is the fallback

**Updated (measured, 2026-11): the b12x loader is now the shipped default —
with the capacity overlay and the 5 GiB posture it loads the same checkpoint
in ~32 s versus ~190 s for safetensors mmap, pool and decode unchanged, and
the historical OOM no longer reproduces.** The original wall, kept for
context: the b12x O_DIRECT loader uses pinned bounce buffers that add
~10–16 GiB of unreclaimable host memory during weight loading, which on a
busy head node tipped boxes into kernel OOM. If you hit that class of
failure on a smaller-memory configuration, the conservative fallback is:

```sh
--load-format safetensors    # instead of b12x
```

Plain safetensors streams through kernel-evictable page cache instead
of pinned bounce buffers — slower (~6×), but unconditionally safe.

### 4. The startup free-memory check

The fork checks available GPU memory at startup against the requested
utilization fraction. On a busy head node with even 2 GiB of extra
daemons, the default 0.92 target fails. Fix: set gmu to 0.85 — the
launcher drops the page cache before every boot (best-effort, needs
passwordless sudo; otherwise run `sync; echo 3 > /proc/sys/vm/drop_caches`
manually).

### 5. Vocabulary divisibility at TP3

This model's vocabulary is 129,280, and 129,280 % 3 ≠ 0, and that has to
be settled before the engine serves. The fork's runtime TP3 padding
handles attention heads, o_groups and drafter experts, so the checkpoint
itself is the NATURAL one; the served-name path is the other half, and
the lane serves a single name (e.g. `main`). The boot gate proves the
model directory is the original pull before the engine starts.

The assertion that fired during bring-up is not present in the pinned
build: the only vocabulary-divisibility check in this image is in the
b12x ngram-embedding path, and it divides a padded vocabulary. Treat the
single-served-name posture as the requirement — the reason it works is
not readable in the shipped build.

### 6. io_uring requires seccomp=unconfined in Docker

The b12x image's I/O infrastructure uses io_uring, which Docker's
default seccomp profile blocks. Fix:

```sh
--security-opt seccomp=unconfined
```

## Troubleshooting

The six walls above are the bring-up failures; the running symptom →
root cause → fix table — including defects found in this recipe's own
scripts during re-runs of the published quick start — is
**`docs/FIXLOG.md`**. Diagnose forward from the symptom; nothing here
rolls back automatically.

## Stack

| Component | Pin |
|---|---|
| Image | `eugr/spark-vllm-b12x:nightly-20261015` (digest `sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (~510 GB) |
| vLLM | v0.1.dev21830+g8244d1c44 (fork build) |
| MoE backend | `B12X_MXFP4_MXFP8` (b12x, with the tiny-decode kill switch on) |
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
CTX=1048576                        # 1M; pool bytes/blocks unchanged by this knob
KV_CACHE_MEMORY_BYTES=5g
PAR_PREFILLS=2                     # short-request TTFT beside deep prefills (see FIXLOG);
PREFILL_POLICY=decode-aware        # requires the count above 1
PREFILL_COMPUTE_SHARE=0.2          # decode-under-prefill share, shipped posture (FIXLOG mode 16)
PREFILL_TOKENS_PER_STEP=2048       # the contended-prefill cap paired with it
B12X_AUTOTUNE=0                    # heuristic cubin plans: ~half boot time, decode parity

# Serving
PORT=8600
TP=3
MOE_BACKEND=b12x                      # the measured-keep MoE backend; empty = engine auto
                                      # (DEEPGEMM_MXFP4)
B12X_W4A8_TINY_DECODE=0               # REQUIRED with b12x — it carries the SwiGLU clamp
                                      # (FIXLOG failure mode 14)
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=8192       # do not raise without a memory plan: 16384
                                  # OOM-killed both worker ranks during the b12x
                                  # compile phase (see FIXLOG failure mode 9)
BLOCK_SIZE=256
LIMIT_MM='{"image": 32}'             # images per prompt: the checkpoint's 32-layer
                                  # ViT tower is served above 0 (verified live);
                                  # at 0 the engine stubs the tower (text-only)

# Loader (b12x: ~32 s load vs ~190 s; safetensors is the conservative fallback)
LOAD_FORMAT=b12x
ENGRAM_CONFIG='{"cpu_offload":false,"table_memory":"disk"}'

# NCCL (the critical caps)
NCCL_BUFFSIZE=1048576
NCCL_LL128_BUFFSIZE=262144
NCCL_PROTO="^LL128"
NCCL_MAX_NCHANNELS=8

# Allocator
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # mandatory: :True → NaN logits
                                                     # on >64-token prompts (FIXLOG 8)
# the launcher also forwards MALLOC_ARENA_MAX=2 to the engine
# (see "Where the boot time goes")

# Parsers
TOOL_CALL_PARSER=deepseek_v41
REASONING_PARSER=deepseek_v41

# Model name (single alias; see wall #5)
SERVED_MODEL_NAME=main
```

Concurrency here means how many full-length requests fit in the KV pool.
It is a capacity figure, not a promise about simultaneous streams: the
shipped `--max-num-seqs 8` caps how many requests are served at once, so
the ~9.7× figure at 512K describes pool depth, not concurrency.

### BLOCK_SIZE=256 — what it is, and what `/metrics` will say

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
| MoE backend | `B12X_MXFP4_MXFP8` (the b12x MXFP4 experts; the launcher pins `--moe-backend b12x`) |
| dense linear | `FlashInferCutlassMxfp8LinearKernel` |
| attention | b12x (DSA indexer) |
| TP=3 head padding | 64 heads → 72, in 9 output groups |
| speculative decode | off |
| load format | `b12x` |
| all-reduce transport | `PYNCCL` (see **The one-shot RoCE all-reduce is not engaged on this fabric**) |

Two of those are load-bearing and are checked, not assumed:

- the MoE backend IS the b12x one, pinned by the launcher — and the
  thing that makes that safe is the **tiny-decode kill switch**
  (`B12X_W4A8_TINY_DECODE=0`, forwarded by the launcher): the b12x
  MoE's tiny-decode path computes SwiGLU with no clamp on the
  small-batch decode step, silently dropping the model's clamped-SwiGLU
  limit. With the switch off, the general kernel runs and receives the
  clamp. The verifier asserts the switch's resolved value from the
  container's own environment on every run — the check that used to
  forbid the backend now guards its safety condition;
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
peer. The topology this recipe ships has **one cable per pair of boxes** —
three cables, each device connected to the other two, and **both** of each
device's ports in use — so that collective cannot form. The engine's own
log says so and moves on:

```
RoCEnante initialization failed: RoCE all-reduce connect failed: rank 0: … 1 of 2 RoCE rails have a link to rank 1 …
```

(one line; abridged — it continues with each rank's local and remote
interface list, and repeats the same failure for the other two ranks.)

and then reports what it engaged instead:

```
Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'
out of potential backends: ['FLASHINFER_PCIE_IPC', 'B12X_PCIE', ...]
```

(abridged: the engine lists every backend it could have used.)

`B12X_ROCENANTE` is not among the backends the engine **engaged** — the
enabled subset is `['PYNCCL']` — though it *does* appear in the potential
list. That is the whole mechanism: the engine tried the one-shot
collective, the collective disqualified itself on its own connectivity
check, and the reduction fell through to stock NCCL over the same fabric.
It works, and every number published in this recipe was measured that
way.

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

This is not a defect to fix in the recipe, and there is **nothing to
re-cable**. The [DGX Spark release
notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html) record
that NCCL 2.30u1 "has updated support for connecting three DGX Spark
systems in a ring topology", and NVIDIA's
[cluster guidance](https://docs.nvidia.com/sync/0.97.6/cluster-assistant.html)
describes that topology as "three cables for a direct connection in which
each device connects to the other two" — which is exactly the triangle
this recipe ships, with both of each device's ports used, one toward each
of the other two boxes.

Engaging the collective would need a **second cable between the same pair
of boxes**, and that is off NVIDIA's documented topology: their guidance
says to use "only one cable per link" — a link being a pair of devices, or
a device and its switch — and that "connecting two devices with two
cables will not improve performance". So the collective is unreachable on
the supported topology, the stock path is the supported one, and every
number published in this recipe was measured on it. Engaging the one-shot
collective would be a performance experiment against NVIDIA's own
guidance — not a fix.

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
wired as a full CX-7 RoCE triangle (one /24 subnet per pair of boxes —
one cable per pair, 200G per cable, MTU 9000) plus a management LAN for SSH and
API access.

Per-node memory budget at the shipped configuration:

| Component | Size |
|---|---|
| Model weights | 95.4 GiB |
| KV cache (5 GiB reserved) | 4.7 GiB |
| NCCL buffers (capped) | 0.14 GiB |
| CUDA context + framework | ~4 GiB |
| Compile/CUDA-graph arenas | ~2 GiB |
| Host OS + daemons | ~8 GiB |
| **Free (serving, static estimate)** | **~7 GiB — see below** |

This table is an arithmetic estimate, not a measurement, and the
difference matters. Once the weights are resident, the *measured*
free memory on a loaded box can sit far below that estimate and moves
with the page cache and the engram staging path: observed readings on
a running triangle have ranged from a few tens of MiB to a few GiB,
and loads have been OOM-killed at the low end. Treat
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
FP8 subset or a lower-bit-width variant, that is a checkpoint question and
it would change the arithmetic; it is not a prefetch question.

## Vision: built in, one flag

The 1017 checkpoint carries a 32-layer ViT tower, and the fork ships the
matching image processor inside itself — constructed from the model
config, no `preprocessor_config.json` required. The engine loads the
multimodal class unconditionally on this hardware and builds the tower;
setting `LIMIT_MM` above zero turns it from stubbed to served:

```sh
LIMIT_MM='{"image": 32}'  # images per prompt; 0 = text-only (the tower
                          # is stubbed at 0, served above it)
```

Verified live: exact shape/color/position recovery on a generated test
image through the OpenAI-compatible endpoint (`image_url` with a data
URL); a 12-image ordered identification scored 12/12 in ~4 s; a full
32-image batch processes in ~4 s (~185 tokens per image after
downsampling — small images cost far less than the 1,024-token cap).
Text decode speed unchanged (250 tokens in ~8 s) and the KV pool
unchanged. **Budget note (b12x MoE posture):** vision requests emit a
short run of invisible completion tokens (counted in the token usage,
empty in the text fields) before the answer — give vision requests
`max_tokens ≥ ~200` or the budget can be consumed before the answer
lands. One VLM-scale caveat: per-image identification stays strong
at the limit, but aggregate counting drifts (a 32-image batch was
counted as 30).

## Speculative decode: tested and rejected (final)

DSpark (the fork's built-in speculative drafter) was tested across **six boots** covering
every knob-accessible configuration, and the rejection is final — but
the mechanism is not the one the first capacity reading suggested, and
the distinction matters for anyone re-trying it:

1. **The drafter itself is cheap and correct**: ~3 GiB per rank resident,
   loads in seconds, and the k=3 capture ladder is a fraction of k=5's.
2. **The early boot OOMs were the kernel tuners, not the drafter.** The
   b12x autotune arena (which self-budgets at half of free memory) and
   the fork's FlashInfer autotuner each killed boots that now complete:
   with `B12X_AUTOTUNE=0` and `--no-enable-flashinfer-autotune`, a spec
   boot at a reduced 2.3 GiB KV pool **allocates the pool, captures its
   CUDA graphs, and serves** — the first spec boot on this hardware to
   get that far.
3. **The real wall is steady state, and it is fatal.** Roughly a minute
   into its first deep-context prefill (a few hundred K tokens — exactly
   the RAG shape this recipe exists for), the head rank was
   kernel-OOM-killed: the drafter's ~3 GiB/rank residency leaves the
   head unable to absorb the deep-prefill memory transient at any pool
   that leaves the drafter room. Smaller pools cut concurrency toward
   the 50% line without adding headroom; the failure is the prefill
   transient, not a per-sequence buffer.

The conclusion: on this pin, DSpark cannot serve this recipe's own
deep-context workload at any knob setting. Reopening it requires more
memory per rank (a 64 KiB-page kernel, the display carve-out) or a
smaller *trained* drafter — engine/OS projects, not flags. The
zero-weight alternative (n-gram prompting off the request's own tokens)
is untested here and is the only spec-adjacent direction that does not
carry the drafter's residency.

## Comparison with other V4.1-Flash recipes

Every figure in the other rows is **other people's published number**,
read from the repository linked in that row — not measured here, and not
comparable cell for cell: the checkpoint, the weight format, the engine
and the transport all differ. They are here to show the shape of the
landscape, not to claim a win.

| Recipe | Engine | TP | Image | KV pool | Context |
|---|---|---|---|---|---|
| **This one** | vLLM (b12x fork), native weights | **3** | **public Docker Hub, digest-pinned** | **~5.1M tokens** | **1M** |
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
| `--kv-cache-memory-bytes` | `5g` | `0.5g` |
| `--gpu-memory-utilization` | `0.85` | `0.87` |
| `--max-num-seqs` | `8` | `2` |
| `--max-num-batched-tokens` | `8192` | `4096` |
| `--max-model-len` | `1048576` | `auto` |
| speculative decode | off | dspark, 7 tokens |
| `--load-format` | `b12x` (measured keep: ~32 s load; was `safetensors`) | `b12x` |
| linear / attention backend | unset — the engine's own policy | pinned `b12x` ×3 (this recipe pins the MoE backend too, `b12x` + the kill switch — see below) |

Two rows are the recipe's whole thesis, and they pull in opposite
directions from upstream's: upstream asks for **10× less KV cache** and **4×
fewer concurrent sequences**, with the context derived from that smaller
pool, which is a speed-shaped lane. This recipe asks for a **large pool and a pinned 1M
context**, which is a capacity-shaped lane, and accepts roughly 35 tok/s
for that.

Two rows are not a preference but a measured constraint on this hardware.
The loader row is wall 3 in the bring-up below; the 2026-11 re-measure
flipped it (the OOM was a pre-overlay posture, and the load-time win is
~6×) — `b12x` ships, `safetensors` stays documented as the fallback.
Leaving the dense and attention backends unset is deliberate — the
engine's own policy is what gate 8 audits. The MoE backend is the one
exception: this recipe pins it (`b12x`, with the kill switch) because
the pin's safety condition — the switch — is itself asserted by the
same gate, so pinning removes nothing the gate exists to check.

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
| [#58720](https://github.com/vllm-project/vllm/pull/58720) indexed expert-mapping lookups | v0.32.0 | yes | −25 s weight load on DGX Spark; the load here takes ~32 s (b12x loader) |
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
requests moved after the pin, and the seven rows (eight pull requests)
marked `no` are the ones worth
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
- The christopherowen/dgx-spark-networking project for the optional
  one-shot transport's wiring (Apache-2.0; `launch/tp3-kit/sparknet/`)
- The upstream vLLM `safetensors` loader path, kept here as the
  conservative weight-loading fallback (wall 3)

## Changelog

- **2026-11** — decode-under-prefill share 0.2 shipped (FIXLOG failure
  mode 16); the 2026-11 kernel stack (~35 tok/s single-stream); the
  capacity overlay + 5 GB KV at 1M context; the b12x loader as the
  shipped default; the optional one-shot transport kit.
- **2026-10** — first publication: the TP3 bring-up, the six walls, and
  the acceptance-gate suite.

## License

MIT for the files original to this repository. The files under
`launch/tp3-kit/overlay/` are modified copies of Apache-2.0 vLLM
sources and remain Apache-2.0; the MIT grant does not cover them.
