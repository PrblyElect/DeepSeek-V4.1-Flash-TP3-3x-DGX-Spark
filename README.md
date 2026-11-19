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

# DeepSeek-V4.1-Flash at TP=3 across 3× NVIDIA DGX Spark, at 1M context

**TL;DR** — Serve DeepSeek-V4.1-Flash (552B total, native FP4+FP8 MoE with a
196B engram conditional-memory block) tensor-parallel across three DGX
Sparks on a 200G RoCE triangle, on a public digest-pinned Docker image.
The shipped **serving profile**: ~4.8 GiB KV at **1M context** (~4.85M-token
pool, whole-pool fill certified), **DSpark k=5** speculative decode,
**~74 tok/s single-stream (~2× the no-spec lane)**, ~225 tok/s at 8 streams,
**~305–346 tok/s at 16**, and **~1 s first token for a chat arriving beside
any deep prefill**. A cold boot loads weights in **~32 s**. The optional **4096-token deep-prefill layer** cuts deep-prefill
walls ~10% and ships in a **memory-safe paired form** (FIXLOG 27): cap
`MAX_NUM_BATCHED_TOKENS`, not the chunk, and the concurrent cold-prefill
memory transient halves while the solo speed stays.

(b12x is the kernel lineage of these images — it prefixes their env vars,
the MoE backend and the loader; "the b12x fork" below means this image
family.)

**Contents** — Quick start · Served numbers · The prefill step budget (the
memory lever) · The decode-step stack · Transport · The
six walls · Stack · Configuration · Re-audit triggers · Hardware · Engram ·
Vision · Speculative decode · Comparison · Upstream's 3-node recipe ·
Credits · License

**A published vLLM TP3 recipe on a publicly available image — three
commands to a serving lane, with acceptance gates.** The project that
publishes the image also ships its own
`recipes/3x-spark-cluster/deepseek-v4.1-flash.yaml` for their cluster
runner; the side-by-side is in **Upstream's own 3-node recipe** below, and
where they differ, that table is the reason this recipe reads the way it
does. For landscape context, the community SGLang recipe for this model
reads ~51 tok/s single-stream at 750K KV tokens and a quarter of this
context; a source-built community fork reaches a slightly larger pool on a
64 KiB-page kernel — see **Comparison**.

Prerequisites:

- three DGX Sparks wired as a full CX-7 RoCE triangle (three cables, one
  per pair, MTU 9000) plus a management LAN
- the ~510 GB checkpoint staged on every box
- the digest-pinned image staged on every box
- Docker with GPU support on every box

## Quick start

Edit `config/cluster.env` first: the three box names and IPs, the
checkpoint path (`MODEL_DIR` — the path **on the boxes**; the shipped
default is a placeholder), and the two fabric selectors (`NCCL_IB_HCA`
from `ibdev2netdev`, and `NCCL_SOCKET_IFNAME`). The preflight refuses to
run while the fabric selectors still hold placeholders.

```sh
# once per box: the image (digest-pinned; a tag can move, a digest cannot)
docker pull eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595

# once per box: the checkpoint (~510 GB) into MODEL_DIR
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e10 --local-dir "$MODEL_DIR"
```

The boot gate proves the checkpoint is a complete snapshot (48 shards at
the root, engram included; the 12 engram index entries). `hf download`
does not write the completion marker — add it yourself **after** the
download has fully completed, so the gate checks your honest claim:

```sh
echo '{"complete": true}' > "$MODEL_DIR/_DOWNLOAD_COMPLETE.json"
```

Then, from any box that can SSH the other two:

```sh
./launch/ds41-flash-tp3-up preflight    # image digest + content, checkpoint, fabric
./launch/ds41-flash-tp3-up stage-env     # copies cluster.env + launcher + tp3-kit
./launch/ds41-flash-tp3-up start --yes   # head first, then workers; API on :8600
```

`preflight` is read-only (never pulls) and checks, per box, the image's
manifest digest, the sha256 of its RootFS layer list, and the checkpoint
directory. `stage-env` copies `config/cluster.env`,
`launch/start-tp3-fork.sh` and the `launch/tp3-kit/` gate into
`~/ds41-flash-tp3/` on every box. The kit is the in-container
engine-identity gate: the image is patched at runtime by the bootstrap, so
the manifest digest alone cannot prove what executes — the gate asserts
the build stamp and the hashes of the engine-critical files before the
engine starts. A newer upstream nightly fails that gate until re-audited;
that is the intended adoption gate.

Two files are the load-bearing part and worth reading before you run
anything: `launch/start-tp3-fork.sh` (what boots a box) and
`launch/tp3-kit/engine-identity.json` (what the boot is allowed to
execute).

**First request** (the full acceptance suite is `docs/VALIDATION.md`):

```sh
curl http://HEAD_IP:8600/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "main", "messages": [{"role": "user", "content": "Reply with the single word: ready"}], "max_tokens": 8, "temperature": 0}'
```

Expect HTTP 200, a short completion, `finish_reason: "stop"`.

**Reading the container command without starting it:** `PRINT_CMD=1`
prints the `docker run` command the boot would run, one argument per line
plus a copy-pasteable single line, after the recipe's own pre-boot
assertions pass:

```sh
cd ~/ds41-flash-tp3
set -a; . config/cluster.env; set +a
NODE_IP=192.168.0.10 NODE_RANK=0 ROLE=head PRINT_CMD=1 bash start-tp3-fork.sh
```

This is how you check every value in `config/cluster.env` reached the
container — a knob that is read but never forwarded shows up as the
engine's default where you expected your value, a defect class this
recipe has already been bitten by (FIXLOG).

## Served numbers

One profile (gmu 0.85, 1M context, kernel autotune off, engram on NVMe).

**The serving profile** (~4.8 GiB KV + the capacity overlay,
DSpark k=5, one serialized prefill, the interactive-admission threshold,
`MAX_NUM_SEQS=8 or 16`, the host-hash kit):

| Metric | Value |
|---|---|
| Single-stream decode (counting-task) | **~74 tok/s** — about 2× the no-spec lane; code-class ~68 |
| Aggregate decode (counting-task, idle lane) | 1× **~74** · 4× ~160 · 8× **~225** · 16× **~305–346 tok/s** (record speed; zero preemptions, flat acceptance at 16; the sweep is not monotone — the 2-stream point dips below 1-stream) |
| Deep-prefill wall (cold ~350K, this hardware class) | high-80s of seconds at the shipped 2048 step; **~10% faster with the optional 4096 layer** (solo and beside traffic; short-prefill rates low-4K → mid-4K tok/s) |
| Host memory under concurrent cold prefills | the transient scales with the **step budget**, not the chunk — see the next section; the paired 4096 layer held worst-moment available memory in a **~0.6–0.8 GiB band** on both a fresh boot and a memory-drained lane, all fills completing — and the full pool-envelope fill (eight concurrent deep fills) completes end to end on the same layer |
| Arrival TTFT of a NEW short chat beside any deep prefill | **~1 s** (measured at 350K and 1M depths; without the threshold the arrival waits the whole prefill). A session resumed with a large cold context additionally pays its own prefill first — roughly 20–30 s per 100K resumed tokens at the published rates |
| Decode beside a deep prefill | **~30–40 tok/s** per stream (against ~70 solo; worst gap under a second) |
| KV reservation / pool | **~4.8 GiB** → **~4.85M tokens (~4.6× at 1M)** with the overlay; whole-pool fill certified clean (FIXLOG 22) |
| Engram drain-sync | deleted by the host-hash kit: ~8% → **~2%** of worker wall-time |
| Quality | clean across the acceptance gates (FIXLOG 21/22) |

(Figures are single-boot point estimates; boot-to-boot variance ≈2%. The
**capacity overlay** is a two-file patch the bootstrap installs before the
boot gate: stock KV accounting charges every request the worst-case span
of all its sliding-window groups; the overlay charges each request its
window+lookahead and the unsettled batches once — which is why the same
reservation reports a larger pool. FIXLOG failure mode 15.)

## The prefill step budget is the memory lever

The knob that controls the prefill **memory** transient is not the chunk
size — it is `MAX_NUM_BATCHED_TOKENS`, the scheduler's per-step token
budget. The engine fills that budget with whatever prefills are runnable:
two 4096-token chunks and four 2048-token chunks build the **same
8192-token step workspace** (a ~1–1.5 GiB class draw on host available
memory on this hardware, transient, in the first seconds of concurrent
cold prefills). Lowering the KV reservation does not touch it; on a
loaded box, that transient is what reaches kernel-OOM territory (FIXLOG
27).

The fix composes with the optional 4096 layer: cap the budget to 4096 and
a lone request's chunk still fills the whole step — **the layer's solo
speed is kept**, concurrent cold prefills share the smaller step, and the
transient roughly halves. Measured with four concurrent ~350K cold fills:
worst-moment available memory held a **~0.6–0.8 GiB band** on a fresh
boot, and again — same band, same completion count, same aggregate rate —
on a memory-drained lane immediately after. That is the shipped form of
the layer (the three commented lines in `config/cluster.env`).

The paired layer has since been pushed to the box's worst shapes: the
full pool-envelope fill — eight concurrent deep fills, the sequence cap
saturated — completed end to end with host available memory never
approaching the kernel-OOM line (an envelope run rides a few hundred MiB
for its duration; plan co-tenants accordingly); and a maximum-context
single request ran with an interactive stream decoding beside it the
whole time, holding sub-second worst gaps. Deep-fill walls at the 4096
step pay the documented threshold class (+31–41 %), and long-context
retrieval was re-verified at full depth on this geometry (3/3 at the
context limit).

Also budget-relevant: `MAX_NUM_BATCHED_TOKENS` is the knob whose
**raising** has killed worker ranks outright — 16384 OOM-killed both
workers during the b12x compile phase (FIXLOG 9), and any raise re-pays
kernel compilation on the next boot. Treat it as a memory knob first and
a speed knob second.

## The decode-step stack (all measured and kept, all one-line undo)

The single-stream figures are the b12x MoE backend plus
four additions, landed and A/B-measured one variable at a time:

1. **The LM-head kernel** — the engine hard-codes the `cutlass` MXFP8
   GEMV for linear heads; the image ships a b12x warp-MMA kernel ~2.1×
   faster at the decode shape with bit-identical output
   (`VLLM_MXFP8_LINEAR_BACKEND=auto`, +~6%).
2. **L2 behaviour** — the indexer's prefetcher stays ON (disabling was
   ~−5%); a 12 MB persisting-L2 set-aside adds ~1–2%
   (`VLLM_GLM53_L2_PREFETCH_PERSIST_MB=12`).
3. **The b12x loader** — a boot win: ~32 s loads (vs ~190 s safetensors).
4. **The persistent compile cache** — ~60 s off every cold start after
   the first.

Each is a launcher line or a kit overlay; removing the line restores the
prior behaviour. Keep `B12X_AUTOTUNE=0` and the FlashInfer autotune off:
the b12x tuning arena self-budgets at **half of free memory and
pre-allocates it**, so on this unified-memory kernel nothing fails to
allocate and everything else starves — a full-race boot wedged every box
at the SSH layer. With autotune off, plans fall back to validated
heuristic cubins at decode parity; the cost is a one-time cold-prefill
penalty of a few percent per new document (prefix-cache hits — the bulk
of real traffic — unaffected).

## Transport: NCCL today, one-shot optional

The stock TP all-reduce rides NCCL over the triangle, with buffer caps
that matter (wall 1). The image's own one-shot RoCE collective
(`B12X_ROCENANTE`) cannot form on this fabric — it demands both RoCE
rails of a link, and the supported three-cable triangle uses one cable
per pair (NVIDIA's cluster guidance: "only one cable per link"). The
engine logs the failed connect and falls through to `PYNCCL` silently;
the gate asserts the engaged transport so the fallback is never silent
(VALIDATION 8).

`tp3-kit/sparknet/` ships an **optional layer** (per-peer-HCA-map
one-shot transport, Apache-2.0 wiring, fenced-NCCL build recipe)
measured at **~+5% single-stream and +12–15% two-stream aggregate**, no
memory cost. **Every performance number in this README was measured with
the layer armed**; a stock boot runs the `PYNCCL` path and lands roughly
those deltas subtracted. The launcher refuses a half-armed boot. See
`launch/tp3-kit/sparknet/README.md` — including its GID-index trap
(`SPARKNET_ROCE_GID_INDEX=2` on boxes whose firmware carries the IPv4
GID at index 2 only) and the NCCL send-path fence patch it builds at
version parity.

## The bring-up: six walls and their fixes

Each is documented so you don't have to rediscover them. The full
symptom → root cause → fix table, including defects found in this
recipe's own scripts, is `docs/FIXLOG.md`.

### 1. NCCL connection buffers pin 4.7 GiB of host RAM

Default NCCL buffer sizing on a 3-node RoCE ring reserves ~4.7 GiB of
**pinned, unreclaimable** host memory per node. Fix (cut to 139 MiB):

```sh
NCCL_BUFFSIZE=1048576        # 1 MiB (default 4 MiB)
NCCL_LL128_BUFFSIZE=262144   # 256 KiB
NCCL_PROTO="^LL128"
NCCL_MAX_NCHANNELS=8
```

### 2. Engram tables stage ~31 GiB per rank in host memory

The 196B-parameter engram block ships two ~94.5 GiB table shards (~189
GiB; each rank reads its ~63 GiB share from local NVMe). Without explicit
configuration the loader stages ~31 GiB of rows into host memory beside
the 95 GiB of weights. The single setting that makes V4.1 bootable at TP3
on this hardware:

```sh
--engram-config '{"cpu_offload":false,"table_memory":"disk"}'
```

### 3. Weight loading: b12x ships (~32 s); safetensors is the fallback

The b12x O_DIRECT loader loads this checkpoint ~6× faster than
safetensors mmap with identical pool and decode; the historical
bring-up OOM (pinned bounce buffers adding 10–16 GiB during load) no
longer reproduces on this posture. On a smaller-memory configuration
that class of failure returns — the conservative fallback is
`--load-format safetensors` (kernel-evictable page cache, ~6× slower,
unconditionally safe).

### 4. The startup free-memory check

The fork checks available GPU memory at startup against gmu × total; a
busy head node fails the 0.92 default. gmu 0.85 ships, and the launcher
drops the page cache before every boot (best-effort, passwordless sudo;
manually: `sync; echo 3 > /proc/sys/vm/drop_caches`).

### 5. Vocabulary divisibility at TP3

The vocabulary (129,280) does not divide by 3. The fork's runtime TP3
padding handles attention heads, o_groups and drafter experts, so the
checkpoint is the natural pull, and the lane serves a single name
(`main`). Treat the single-served-name posture as the requirement — the
assertion that fired during bring-up is not present in the pinned build.

### 6. io_uring requires seccomp=unconfined in Docker

The b12x I/O layer uses io_uring, which Docker's default seccomp profile
blocks: `--security-opt seccomp=unconfined`.

## Stack

| Component | Pin |
|---|---|
| Image | `eugr/spark-vllm-b12x:nightly-20261015` (digest `sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (~510 GB) |
| vLLM | v0.1.dev21830+g8244d1c44 (fork build) |
| MoE backend | `B12X_MXFP4_MXFP8` (b12x, with the tiny-decode kill switch on) |
| Dense GEMM | FlashInferCutlassMxfp8LinearKernel |
| Model runner / KV layout / CUDA graphs | V2 / BLHNC / FULL_AND_PIECEWISE |

## Configuration

`config/cluster.env` is the launch environment. The load-bearing knobs:

```sh
# Cluster (EDIT THESE)
SPARK_HEAD="spark-head.local"; SPARK_A="spark-a.local"; SPARK_B="spark-b.local"
HEAD_IP="192.168.0.10"; WORKER_A_IP="192.168.0.11"; WORKER_B_IP="192.168.0.12"

# Fabric (EDIT THESE — the preflight refuses the placeholders)
NCCL_IB_HCA="EDIT_ME_ROCE_PORT_PAIR"      # from ibdev2netdev
NCCL_SOCKET_IFNAME="EDIT_ME_MGMT_IF"

# Image pin (the digest is what the preflight asserts)
IMAGE="eugr/spark-vllm-b12x:nightly-20261015"
IMAGE_DIGEST="sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"
IMAGE_CONTENT_SHA256="41e4f9a3005bca761eb883c6565c1a3667cfbb014d49ec3d97735c7cfe146616"

KV_CACHE_MEMORY_BYTES=4.8g         # the serving profile's ~4.8 GiB pin
GMU=0.85
CTX=1048576
SPEC_CFG='{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":3,"draft_sample_method":"greedy","rejection_sample_method":"block"}'
PAR_PREFILLS=1                     # serialized prefills keep decode batches pure (FIXLOG 19)
PREFILL_COMPUTE_SHARE=0.2          # decode-under-prefill share (FIXLOG 16)
PREFILL_TOKENS_PER_STEP=2048       # the contended-prefill cap paired with it
EXTRA_SERVE_ARGS='--long-prefill-token-threshold 1024'   # ~1 s arrivals (FIXLOG 21)
B12X_AUTOTUNE=0                    # heuristic cubin plans; see the decode-stack section

MAX_NUM_SEQS=8                     # 16 extends the serving aggregate ladder
MAX_NUM_BATCHED_TOKENS=8192        # the MEMORY lever of the prefill transient
                                   # (this section's pairing note + FIXLOG 27);
                                   # do NOT raise — 16384 OOM-killed both workers
                                   # in the compile phase (FIXLOG 9)
BLOCK_SIZE=256                     # the sparse-SWA backend's preferred size
LIMIT_MM='{"image": 32}'           # 0 = text-only (the ViT tower stubs)

LOAD_FORMAT=b12x                   # ~32 s load; safetensors = conservative fallback
ENGRAM_CONFIG='{"cpu_offload":false,"table_memory":"disk"}'
MOE_BACKEND=b12x                   # with B12X_W4A8_TINY_DECODE=0 (the clamp kill switch)
PORT=8600; TP=3; SERVED_MODEL_NAME=main   # single alias; wall 5

# NCCL (the critical caps — wall 1)
NCCL_BUFFSIZE=1048576; NCCL_LL128_BUFFSIZE=262144
NCCL_PROTO="^LL128"; NCCL_MAX_NCHANNELS=8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
                                                   # ~1.9 GB headroom (FIXLOG 8)
TOOL_CALL_PARSER=deepseek_v41; REASONING_PARSER=deepseek_v41
```

The optional 4096 layer is a commented block in the same file — switch it
whole. A historical no-spec capacity form (5 GB KV, parallel prefills)
and its receipts are recorded in FIXLOG 20–22; on the current stack the
serving profile's pool is within ~5% of it, so it is no longer shipped as
a profile.

**`BLOCK_SIZE=256` note:** it is the sparse-SWA backend's own preferred
size, pinned explicitly. The engine logs the scheduler LCM (256) while
`/metrics` reports 128 — the minimum across prefix-cacheable KV groups
(this model's sliding-window group is 128). Not a mistake; gate 8 checks
the requested value from the container's environment, the logged LCM, and
the metric label.

**Concurrency figures are pool depth**, not simultaneous streams:
`MAX_NUM_SEQS` caps live sequences; the ~4.6× figure describes how
many full-length requests fit the pool.

## Re-audit triggers

- **A new upstream image** — the engine-identity gate refuses any image
  whose build stamp or engine-critical hashes differ. Re-audit, then
  re-pin.
- **A rebase that touches TP=3** — upstream vLLM rejects TP=3 for this
  model outright (64 heads do not divide by 3); the head padding this
  recipe depends on is fork-local.
- **A 3-node TP mesh is an off-recommended shape** — upstream has a
  documented failure mode where ranks sit in a collective that never
  completes, no NCCL error, nodes idle (vLLM issue #46097, 4× Spark
  mesh). Worth knowing before debugging a hang.
- **Upstream moves worth watching** (not adopting): the DeepSeek-V4
  batch-invariance series is still open upstream — this recipe does not
  claim the property (gate 9 measures single-request repeat/cache
  invariance, not batch composition); the NVFP4-KV path and the
  ModelOpt quantisation path each change what the recipe **is** (a
  different KV layout; a different checkpoint) — both are re-audits, not
  upgrades.

## Hardware

Three NVIDIA DGX Spark (GB10, 121.69 GiB unified CPU/GPU memory each) as
a full CX-7 RoCE triangle (one /24 per pair, one cable per pair, 200G,
MTU 9000) plus a management LAN.

Per-node memory budget:

| Component | Size |
|---|---|
| Model weights | 95.4 GiB |
| KV cache | ~4.8 GB (≈4.5 GiB) |
| NCCL buffers (capped) | 0.14 GiB |
| CUDA context + framework | ~4 GiB |
| Compile/CUDA-graph arenas | ~2 GiB |
| Host OS + daemons | ~8 GiB |

This table is arithmetic, not a measurement — once the weights are
resident, the measured free memory on a loaded box can sit far below it
and moves with the page cache: observed readings range from tens of MiB
to a few GiB, and loads have been OOM-killed at the low end. Treat
headroom as a budget to measure: the launcher drops page cache before
the load, and its boot probe samples the minimum **reclaimable**
headroom across the whole boot and prints it (`MemAvailable` alone can
read zero on this unified-memory kernel even when memory is available).
If a boot is killed, the kernel log says so (`journalctl -k | grep -i
"killed process"`); the fix is freeing page cache before the load, not
lowering gmu. Under **serving** load, the transient that matters is the
prefill step budget — see its section.

## Engram on NVMe

The two ~94.5 GiB engram table shards stay on each node's local NVMe and
are read on demand (`table_memory:disk`) — the same posture as upstream's
own three-node recipe. Auto-prefetch is arithmetically unavailable for
this checkpoint: the engine's precondition is that the checkpoint fits
inside 90% of memory available at load time, and ~510 GB never fits
inside 90% of ~121 GiB. If a smaller checkpoint ever exists, that
arithmetic changes; the prefetch question does not.

One cost of disk tables is addressed by the shipped kit at
**`patches/engram-host-hash/`** (part of the serving profile,
`DS41_ENGRAM_HOST_HASH=1`): the stock path runs a GPU hash → D2H →
**event synchronize** every decode step (~8% of worker wall-time, growing
under memory pressure). The kit computes the same 24 row-ids per token on
the host — bit-identical to the engine's own reference oracle by
construction — and stages them straight into the pinned buffers (share
drops to ~2%), guarded by a per-boot self-check that falls back to the
native path on any mismatch.

## Vision: built in, one flag

The 1017 checkpoint carries a 32-layer ViT tower, and the fork ships the
matching image processor (constructed from the model config — no
`preprocessor_config.json`). The tower is built unconditionally; serving
it is one flag: `LIMIT_MM='{"image": 32}'` (0 = text-only, tower stubbed).
Verified live: exact shape/color/position recovery through the
OpenAI-compatible endpoint; a 12-image ordered identification 12/12 in
~4 s; a full 32-image batch in ~4 s (~185 tokens per image after
downsampling). Text decode speed and the KV pool unchanged. Two caveats:
vision requests emit a short run of invisible completion tokens before
the answer — give `max_tokens ≥ ~200`; and per-image identification stays
strong at the limit while aggregate counting drifts (a 32-image batch
counted as 30).

## Speculative decode: the shipped default

DSpark k=5 ships on the serving profile. The measured facts that shaped
it:

- **The drafter is cheap and correct** (~3 GiB/rank, loads in seconds);
  it buys **~2× single-stream for a smaller pool** — the early
  capacity rejection was a pool verdict, not a stability one (the gap
  has since closed — see Configuration).
- **The 8-stream ceiling was the scheduler, not the drafter**: with two
  parallel prefills the engine interleaves prefill chunks into decode
  steps and the spec path pays for the mixing — `PAR_PREFILLS=1` (one
  serialized prefill) moves 8-stream aggregate from ~140 to ~225 tok/s,
  single-stream unchanged. Raising `MAX_NUM_SEQS` to 16 extends the same
  curve to ~305–346 tok/s with the default capture ladder already
  covering the shapes.
- **The adaptive verifier is burst-only on this hardware**: its
  workspace is elastic and pins idle free memory near ~1.5 GB at every
  pool size tried. Kernel-side draft levers moved 8-stream only a few
  percent; per-batch depths + serialization compose but drop the
  deep-prefill trough under the floor, so the shipped profile is the
  plain k=5 drafter with one serialized prefill.
- **A middle tier exists (measured)**: a 3.0 GiB pool with k=3 reads
  ~2.7× at 1M and solo ~49–58 tok/s — strong at ≤4 streams (~131 tok/s
  aggregate), drops at 8. A niche option, documented in FIXLOG 21/22.

## Comparison with other V4.1-Flash recipes

Every figure in the other rows is **other people's published number**,
read from the linked repository — not measured here, and not comparable
cell for cell (checkpoint, weight format, engine and transport all
differ). The landscape's shape, not a claim to a win:

| Recipe | Engine | TP | Image | KV pool | Context |
|---|---|---|---|---|---|
| **This one** | vLLM (b12x fork), native weights | **3** | **public Docker Hub, digest-pinned** | **~4.85M** | **1M** |
| [MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) | SGLang | 3 (a 4-node line too) | SGLang base pinned by digest + local overlay | 750K (TP3) | 256K (TP3) |
| [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f) | vLLM (custom fork) | 3 or 4 | built from the repo | 1.3M–5.5M (TP3) – 11.6M (TP4) | 256K (TP3, 4k) / 1M (TP3, TP4) |
| [tonyd2wild](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks) | TensorFold (a Zig engine); older vLLM lanes | 2 or 4 | built from source | 1.05M (TP2) – 29M positions | 300K – 1M |

Two things this recipe does differently from all three: the engine image
is **public** — pull it by digest and the boot is reproducible — and the
shape is three nodes on a RoCE triangle rather than a ring of four.

## Upstream's own 3-node recipe

The image's project ships
`recipes/3x-spark-cluster/deepseek-v4.1-flash.yaml` — the same model,
image lineage and triangle, as a YAML for their cluster runner
(`cluster_only: true`, no acceptance gates of its own). Where the two
differ:

| knob | this recipe | upstream 3-node |
|---|---|---|
| `--kv-cache-memory-bytes` | `4.8g` | `0.5g` |
| `--gpu-memory-utilization` | `0.85` | `0.87` |
| `--max-num-seqs` | `8` (or `16` serving) | `2` |
| `--max-num-batched-tokens` | `8192` (`4096` with the paired layer) | `4096` |
| `--max-model-len` | `1048576` | `auto` |
| speculative decode | dspark k=5 | dspark, 7 tokens |
| `--load-format` | `b12x` | `b12x` |
| linear / attention backend | unset — the engine's policy, gated | pinned `b12x` ×3 |

Two rows are the thesis: upstream asks for ~10× less KV and 4× fewer
sequences — a speed-shaped lane; this recipe asks for a large pool and a
pinned 1M context, with the drafter kept on top of the
pool. Leaving the dense and attention backends unset is deliberate — the
engine's own policy is what gate 8 audits; the MoE backend is the one
exception (pinned `b12x` with the kill switch, whose safety condition the
same gate asserts).

## Credits

- The eugr/spark-vllm-docker project — the b12x images and the
  GB10-optimized kernels this recipe runs on
- DeepSeek — the model
- The MiaAI-Lab recipe — the NCCL buffer-cap fix (their documented
  4.7 GiB → 139 MiB change is what unlocked the head node), and the
  SGLang comparison numbers
- The christopherowen/dgx-spark-networking project — the one-shot
  transport's wiring (Apache-2.0; `launch/tp3-kit/sparknet/`)

## Changelog

- **2026-11 (hardening receipts)** — the paired 4096 layer proven to the
  box's worst shapes: the full pool-envelope fill (eight concurrent deep
  fills) end to end, a maximum-context request with interactive decode
  beside it, retrieval re-verified at the context limit; and the
  parallel-prefill/policy finding recorded (FIXLOG 28 — the policy pair
  is inert at a count of 1, and PAR=2 trades the multi-stream aggregate
  for decode-beside-prefill).
- **2026-11 (single profile)** — the no-spec capacity alternative
  removed: its historical raison d'être (the drafter costing half the
  pool) closed when the serving profile reached ~4.85M tokens; its
  receipts and the KV ceiling ladder stay in FIXLOG 20–22.
- **2026-11 (memory pass)** — the prefill step budget identified as the
  memory lever (FIXLOG 27); the optional 4096 layer ships in its
  memory-safe paired form; the README rebuilt around it.
- **2026-11** — the serving profile (4.8 GB KV + DSpark k=5 + the
  admission threshold + the host-hash kit) promoted to the shipped
  default; decode-under-prefill share 0.2; the capacity overlay + 5 GB
  KV at 1M; the b12x loader default; the optional one-shot transport;
  the mHC tile patch (`patches/ds41mhc-tile/`).
- **2026-10** — first publication: the TP3 bring-up, the six walls, and
  the acceptance-gate suite.

## License

MIT for the files original to this repository. The files under
`launch/tp3-kit/overlay/` are modified copies of Apache-2.0 vLLM sources
and remain Apache-2.0; the MIT grant does not cover them.
