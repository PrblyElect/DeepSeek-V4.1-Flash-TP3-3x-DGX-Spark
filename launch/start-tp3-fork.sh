#!/usr/bin/env bash
# start-tp3-fork.sh -- boot ONE box of a 3x DGX Spark TP=3 triangle for
# DeepSeek-V4.1-Flash on the B12X vLLM fork.
#
# WHY THE FORK: mainline vLLM is load-blocked on GB10 (sm_121a) for this
# model -- the load-stage OOM class the one-pot pinned-staging path hits.
# The community B12X lineage is the proven path for V4.1 on this hardware:
# the pinned nightly carries DeepseekV41ForCausalLM plus
# DSparkV41DraftModel, the B12X loader, and the runtime q-head padding
# (_pad_to_supported_q_heads in flashinfer_sparse).
#
# CHECKPOINT POSTURE: the NATURAL checkpoint is REQUIRED -- heads 64, vocab
# 129280, no padded drafter expert. The fork pads at runtime (the engine
# logs "heads 64 -> 72"), and the boot gate proves the model directory is
# the original pull before the engine starts. The B12X loader reads that
# checkpoint as it is: this path needs no vocabulary tensor pad.
# Engram: ON, DISK-RESIDENT -- on GB10 (compute capability 12.x) the model
# dispatches to vllm.models.deepseek_v4_1, whose engram reads the tables
# (~63 GiB per rank on NVMe) through the B12X native io_uring reader, two
# planes per access (256 B weight rows + 8 B scale rows). table_memory
# "disk" is consumed THERE. The "model loading" figure is the non-engram
# weights; the tables stream at runtime.
#
# SERVE LINE: gmu 0.85, kv-cache-memory-bytes 5g (~21,945 blocks; with
# the capacity overlay ~5.1M tokens), ctx 1048576 (1M; the reservation
# pins the pool, not ctx), parallel prefills
# x2 + decode-aware (a short request's first token beside a deep prefill
# drops from tens of seconds to single-digit seconds, at no measurable
# long-prefill cost), a static decode-under-prefill share of 0.2 plus a
# 2048/step contended-prefill cap (a short request DECODING beside that
# same prefill: ~1.4 -> ~25 tok/s; the price is ~+1 s on that first-token
# time; the measured ladder sits on the PREFILL_COMPUTE_SHARE comment),
# b12x kernel autotune OFF (heuristic plans ship as prebuilt
# cubins; boots in roughly half the tuned time, decode parity, a few
# percent slower one-time cold prefill), speculative decoding OFF --
# REJECTED: the drafter's ~3 GiB/rank residency OOMs the head on the
# first deep prefill at every knob-accessible pool.
#
# FABRIC: the RoCE pair and the management interface are site-specific.
# Set NCCL_IB_HCA and NCCL_SOCKET_IFNAME before booting; find the RoCE
# pair with `ibdev2netdev` on each box.
set -euo pipefail

# ---------------------------------------------------------------- image pin
# DIGEST-pinned to the published manifest of nightly-20261015 (the tag
# aliases it). A tag can move between deployments; a digest cannot.
#
# All three boxes must hold this manifest locally. A box whose image was
# imported rather than pulled can be content-identical yet carry no Hub
# RepoDigest; reconcile it with a manifest-only pull (the layers are
# already present by content digest, so nothing is re-downloaded).
#
# The image is also PATCHED at runtime by the bootstrap (the valid_object
# shim), so the digest alone cannot prove what executes. The engine-identity
# contract (tp3-kit/engine-identity.json) is enforced INSIDE the container
# by boot-gate-fork.py; the host side asserts the image CONTENT identity —
# the sha256 of the RootFS layer list, which is the invariant that is
# stable across pull and import.
IMAGE="${IMAGE:-eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595}"
EXPECT_IMAGE_LAYERS_SHA256="${EXPECT_IMAGE_LAYERS_SHA256:-${IMAGE_CONTENT_SHA256:-41e4f9a3005bca761eb883c6565c1a3667cfbb014d49ec3d97735c7cfe146616}}"
EXPECT_IMAGE_REPO_DIGEST="${EXPECT_IMAGE_REPO_DIGEST:-${IMAGE_DIGEST:-sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595}}"

image_layers_sha() {
  docker image inspect "$1" --format '{{range .RootFS.Layers}}{{println .}}{{end}}' 2>/dev/null \
    | sha256sum | cut -d' ' -f1
}

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[ds41-fork] IMAGE NOT PRESENT LOCALLY: $IMAGE" >&2
  echo "[ds41-fork]   reconcile it deliberately first (docker pull $IMAGE) —" >&2
  echo "[ds41-fork]   a boot window must not trigger a ~10 GB download." >&2
  exit 1
fi
LAYERS_SHA="$(image_layers_sha "$IMAGE" 2>/dev/null || true)"
if [ "$LAYERS_SHA" != "$EXPECT_IMAGE_LAYERS_SHA256" ]; then
  echo "[ds41-fork] IMAGE CONTENT MISMATCH" >&2
  echo "[ds41-fork]   expected layer-list sha256 $EXPECT_IMAGE_LAYERS_SHA256" >&2
  echo "[ds41-fork]   got                   ${LAYERS_SHA:-<none>}" >&2
  echo "[ds41-fork]   the engine build CHANGED — re-audit before serving it." >&2
  exit 1
fi
echo "[ds41-fork] image content OK (layer-list sha256 $LAYERS_SHA)"
REPO_DIGESTS="$(docker image inspect "$IMAGE" --format '{{json .RepoDigests}}' 2>/dev/null || true)"
case "$REPO_DIGESTS" in
  *"$EXPECT_IMAGE_REPO_DIGEST"*)
    echo "[ds41-fork] image repo digest OK ($EXPECT_IMAGE_REPO_DIGEST)" ;;
  "")
    echo "[ds41-fork] WARNING: image carries no Hub RepoDigest — content is pinned, provenance is not" >&2 ;;
  *)
    echo "[ds41-fork] IMAGE REPO DIGEST MISMATCH: $REPO_DIGESTS" >&2
    echo "[ds41-fork]   expected $EXPECT_IMAGE_REPO_DIGEST" >&2
    exit 1 ;;
esac

# ---------------------------------------------------------------- knobs
ROLE="${ROLE:-head}"
NODE_IP="${NODE_IP:?set NODE_IP to the cluster-facing IP of this box}"
HEAD_IP="${HEAD_IP:-$NODE_IP}"
CLUSTER_SIZE="${CLUSTER_SIZE:-3}"
PORT="${PORT:-8600}"
TP="${TP:-3}"
GMU="${GMU:-0.85}"                        # the eugr recipe's proven budget
CTX="${CTX:-1048576}"                     # 1M, measured and kept: pool bytes/blocks unchanged (kv-cache-memory-bytes
                                          # pins them); decode in band; deep-prefill admission proven to ~900K with
                                          # the trough well above the floor. The engine's token figure scales with
                                          # max_model_len; with the capacity overlay the figure is ~5.1M at both.
                                          # Quote BLOCKS as the invariant. Do NOT lower without a new boot result.
MNBT="${MNBT:-${MAX_NUM_BATCHED_TOKENS:-8192}}"   # chunked-prefill batch width. 16384 REJECTED (worker OOM during compile at gmu 0.85); 8192 is the proven posture. Retest only with a memory plan
LIMIT_MM="${LIMIT_MM:-{\"image\": 32}}"               # vision: images per prompt (0 = tower stubbed); each image caps at 1,024 tokens
# Parallel prefills — measured and kept ("2"): a short request's first token
# beside a deep prefill drops from tens of seconds to single-digit
# seconds, at no measurable cost to the long prefill (the shared chunk
# budget absorbs the small prefill in the big one's inter-chunk gaps).
# Where it lands inside that band is set by the decode-under-prefill share
# below (~4.6-4.8 s at shares 0.3-0.5, ~5.8 s at the shipped 0.2).
# "1" restores legacy single-lane behavior. The policy requires
# max-parallel-prefills > 1 (engine-side validation) and is only passed
# when the count is above 1.
PAR_PREFILLS="${PAR_PREFILLS:-2}"
PREFILL_POLICY="${PREFILL_POLICY:-decode-aware}"
# Decode-under-prefill share — KEEP (measured 2026-11): 0.2 static + 2048/step cap;
# see the bootstrap comment for the measured why. Empty ("") = legacy behavior:
# the ENGINE flag is then omitted (no --prefill-compute-share), though the
# container's ENV still carries an empty PREFILL_COMPUTE_SHARE.
PREFILL_COMPUTE_SHARE="${PREFILL_COMPUTE_SHARE-0.2}"
PREFILL_TOKENS_PER_STEP="${PREFILL_TOKENS_PER_STEP:-2048}"
# Row 13 (KEEP, 2026-11): the b12x loader routes checkpoint views through an
# asynchronous read ring (O_DIRECT bounce buffers on GB10 - read_mode=auto
# resolves to "bounce" because capabilities(0).host_page_tables == 1), and cut
# the model-load phase ~190 s -> ~32 s. Override with LOAD_FORMAT=safetensors
# to fall back to the default path.
LOAD_FORMAT="${LOAD_FORMAT:-b12x}"
# The engram default is assigned in TWO steps, never inline in ${VAR:-...}.
# A brace inside a parameter-expansion default makes bash terminate the
# expansion early once the variable IS set, which silently appended a stray
# "}" to the JSON — the value was only correct when nothing set it. A
# single-quoted literal has no escapes and no scanner ambiguity.
ENGRAM_CONFIG="${ENGRAM_CONFIG:-}"
if [ -z "$ENGRAM_CONFIG" ]; then
  # a disk-resident scale plane TESTED AND REJECTED: the plane fits and
  # the mechanism works, but prefill/decode measured IDENTICAL to the streamed
  # path (5.1K vs 5.2K tok/s prefill, 28.9 vs 29.3 decode) — engram I/O is not
  # a wall-time factor here, and the plane pins 1.91 GiB/rank of host headroom
  # serving wants back. Retest only with a workload that is engram-I/O-bound.
  ENGRAM_CONFIG='{"cpu_offload":false,"table_memory":"disk"}'
fi
# DSpark speculative decoding — empty default = OFF. REJECTED, FINAL on
# this pin (see README): the drafter's residency OOMs the head on the
# first deep prefill at every knob-accessible pool — engine/OS memory
# projects reopen it, not this knob. If ever attempted despite that, the
# JSON must
# stay COMPACT — it crosses an unquoted word-split expansion in the
# bootstrap, so any space breaks it into garbage args.
SPEC_CFG="${SPEC_CFG:-}"
# 5g, measured and kept (provisional on the trough soak): ~21,945 blocks,
# ~5.1M tokens / ~4.9x at 1M with the capacity overlay; the 7x512K
# verification ran at a ~1.7 GB trough (floor 1.6). Drop to 4g if
# in-service troughs breach the floor (one env; a 24h soak sampler
# arbitrates).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-5g}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# The served name: the profile's knob wins, and ALIAS_NAME stays a working
# alias when the profile does not set one. Resolved ONCE here, because a
# two-variable fallback inside the serve line lets one of them shadow the
# other for good. ALIAS_NAME is read BEFORE its own default is applied.
ALIAS_NAME="${ALIAS_NAME:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${ALIAS_NAME:-main}}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v41}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_v41}"
DIST_MASTER_PORT="${DIST_MASTER_PORT:-25000}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/deepseek-v4.1-flash-1017-1bae0e10}"
CONTAINER_NAME="${CONTAINER_NAME:-ds41-flash-tp3}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"
# Fabric selectors — site-specific; fill them in before booting. The guard
# below refuses a boot while a placeholder is still in place.
NCCL_IB_HCA="${NCCL_IB_HCA:-EDIT_ME_ROCE_PORT_PAIR}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-EDIT_ME_MGMT_IF}"
SECURITY_OPTS="${SECURITY_OPTS:---security-opt seccomp=unconfined}"
# The NCCL caps and the allocator setting are knobs too — the shipped
# values are the proven posture, but a profile may want to change them.
NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
NCCL_LL128_BUFFSIZE="${NCCL_LL128_BUFFSIZE:-262144}"
NCCL_PROTO="${NCCL_PROTO:-^LL128}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
# b12x autotuner switch — DEFAULT OFF ("0"), measured and kept (an
# A/B boot): defaults are prebuilt cubins (0 compilations, 0 measured),
# engine time ~6.6 min vs ~12-15 tuned, decode 29.5 tok/s (at/above the
# tuned 28.8-29.3 band), determinism/vision/relay PASS. The arena
# self-budgets at HALF OF FREE MEMORY (_race_budget) and on these boxes the
# broken MemAvailable means it never self-limits — it was the elastic
# transient that kernel-OOM-killed early boots. Set "=1" to
# restore per-shape benchmarking only with a specific reason.
# b12x MoE tiny-decode kill switch — CORRECTNESS-CRITICAL, default 0:
# the tiny-decode path checks num_tokens/k/n/activation but NOT
# swiglu_limit, so with it enabled ("1") this model's clamped-SwiGLU
# limit is silently dropped on small-batch decode. 0 forces the general
# kernel, which receives the clamp through plan_weights. Only relevant
# when the MoE backend is b12x; harmless otherwise.
B12X_W4A8_TINY_DECODE="${B12X_W4A8_TINY_DECODE:-0}"
# MoE backend — DEFAULT b12x, measured and kept: ~31 tok/s
# solo, clean 4-stream parity, prefill at the top of its band, trough
# under 3 GB, zero memory cost — vs DeepGEMM's ~29 band. The kill
# switch above is REQUIRED with this backend (the general kernel carries
# the SwiGLU clamp; tiny-decode drops it). Set empty ("") to restore the
# engine's auto resolution (DEEPGEMM_MXFP4).
MOE_BACKEND="${MOE_BACKEND:-b12x}"
# Scheduler step-cap companion (env-only knob; no CLI form): when
# --max-num-scheduled-tokens caps a step budget, "1" leaves PREFILL-ONLY
# steps at the full MNBT (solo/uncontended prefill unchanged) while mixed
# prefill+decode steps take the cap — the decode-under-prefill lever.
# Default 1: with no step cap set, the value is inert.
VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS="${VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS:-1}"
B12X_AUTOTUNE="${B12X_AUTOTUNE:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
KIT_DIR="$(cd "$(dirname "$0")/tp3-kit" && pwd)"

[[ -d "$MODEL_DIR" ]] || { echo "model not staged: $MODEL_DIR" >&2; exit 1; }

# The fabric selectors are site-specific. Refuse to boot a triangle with
# the placeholders still in place — an unset RoCE pair fails much later,
# as a collective hang rather than a clear error.
case "${NCCL_IB_HCA:-}" in
  ""|EDIT_ME*) echo "set NCCL_IB_HCA to your RoCE port pair (ibdev2netdev)" >&2; exit 1 ;;
esac
case "${NCCL_SOCKET_IFNAME:-}" in
  ""|EDIT_ME*) echo "set NCCL_SOCKET_IFNAME to your management interface" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------- bootstrap
read -r -d '' BOOTSTRAP <<'EOF' || true
set -euo pipefail
# The host always forwards ENGRAM_CONFIG; this is the belt-and-braces
# default for a hand-run bootstrap. Assigned in two steps for the same
# reason as the host side: a brace inside a ${VAR:-...} default is not
# safe to expand.
if [ -z "${ENGRAM_CONFIG:-}" ]; then
  ENGRAM_CONFIG='{"cpu_offload":false,"table_memory":"disk"}'
fi
# nightly-20261015 ships a broken pair: preparation/artifacts.py imports
# valid_object from b12x._lib.cache_integrity, which no longer defines it.
# Patch it in-container before anything imports vllm: local shim, always
# "regenerate" — exactly first-boot semantics, correctness-neutral.
ART=/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/preparation/artifacts.py
if grep -q "valid_object" "$ART" && ! python3 -c "from b12x._lib.cache_integrity import valid_object" 2>/dev/null; then
  python3 - "$ART" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
src = src.replace(
    "from b12x._lib.cache_integrity import _fsync_directory, _mkdir_durable, valid_object",
    "from b12x._lib.cache_integrity import _fsync_directory, _mkdir_durable\n"
    "def valid_object(*a, **k):\n"
    "    return False  # shim: nightly-20261015 lacks the symbol; regenerate")
p.write_text(src)
print("artifacts.py patched (valid_object shim)")
PY
fi
# b12x direct-routing threshold (2026-11, measured). The dynamic MoE takes
# DIRECT routing while routed_rows <= _DIRECT_ROUTING_MAX_ROUTED_ROWS (32):
# one full expert weight stream PER ROUTE, no union gather. Above 32 it
# gathers the UNION of the batch's experts and streams each once. At top-6 a
# 4-row verify is 24 routed rows -> direct, so the same expert is streamed
# twice (a measured arbitration prices that duplication at ~6.7 ms/step). Setting
# B12X_DIRECT_ROUTING_MAX_ROUTED_ROWS patches the constant in-container
# BEFORE anything imports the module. The patch is VERIFIED -- the old text
# must appear exactly once, or the boot fails loudly rather than silently
# running the incumbent. OPT-IN: unset (the default) means the incumbent
# behaviour, byte-identical.
if [ -n "${B12X_DIRECT_ROUTING_MAX_ROUTED_ROWS:-}" ]; then
  python3 - "$B12X_DIRECT_ROUTING_MAX_ROUTED_ROWS" <<'PY'
import pathlib, sys
want = int(sys.argv[1])
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/fused_moe/_impl.py")
src = p.read_text()
old = "_DIRECT_ROUTING_MAX_ROUTED_ROWS = 32"
assert src.count(old) == 1, f"expected exactly one {old!r}, found {src.count(old)}"
p.write_text(src.replace(old, f"_DIRECT_ROUTING_MAX_ROUTED_ROWS = {want}"))
print(f"b12x direct-routing threshold patched: 32 -> {want}")
PY
  echo "[ds41-fork] b12x union-routing patch installed (threshold $B12X_DIRECT_ROUTING_MAX_ROUTED_ROWS)"
fi
# KV-capacity accounting overlay (the steady/transient split, class of
# the open upstream PR #52495; measured):
# stock accounting charges every request its 15 sliding-window groups'
# peak (window + one whole batch per in-flight step); the overlay charges
# the window+lookahead per request and the unsettled batches once per
# pool. Installed BEFORE the boot gate, whose manifest pins the PATCHED
# hashes — a drifted overlay fails the gate loudly.
if [ -f /kit/overlay/vllm/v1/core/kv_cache_utils.py ]; then
  cp /kit/overlay/vllm/v1/core/kv_cache_utils.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
  cp /kit/overlay/vllm/v1/kv_cache_interface.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py
  # apply-time self-check: the named attributes
  # must exist in the PINNED image before the window is spent
  python3 - <<'OV'
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.config.vllm import VllmConfig
assert hasattr(SlidingWindowSpec, "max_capacity_blocks"), "overlay missing max_capacity_blocks"
assert hasattr(VllmConfig, "max_in_flight_tokens"), "VllmConfig.max_in_flight_tokens absent"
assert hasattr(VllmConfig, "num_lookahead_tokens"), "VllmConfig.num_lookahead_tokens absent"
print("overlay self-check OK (max_capacity_blocks + in-flight/lookahead attrs)")
OV
  echo "[ds41-fork] KV-capacity overlay installed"
else
  echo "[ds41-fork] KV-capacity overlay ABSENT — boot gate will fail if the manifest expects it" >&2
fi
# sparknet one-shot communicator overlay (the transport layer): installed ONLY when the fabric
# is rendered. The stock file is unchanged for every other boot; the adapter
# import is inside the RoCE branch, so a non-sparknet boot never imports it.
if [ "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" = "1" ] && [ -n "${SPARKNET_ROCE_TOPOLOGY:-}" ]; then
  # the engine-side policy guard is absent, so the knob would silently kill
  # the transport. Refuse it here instead of booting a dead one-shot path.
  case " ${EXTRA_SERVE_ARGS:-} " in
    *" --disable-custom-all-reduce "*)
      echo "[ds41-fork] REFUSING: --disable-custom-all-reduce disables the sparknet one-shot transport (policy guard)" >&2
      exit 2 ;;
  esac
  if [ -f /kit/overlay/vllm/distributed/device_communicators/cuda_communicator.py ]; then
    cp /kit/overlay/vllm/distributed/device_communicators/cuda_communicator.py \
       /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py
    echo "[ds41-fork] sparknet communicator overlay installed"
  else
    echo "[ds41-fork] RoCE transport requested but the communicator overlay is ABSENT" >&2
    exit 1
  fi
  if [ -f /kit/overlay/vllm/v1/worker/gpu_worker.py ]; then
    cp /kit/overlay/vllm/v1/worker/gpu_worker.py \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py
    echo "[ds41-fork] sparknet gpu_worker freeze overlay installed"
  else
    echo "[ds41-fork] sparknet requested but the gpu_worker overlay is ABSENT" >&2
    exit 1
  fi
  # The sparknet runtime (pure-python, --no-deps) and this rank's rendered
  # routing envs (per-peer HCA map + subnets from the node map). The proxy
  # .so and kernels come from the pre-warmed cache mount; the transport kit README
  # builds them so a boot never compiles.
  # the cache/fence mounts key on the HOST env, this gate on the forwarded
  # topology — a topology set without the host env would install the overlay
  # but boot without the warmed cache and the NCCL fence. Close that hole:
  if [ -z "${SPARKNET_ROCE_CACHE_DIR:-}" ]; then
    echo "[ds41-fork] REFUSING: sparknet gate fired but no cache mount (host VLLM_ENABLE_ROCE_ALLREDUCE unset?) " >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1091
  . "/kit/sparknet/env-rank${NODE_RANK:-0}.sh"
  set +a
  python3 -m pip install --no-deps --no-index --quiet /kit/dgx_spark_networking-0.5.0-py3-none-any.whl \
    || { echo "[ds41-fork] sparknet wheel install FAILED" >&2; exit 1; }
  echo "[ds41-fork] sparknet wheel installed; rank ${NODE_RANK:-0} routing: topology=${SPARKNET_ROCE_TOPOLOGY:-unset} peers=${SPARKNET_ROCE_PEER_HCAS:-unset}"
  # NCCL fence gate: when the fenced .so is bind-mounted, the sha proves the
  # bytes and ncclGetVersion proves the ABI (23203 at version parity).
  if [ -n "${NCCL_FENCE_SHA:-}" ]; then
    GOT_SHA=$(sha256sum /usr/lib/aarch64-linux-gnu/libnccl.so.2.32.3 | cut -d" " -f1)
    if [ "$GOT_SHA" != "$NCCL_FENCE_SHA" ]; then
      echo "[ds41-fork] NCCL fence sha MISMATCH: $GOT_SHA != $NCCL_FENCE_SHA" >&2
      exit 1
    fi
    python3 - <<'PYNCCLGATE'
import ctypes
lib = ctypes.CDLL("libnccl.so.2")
v = ctypes.c_int(0)
rc = lib.ncclGetVersion(ctypes.byref(v))
assert rc == 0, rc
assert v.value == 23203, v.value
print("[ds41-fork] NCCL fence verified: sha match, ncclGetVersion 23203")
PYNCCLGATE
  fi
fi
python3 /kit/boot-gate-fork.py /model
# the LM-head kernel (2026-11): the LM head's MXFP8 GEMM backend. The generic FlashInfer
# MXFP8 linear path hard-codes backend="cutlass", and that CUTLASS kernel runs
# the LM head's M=1 GEMV at ~41% of the HBM roofline (~2.1 ms
# of MXFP8 weights). flashinfer's own b12x "warp-level MMA kernel with
# small-M decode tiles" runs the identical math at ~87% (~0.96 ms) and was
# BIT-IDENTICAL in the in-container bench (max abs diff 0). The overlay is
# installed ONLY when the knob is set, and only AFTER the boot gate has pinned
# the audited stock file -- so the standing posture is the untouched image and
# a new image cannot be silently downgraded by this file. Undo: leave the
# knob unset.
if [ -n "${VLLM_MXFP8_LINEAR_BACKEND:-}" ]; then
  LMH_OV=/kit/overlay/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py
  LMH_DST=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/mxfp8/flashinfer.py
  if [ ! -f "$LMH_OV" ]; then
    echo "[ds41-fork] VLLM_MXFP8_LINEAR_BACKEND is set but the LM-head overlay is ABSENT" >&2
    exit 1
  fi
  cp "$LMH_OV" "$LMH_DST"
  python3 - "$LMH_DST" <<'PYLMH'
import sys
src = open(sys.argv[1]).read()
assert 'os.environ.get("VLLM_MXFP8_LINEAR_BACKEND", "cutlass")' in src, \
    "overlay missing the backend knob"
assert 'backend="cutlass"' not in src, "overlay still hard-codes cutlass"
print("[ds41-fork] LM-head overlay self-check OK (backend knob present)")
PYLMH
  echo "[ds41-fork] LM-head MXFP8 backend overlay installed (backend=${VLLM_MXFP8_LINEAR_BACKEND})"
fi
if [[ "${ROLE:-head}" == "worker" ]]; then
  echo "[ds41-fork] worker rank ${NODE_RANK:?} joining ${HEAD_IP:?}:${DIST_MASTER_PORT:-25000}"
  HEADLESS_FLAG="--headless"
else
  echo "[ds41-fork] head serving (TCPStore ${NODE_IP}:${DIST_MASTER_PORT:-25000})"
  HEADLESS_FLAG=""
fi
# Spec decode flag, built here so the serve line never carries a
# conditional. Empty-default expansion only — a brace-bearing default
# would terminate early on the JSON (the ENGRAM_CONFIG lesson). The JSON
# is compact by contract, so the unquoted expansion word-splits into
# exactly two tokens: the flag and the config.
SPEC_FLAG=""
if [ -n "${SPEC_CFG:-}" ]; then
  SPEC_FLAG="--speculative-config ${SPEC_CFG}"
fi
# Parallel-prefill flags, built here so the serve line never carries a
# conditional (the SPEC_FLAG pattern). Only passed when the count is
# above the legacy single-lane default — the engine rejects the policy
# at a count of 1.
PREFILL_FLAGS=""
if [ "${PAR_PREFILLS:-1}" != "1" ]; then
  PREFILL_FLAGS="--max-parallel-prefills ${PAR_PREFILLS} --prefill-policy ${PREFILL_POLICY}"
fi
# Decode-under-prefill share (KEPT 2026-11, measured ladder): a
# short request decoding beside a deep prefill crawls at 1.4 tok/s on the
# default posture (the scheduler alternates PURE steps, so a mixed-step cap
# is a no-op — a measured boot refuted --max-num-scheduled-tokens). A STATIC
# prefill-compute-share + a 2048/step contended-prefill cap fixes it. 0.5 was
# the first keep: it lifted that decode from 1.4 to 8.2-9.6 tok/s (~6.5x) at
# ZERO idle cost (solo 26.1-28.9 tok/s, TTFT 0.46-0.73s vs the 29.2/0.5s
# baseline) and a +2-11% deep-prefill wall.
#
# 0.3 then moved it and the whole trade was measured pairwise:
# decode-under-prefill ~15.4 -> ~22.3 tok/s (+~45 %), short-request TTFT
# ~4.6 -> ~4.7 s (+~3 %), idle decode unchanged (~36), the deep prefill's
# wall FLAT (~87 -> ~86 s — the anticipated +27-31 % price did not
# appear), MemFree ~2 GB, 0 preemptions, and no capability change (a
# 14-point capability check passed unchanged).
#
# 0.2 then moved it again, measured against a concurrent control with
# the arrival schedule fixed, so the arrival-phase profile cancels
# instead of being averaged over:
# decode-under-prefill ~22.2 -> ~25.0 tok/s (+~13 %, the widened 95 %
# interval spanning roughly +11 to +14 %; an initial pass read the same
# effect at +~13 %), the prefill's charged contended seconds/pair ~7.5 ->
# ~6.2 and the split ~51 % -> ~46 % (the mechanism: less contended time
# charged to the prefill), the deep prefill's wall FLAT (~86 s in every
# arm), idle decode unchanged (~36), MemFree ~1.7-1.8 GB, 0 preemptions,
# and no capability change (the same 14-point check, passed). Those
# measurements could only BOUND the short-request TTFT price (point
# ~+0.4 s, ~+8 %; bounded at <= +25 % -- the ceiling the trade was
# measured against). Same-boot A/B comparisons (the engine's runtime
# share setter, so no cross-boot term exists by construction) settled
# it: TTFT +~7 % with the whole 95 % interval inside the +25 % ceiling,
# buying decode-under-prefill +~13 % -- reproduced on independent boots
# both times. NOTE the knob is the PREFILL
# fraction on this pin (0.2 = prefill 20 % / decode 80 %), so 0.7 would
# have made the contention WORSE, not better.
# "auto" is explicitly NOT the keep: it charged ~15% idle decode and
# 1.6-4.3s TTFT (boot the pre-change record). Set PREFILL_COMPUTE_SHARE="" to restore the
# legacy behavior.
if [ -n "${PREFILL_COMPUTE_SHARE:-}" ]; then
  PREFILL_FLAGS="${PREFILL_FLAGS} --prefill-compute-share ${PREFILL_COMPUTE_SHARE} --max-num-prefill-tokens-per-step ${PREFILL_TOKENS_PER_STEP:-2048}"
fi
MOE_FLAG=""
if [ -n "${MOE_BACKEND:-}" ]; then
  MOE_FLAG="--moe-backend ${MOE_BACKEND}"
fi
exec vllm serve /model \
  --load-format "${LOAD_FORMAT:-b12x}" \
  --engram-config "${ENGRAM_CONFIG}" \
  --tensor-parallel-size "${TP:-3}" \
  --distributed-executor-backend mp \
  --nnodes "${CLUSTER_SIZE:-3}" \
  --node-rank "${NODE_RANK:-0}" \
  --master-addr "${HEAD_IP}" \
  --master-port "${DIST_MASTER_PORT:-25000}" \
  ${HEADLESS_FLAG} \
  --max-model-len "${CTX}" \
  --gpu-memory-utilization "${GMU:-0.85}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-5g}" \
  --served-model-name "${SERVED_MODEL_NAME:-main}" \
  --limit-mm-per-prompt "${LIMIT_MM}" \
  --port "${PORT:-8600}" \
  --block-size "${BLOCK_SIZE:-256}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MNBT}" \
  ${SPEC_FLAG} \
  ${PREFILL_FLAGS} \
  ${MOE_FLAG} \
  --no-scheduler-reserve-full-isl \
  --no-enable-flashinfer-autotune \
  --no-enable-log-requests \
  --enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER:-deepseek_v41}" \
  --reasoning-parser "${REASONING_PARSER:-deepseek_v41}" \
  ${EXTRA_SERVE_ARGS:-}
EOF

# ---------------------------------------------------------------- boot probe
# PASSIVE — it never touches the container. It samples host-visible
# available RAM for the whole boot and keeps the MINIMUM: that minimum is
# the number that matters (it captures the post-weight-load state AND the
# compile-arena peak), and it is exactly what an available-RAM gate needs —
# the engine's own engram budget check runs at engine-config time, BEFORE
# the weights are resident, so it cannot see it. When the container exits,
# the probe records the boot outcome so the boot log gets one line instead
# of a discovery weeks later.
PROBE_OUT="${PROBE_OUT:-/tmp/ds41-boot-probe.txt}"
PROBE_INTERVAL="${PROBE_INTERVAL:-5}"
PROBE_MAX_SECONDS="${PROBE_MAX_SECONDS:-2700}"

# MemAvailable is NOT a usable signal on these boxes: it is reported as 0
# while MemFree is ~3 GiB, with no memory pressure, no cgroup limit and no
# dirty page cache. The engine's own engram budget check compares against
# that number, which is why it "cannot see" the headroom. So the probe
# measures what the kernel can actually HAND OUT:
#     reclaimable = MemFree + SReclaimable + (Active(file)+Inactive(file) - Unevictable)
# and keeps the minimum across the boot.
sample_memory() {
  awk '
    /^MemFree:/          {free=$2}
    /^MemAvailable:/     {avail=$2}
    /^SReclaimable:/     {srec=$2}
    /^Active\(file\):/    {af=$2}
    /^Inactive\(file\):/  {inf=$2}
    /^Unevictable:/      {unev=$2}
    END {
      rf = af + inf - unev; if (rf < 0) rf = 0;
      head = free + srec + rf;
      printf "%d %d %d\n", head, avail, free;
    }' /proc/meminfo 2>/dev/null
}

start_probe() {
  local start=$SECONDS min_kb=0 min_avail_kb=0 min_free_kb=0 seen=0 status="" mem="" head_kb="" avail_kb="" free_kb="" rest=""
  local have_min=0 have_avail=0 have_free=0
  local verdict="UNKNOWN" detail="container never reached a terminal state"
  : > "$PROBE_OUT"
  while [ $((SECONDS - start)) -lt "$PROBE_MAX_SECONDS" ]; do
    status="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
    case "$status" in
      exited|dead|removing) break ;;
    esac
    mem="$(sample_memory || true)"
    if [ -n "$mem" ]; then
      head_kb="${mem%% *}"; rest="${mem#* }"
      avail_kb="${rest%% *}"; free_kb="${rest##* }"
      if [ "$have_min" -eq 0 ] || [ "$head_kb" -lt "$min_kb" ]; then min_kb=$head_kb; have_min=1; fi
      if [ "$have_avail" -eq 0 ] || [ "$avail_kb" -lt "$min_avail_kb" ]; then min_avail_kb=$avail_kb; have_avail=1; fi
      if [ "$have_free" -eq 0 ] || [ "$free_kb" -lt "$min_free_kb" ]; then min_free_kb=$free_kb; have_free=1; fi
      seen=$((seen + 1))
      if [ $((seen % 12)) -eq 1 ]; then
        printf 'RAM-PROBE reclaimable_kb=%s min_reclaimable_kb=%s memavailable_kb=%s min_memavailable_kb=%s t=%ss\n' \
          "$head_kb" "$min_kb" "$avail_kb" "$min_avail_kb" "$((SECONDS - start))"
      fi
    fi
    sleep "$PROBE_INTERVAL"
  done
  if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    # NB: grep -c, not grep -q — with pipefail, `grep -q` exits early, docker
    # takes SIGPIPE and the pipeline reports failure for a successful match.
    # NB: 2>&1 — on this Docker build `docker logs` streams the container log
    # to STDERR, so without the merge the marker is invisible (measured:
    # hits=0 with 2>/dev/null, hits=1 with 2>&1).
    startup_hits="$(docker logs "$CONTAINER_NAME" 2>&1 | grep -c "Application startup complete" || true)"
    if [ "${startup_hits:-0}" -gt 0 ]; then
      verdict="OK"; detail="engine reached 'Application startup complete'"
    else
      verdict="FAIL"; detail="no 'Application startup complete' in the container log"
    fi
  fi
  # The headroom gate: the engram scale plane needs roughly 1.9 GiB per
  # rank of host memory. The gate compares that against the MINIMUM
  # reclaimable headroom observed across the whole boot. The KV pool is
  # deliberately NOT part of this trade -- a pool cut is a concurrency cut,
  # and the measurement says the headroom is there without it.
  local plane_kb="${PLANE_REQUIRED_MIB:-1960}"
  local headroom_mib=$(( min_kb / 1024 ))
  local plane_gate="n/a"
  if [ "$have_min" -eq 1 ]; then
    if [ "$min_kb" -ge $((plane_kb * 1024)) ]; then plane_gate="PASS"; else plane_gate="FAIL"; fi
  fi
  printf 'RAM-PROBE FINAL min_reclaimable_kb=%s min_reclaimable_mib=%s min_memavailable_kb=%s min_memfree_kb=%s samples=%s elapsed=%ss\n' \
    "$min_kb" "$headroom_mib" "$min_avail_kb" "$min_free_kb" "$seen" "$((SECONDS - start))"
  printf 'HEADROOM GATE: %s (min reclaimable %s MiB vs plane %s MiB)\n' "$plane_gate" "$headroom_mib" "$plane_kb"
  printf 'BOOT RESULT: %s (%s)\n' "$verdict" "$detail"
  {
    printf 'min_reclaimable_kb=%s\n' "$min_kb"
    printf 'min_reclaimable_mib=%s\n' "$headroom_mib"
    printf 'min_memavailable_kb=%s\n' "$min_avail_kb"
    printf 'min_memfree_kb=%s\n' "$min_free_kb"
    printf 'plane_required_mib=%s\n' "$plane_kb"
    printf 'headroom_gate=%s\n' "$plane_gate"
    printf 'samples=%s\n' "$seen"
    printf 'elapsed_s=%s\n' "$((SECONDS - start))"
    printf 'result=%s\n' "$verdict"
    printf 'detail=%s\n' "$detail"
  } > "$PROBE_OUT"
}

# ---------------------------------------------------------------- run
# DRY_RUN=1 runs every pre-boot assertion and stops — use it to validate a
# script change against a box without starting a container.
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[ds41-fork] DRY RUN — pre-boot assertions passed; container NOT started"
  exit 0
fi

# The container command is built ONCE, into an array. That is what makes the
# capture mode below honest: there is no second copy of the command that
# could drift away from the one that actually runs, which is the exact
# defect class this recipe has already been bitten by (knobs read but never
# forwarded). A reader can check that every value in the profile reached the
# container by reading the printed command, and that command IS the command.
# sparknet (the transport layer): the RDMA proxy .so and the TileLang one-shot kernels are
# compiled ONCE into a host-side cache dir and bind-mounted read-write — a
# cold compile inside a boot would stall the transport at startup (the
# cold-compile lesson), and a container-local cache would recompile on every boot.
SPARKNET_CACHE_ARGS=""
if [ "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" = "1" ] && [ -n "${SPARKNET_ROCE_TOPOLOGY:-}" ]; then
  SN_CACHE="$HOME/ds41-flash-tp3/sparknet-cache"
  mkdir -p "$SN_CACHE/roce" "$SN_CACHE/tilelang"
  SPARKNET_CACHE_ARGS="-v $SN_CACHE:/sparknet-cache -e SPARKNET_ROCE_CACHE_DIR=/sparknet-cache/roce -e TILELANG_CACHE_DIR=/sparknet-cache/tilelang"
  echo "[ds41-fork] sparknet cache mount: $SN_CACHE"
  # The fenced NCCL (2.32.3 + 0001) rides in on the same cache mount when
  # staged: a read-only bind over the one real file BOTH resolution routes
  # load (the soname symlink resolves here). Rollback =
  # restart without the file. The sha gate in the bootstrap proves the
  # mounted bytes are the built fence, not the stock image .so.
  if [ -f "$SN_CACHE/nccl/libnccl.so.2.32.3" ] && [ -f "$SN_CACHE/nccl/libnccl.so.2.32.3.sha256" ]; then
    FENCE_SHA=$(cut -d" " -f1 "$SN_CACHE/nccl/libnccl.so.2.32.3.sha256")
    SPARKNET_CACHE_ARGS="$SPARKNET_CACHE_ARGS -v $SN_CACHE/nccl/libnccl.so.2.32.3:/usr/lib/aarch64-linux-gnu/libnccl.so.2.32.3:ro -e NCCL_FENCE_SHA=$FENCE_SHA"
    echo "[ds41-fork] NCCL fence mount armed (sha ${FENCE_SHA:0:16})"
  fi
fi
# persistent b12x compile cache: /root/.cache/b12x lives in the
# container's writable layer, so every recreate throws the compiled MoE
# kernels away and each boot pays the compile again inside the model-load /
# init-engine phase (the cold arm of the measured cold/warm pair). This is
# the same pattern as the sparknet cache above: one host directory, one
# bind mount, one env knob to turn it off.
#
# STALE-CACHE SAFETY (the measured note): b12x keys its compile
# entries by CONTENT HASH -- sha256-named .o/.json pairs under compile/ -- so
# a different engine or image yields cache MISSES, never a wrong kernel. The
# host directory is additionally keyed by the image repo digest, so each
# image keeps its own cache and eviction is obvious rather than silent.
# The cache is advisory: a corrupt entry is a compile error, not a bad
# kernel, and removing the directory is always safe.
B12X_CACHE_ARGS=""
if [ "${B12X_PERSISTENT_CACHE:-1}" = "1" ]; then
  B12X_CACHE_KEY="$(printf '%s' "${EXPECT_IMAGE_REPO_DIGEST:-unknown}" | sed 's/^sha256://; s/[^0-9a-f]//g' | cut -c1-12)"
  B12X_CACHE="$HOME/ds41-flash-tp3/b12x-cache/$B12X_CACHE_KEY"
  mkdir -p "$B12X_CACHE"
  B12X_CACHE_ARGS="-v $B12X_CACHE:/root/.cache/b12x"
  echo "[ds41-fork] b12x cache mount: $B12X_CACHE"
fi
run_args=(
  --name "$CONTAINER_NAME"
  ${SECURITY_OPTS}
  --network host
  --gpus all
  --device /dev/infiniband
  --ipc host
  --ulimit stack=67108864
  --ulimit memlock=-1
  --cap-add IPC_LOCK
  -e NODE_RANK="${NODE_RANK:-0}"
  -e DIST_MASTER_PORT="$DIST_MASTER_PORT"
  -e ROLE="$ROLE" -e NODE_IP="$NODE_IP" -e HEAD_IP="$HEAD_IP"
  -e CLUSTER_SIZE="$CLUSTER_SIZE"
  -e VLLM_HOST_IP="$NODE_IP"
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,COLL
  -e NCCL_DEBUG_FILE="/tmp/nccl-rank${NODE_RANK:-0}.log"
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
  -e NCCL_IB_HCA="${NCCL_IB_HCA:?set NCCL_IB_HCA to your RoCE port pair}"
  -e NCCL_CROSS_NIC=1
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=24
  -e NCCL_IB_GID_AUTO=1 -e NCCL_IB_MERGE_NICS=0
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:?set NCCL_SOCKET_IFNAME to your management interface}"
  -e LOAD_FORMAT="$LOAD_FORMAT"
  -e ENGRAM_CONFIG="$ENGRAM_CONFIG"
  -e KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES"
  -e BLOCK_SIZE="$BLOCK_SIZE"
  -e MAX_NUM_SEQS="$MAX_NUM_SEQS"
  -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS"
  -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
  -e TOOL_CALL_PARSER="$TOOL_CALL_PARSER"
  -e REASONING_PARSER="$REASONING_PARSER"
  -e TP_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" -e GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS"
  -e NCCL_BUFFSIZE="$NCCL_BUFFSIZE" -e NCCL_LL128_BUFFSIZE="$NCCL_LL128_BUFFSIZE" -e "NCCL_PROTO=$NCCL_PROTO"
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_PLUGIN=none
  -e VLLM_ENABLE_ROCE_ALLREDUCE=1
  -e VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB
  -e SPARKNET_ROCE_TOPOLOGY="${SPARKNET_ROCE_TOPOLOGY:-}"
  ${SPARKNET_CACHE_ARGS}
  ${B12X_CACHE_ARGS}
  -e B12X_COMPILE_WORKERS=4
  -e CUTE_DSL_ARCH=sm_121a
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
  -e VLLM_MXFP8_LM_HEAD=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
  -e B12X_AUTOTUNE="$B12X_AUTOTUNE"
  -e B12X_W4A8_TINY_DECODE="$B12X_W4A8_TINY_DECODE"
  # the b12x dynamic kernel's work source (2026-11). "materialized_queue" is
  # the production default and is the value in force previously: it
  # takes DIRECT routing while routed_rows <= 32 (_impl.py:2259), i.e. one full
  # expert weight stream PER ROUTE, so a multi-row verify streams the same
  # expert twice. "ready_queue" disables the direct-routing predicate
  # (_impl.py:1992) and the routing falls back to GROUPED -- the union of the
  # batch's experts, streamed once each. Default below = the incumbent value,
  # so a boot without the variable reproduces today's behaviour exactly and the
  # arm is undone by removing the variable.
  -e B12X_DYNAMIC_WORK_SOURCE="${B12X_DYNAMIC_WORK_SOURCE:-materialized_queue}"
  # the b12x direct-routing threshold: the bootstrap patches the
  # library's _DIRECT_ROUTING_MAX_ROUTED_ROWS in-container when this is set.
  # Empty (the default) = no patch = the incumbent routing, byte-identical.
  -e B12X_DIRECT_ROUTING_MAX_ROUTED_ROWS="${B12X_DIRECT_ROUTING_MAX_ROUTED_ROWS:-}"
  -e VLLM_DS41_L2_PREFETCH="${VLLM_DS41_L2_PREFETCH:-1}"
  # the budget sweep (2026-11): the per-window L2-prefetch fill budgets. GB10's L2 is
  # 24 MB and one layer's live working set is the WO + FFN + NEXT windows
  # together, so the shipped defaults (20+16+20 = 56 MB) stage 2.3x what the
  # cache can hold. These knobs are the recipe's way to size that working set;
  # the defaults below are the values in force before this window, so the
  # plumbing alone changes nothing.
  -e VLLM_DS41_L2_PREFETCH_WO_MB="${VLLM_DS41_L2_PREFETCH_WO_MB:-20}"
  -e VLLM_DS41_L2_PREFETCH_FFN_MB="${VLLM_DS41_L2_PREFETCH_FFN_MB:-16}"
  -e VLLM_DS41_L2_PREFETCH_NEXT_MB="${VLLM_DS41_L2_PREFETCH_NEXT_MB:-20}"
  # the L2 set-aside (2026-11): the prefetcher issues its bulk-L2 fills with an
  # `evict_last` policy, which only protects lines inside the CUDA
  # persisting-L2 set-aside. That set-aside is 0 by default, so the
  # protection the prefetcher asks for is inert. GB10's L2 is 24 MB and the
  # device allows an 18 MB set-aside; this knob sizes it. The default below
  # is 0 = the driver limit untouched = the behaviour in force before this
  # window, so the plumbing alone changes nothing. NOTE the env name is the
  # GLM module's: the DS41 prefetcher reuses the GLM one.
  -e VLLM_GLM53_L2_PREFETCH_PERSIST_MB="${VLLM_GLM53_L2_PREFETCH_PERSIST_MB:-12}"
  # the grid sweep (2026-11): the issuing-CTA count of the bulk-L2 fill. The GLM
  # module's default is 4 on GB10 (SM121) -- a value its own docstring records
  # as "measured best" at TP4 with an EIGHT-ROW verifier. We are TP3 with one
  # live token per step, so the knob is shape-sensitive and has never been set
  # here. The default below is the value in force before this window, so the
  # plumbing alone changes nothing. Read at import time by the module.
  -e VLLM_L2_PREFETCH_GRID="${VLLM_L2_PREFETCH_GRID:-4}"
  # the LM-head kernel (2026-11): the backend the generic FlashInfer MXFP8 linear kernel
  # asks for. The stock hard-coded "cutlass" ran the LM head's M=1 GEMV at
  # ~41% of the HBM roofline; flashinfer's own b12x
  # "small-M decode tiles" kernel runs the identical math at ~87%
  # and is BIT-IDENTICAL (bit-identical output across a bench sweep). LANDED:
  # worth ~+6% single-stream, determinism gate
  # PASS. The default below is "auto"; set it to "cutlass" to restore the
  # stock kernel. The knob also gates the kit overlay that makes the argument
  # selectable at all.
  -e VLLM_MXFP8_LINEAR_BACKEND="${VLLM_MXFP8_LINEAR_BACKEND:-auto}"
  -e VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS="$VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS"
  -e MOE_BACKEND="$MOE_BACKEND"
  # AV's profile context len is the drafter-decision lever; the fixed -e list previously
  # dropped it, so a pre-flight that set it would silently run the 8192 default.
  -e VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN="${AV_PROFILE_CTX:-8192}"
  -e PYTHONPYCACHEPREFIX=/tmp/pycache
  -e RUNAI_STREAMER_MEMORY_LIMIT=4294967296
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  -e PORT="$PORT" -e TP="$TP" -e GMU="$GMU" -e CTX="$CTX" -e MNBT="$MNBT" -e LIMIT_MM="$LIMIT_MM" -e SPEC_CFG="$SPEC_CFG"
  -e PAR_PREFILLS="$PAR_PREFILLS" -e PREFILL_POLICY="$PREFILL_POLICY"
  -e PREFILL_COMPUTE_SHARE="$PREFILL_COMPUTE_SHARE" -e PREFILL_TOKENS_PER_STEP="$PREFILL_TOKENS_PER_STEP"
  -e MALLOC_ARENA_MAX=2
  -e EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS"
  # The engine cost grid's instrument (servebench's engine_grid) lives
  # behind the dev-mode collective RPC; it is used only while measuring
  # an engine cost grid. Off by default.
  -e VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-0}"
  -v "$MODEL_DIR:/model:ro"
  -v "$KIT_DIR:/kit:ro"
  --entrypoint /bin/bash
  "$IMAGE" -euo pipefail -c "$BOOTSTRAP"
)

# PRINT_CMD=1 prints the container command and stops. No container is
# started, no GPU is touched, and the recipe's own pre-boot assertions still
# run first — so the printed command is one that would actually have been
# accepted. This is how a reader checks that every value in the profile
# reached the container.
if [ "${PRINT_CMD:-0}" = "1" ]; then
  echo "[ds41-fork] PRINT_CMD — the container command, one argument per line:"
  printf '%s\n' docker run
  printf '%s\n' "${run_args[@]}"
  echo
  echo "[ds41-fork] PRINT_CMD — the same command as one line:"
  printf '%q ' docker run "${run_args[@]}"
  printf '\n'
  echo "[ds41-fork] PRINT_CMD — container NOT started"
  exit 0
fi

# Best-effort page-cache drop before the load (the startup free-memory
# wall's own fix): trims the load trough on unified memory. Non-fatal
# when passwordless sudo is unavailable — the manual command is wall 4.
if sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
  echo "[ds41-fork] page cache dropped pre-boot"
else
  echo "[ds41-fork] drop_caches SKIPPED (no passwordless sudo) — run wall 4's command manually before booting"
fi

start_probe &
exec docker run "${run_args[@]}"
