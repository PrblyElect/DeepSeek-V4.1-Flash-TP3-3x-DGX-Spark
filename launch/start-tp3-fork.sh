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
# CHECKPOINT POSTURE: the 4-tensor vocab pad (129408) and the head pads
# (72) are REQUIRED -- the fork's vocab range still divides raw, verified
# in-image, while the B12X loader reads the standard HF layout.
# Engram: ON, DISK-RESIDENT -- on GB10 (compute capability 12.x) the model
# dispatches to vllm.models.deepseek_v4_1, whose engram reads the tables
# (~63 GiB per rank on NVMe) through the B12X native io_uring reader, two
# planes per access (256 B weight rows + 8 B scale rows). table_memory
# "disk" is consumed THERE. The "model loading" figure is the non-engram
# weights; the tables stream at runtime.
#
# SERVE LINE: gmu 0.85, kv-cache-memory-bytes 4g (the pool is boot-labile,
# roughly 1.6-2.4M tokens -- band it, do not quote it), ctx 524288,
# speculative decoding OFF (rejected on capacity: it OOM'd the worker).
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
# the sha256 of the RootFS layer list, which is the invariant that
# held across both reconciliations.
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
CTX="${CTX:-524288}"                      # 512K, validated; the KV pool is boot-labile (~1.6-2.4M tokens) — do NOT lower without a new boot result
MNBT="${MNBT:-${MAX_NUM_BATCHED_TOKENS:-8192}}"   # chunked-prefill batch width. 16384 REJECTED (worker OOM during compile at gmu 0.85); 8192 is the proven posture. Retest only with a memory plan
LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
# The engram default is assigned in TWO steps, never inline in ${VAR:-...}.
# A brace inside a parameter-expansion default makes bash terminate the
# expansion early once the variable IS set, which silently appended a stray
# "}" to the JSON — the value was only correct when nothing set it. A
# single-quoted literal has no escapes and no scanner ambiguity.
ENGRAM_CONFIG="${ENGRAM_CONFIG:-}"
if [ -z "$ENGRAM_CONFIG" ]; then
  ENGRAM_CONFIG='{"cpu_offload":false,"table_memory":"disk"}'
fi
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4g}"
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
SERVED_NAME="${SERVED_NAME:-deepseek-v4.1-flash}"
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
python3 /kit/boot-gate-fork.py /model
if [[ "${ROLE:-head}" == "worker" ]]; then
  echo "[ds41-fork] worker rank ${NODE_RANK:?} joining ${HEAD_IP:?}:${DIST_MASTER_PORT:-25000}"
  HEADLESS_FLAG="--headless"
else
  echo "[ds41-fork] head serving (TCPStore ${NODE_IP}:${DIST_MASTER_PORT:-25000})"
  HEADLESS_FLAG=""
fi
exec vllm serve /model \
  --load-format "${LOAD_FORMAT:-safetensors}" \
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
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-4g}" \
  --served-model-name "${SERVED_MODEL_NAME:-main}" \
  --port "${PORT:-8600}" \
  --block-size "${BLOCK_SIZE:-256}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  --max-num-batched-tokens "${MNBT}" \
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
  -e B12X_COMPILE_WORKERS=4
  -e CUTE_DSL_ARCH=sm_121a
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
  -e VLLM_MXFP8_LM_HEAD=1
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0
  -e PYTHONPYCACHEPREFIX=/tmp/pycache
  -e RUNAI_STREAMER_MEMORY_LIMIT=4294967296
  -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
  -e PORT="$PORT" -e TP="$TP" -e GMU="$GMU" -e CTX="$CTX" -e MNBT="$MNBT"
  -e SERVED_NAME="$SERVED_NAME" -e ALIAS_NAME="$ALIAS_NAME"
  -e EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS"
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

start_probe &
exec docker run "${run_args[@]}"
