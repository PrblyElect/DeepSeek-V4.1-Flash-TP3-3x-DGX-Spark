# One-shot RoCE transport (optional layer)

This kit switches the TP all-reduce from NCCL/PYNCCL to **sparknet's one-shot
RoCE** collectives — the same wire protocol the image's b12x lineage vendored,
plus a per-peer HCA map that fits a **one-cable-per-pair** triangle (the
image's built-in RoCEnante collective needs two rails per peer pair and
silently falls back to PYNCCL on this cabling).

Measured effect on this recipe's posture (banded): **~+5% single-stream
decode, +12–15% two-stream aggregate**, no memory cost, pool unchanged.
Prefill collectives above the dispatch limit still ride NCCL by policy.

## What to edit for your site

1. **`nodes.template.json`** — replace the placeholder hostnames, management
   IPs and per-pair /24s with your fabric's real values. `rank` must match
   each box's `NODE_RANK` (head = 0). Validate with
   `sparknet topology validate nodes.json --transport oneshot-direct`.
2. **`env-rank{0,1,2}.sh`** — per-rank routing fragments the launcher sources
   inside the container. The templates use documentation ranges
   (`10.10.x.0/24`); set each rank's **peer→local-HCA map** and **HCA→subnet
   map** from your rendered node map
   (`sparknet topology render nodes.json <node> --transport oneshot-direct`
   prints the exact exports).
3. **The wheel** — build once from the public tag (Apache-2.0):
   `python3 -m pip wheel --no-deps "dgx-spark-networking @ git+https://github.com/christopherowen/dgx-spark-networking@v0.5.0"`
   and drop it at `tp3-kit/dgx_spark_networking-0.5.0-py3-none-any.whl`.
   Install is `--no-deps --no-index` (the image already ships torch,
   cuda-python and TileLang 0.1.12, which the compile gate qualifies).

## The GID-index trap (measured across boxes of differing firmware)

sparknet resolves each HCA's IPv4 RoCE v2 GID at a fixed index (default 3).
On firmware families where **index 3 is all-zero** (observed on one box of
three; its IPv4 GID lives at index 2 only), queue-pair connect fails with
`ibv_modify_qp(RTR): Invalid argument / No data available` — fail-stop, at
communicator construction. Before your first boot, read each port's table:

```sh
cat /sys/class/infiniband/<hca>/ports/1/gids/{0,1,2,3}
```

If any port has its IPv4 GID (`0000:...:ffff:xxxx:xxxx`) only at index 2,
export **`SPARKNET_ROCE_GID_INDEX=2`** in all three env-rank files (index 2
carried the IPv4 GID on every box measured). The env templates here already
set it.

## The NCCL fence (required under prefill load)

The image's NCCL (2.32.3) has an IB send-path ordering bug (upstream
NVIDIA/nccl#2393, closed unmerged): `ncclIbIsend` reads `slots[0].nreqs`
after the index check without an acquire fence — under the prefill-sized
all-reduces that remain on NCCL (the transport only takes the small ones)
this can hang. Build **2.32.3 + the 2-line fence patch at version parity**
(not a downgrade), from the upstream sparknet repo (`sparknet/nccl/patches/0001-*`),
with the image's own toolchain (`make src.build
NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`), then stage the
resulting `.so` at `sparknet-cache/nccl/libnccl.so.2.32.3` + a `.sha256`
sidecar. The launcher bind-mounts it read-only over the system path (the one
real file both the soname symlink and the wheel symlink resolve to), and the
bootstrap gates on sha equality + `ncclGetVersion()==23203` before serving.
Rollback = restart without the staged file.

## Warm caches (a boot must not compile)

Pre-warm once per box, caged, with the lane parked or idle:

- the **C proxy**: sparknet compiles `_roce_proxy.c` once per source hash
  into `SPARKNET_ROCE_CACHE_DIR` (needs gcc + libibverbs headers, both in
  the image);
- the **TileLang kernels**: the adapter prepares fp16/bf16/fp32 all-reduce
  and all-gather at construction — compile them once
  (`_oneshot_tilelang._build` / `_allgather_tilelang._build` per dtype) into
  `TILELANG_CACHE_DIR`.

The launcher bind-mounts `sparknet-cache/{roce,tilelang}` so every boot
reuses them, and the worker overlay freezes kernel resolution after warm-up:
a late compile raises instead of stalling a serving step.

## Enable / disable

On: `VLLM_ENABLE_ROCE_ALLREDUCE=1 SPARKNET_ROCE_TOPOLOGY=direct` in the
environment of every `start-tp3-fork.sh` invocation (the launcher refuses a
half-armed boot: the gate fires only when both are present). The engine log
must read `Using ['SPARKNET_ONESHOT', 'PYNCCL'] ... for group 'tp:0'`.

Arming the transport installs its two overlays BEFORE the boot gate, so the
armed boot differs from the shipped default in three manifest places —
switch all three (the launcher's wheel/routing steps tell you when):

1. `resolved_backends.expect.allreduce_backend_by_group` =
   `{"tp:0": "SPARKNET_ONESHOT,PYNCCL", "ep:0": "PYNCCL"}` (delete the
   scalar `allreduce_backend`);
2. the `vllm/distributed/device_communicators/cuda_communicator.py` pin →
   the sha256 of `tp3-kit/overlay/vllm/distributed/device_communicators/cuda_communicator.py`;
3. the `vllm/v1/worker/gpu_worker.py` pin → the sha256 of
   `tp3-kit/overlay/vllm/v1/worker/gpu_worker.py`.

Off: run without those two envs — the stock communicator file is untouched,
the transport stays PYNCCL, and nothing else changes.
