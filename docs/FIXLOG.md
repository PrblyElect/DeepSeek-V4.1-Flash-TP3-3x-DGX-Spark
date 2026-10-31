# FIXLOG — ds41-flash-tp3 (V4.1-Flash on 3× DGX Spark, vLLM b12x)

Every failure mode encountered during bring-up, its root cause, and its
fix. Ordered by severity (boot-fatal → quality).

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Head node kernel-OOM during weight load; workers survive | NCCL connection buffers pin ~4.7 GiB unreclaimable host RAM per node; the head (carrying the API server and the usual daemons) has ~2 GiB less margin than workers | `NCCL_BUFFSIZE=1048576`, `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO="^LL128"` — cuts pinned to 139 MiB |
| 2 | Kernel-OOM ~7 s into load with only 1.4 GiB process RSS | Engram tables staged into host memory (~31 GiB/rank) on top of 95 GiB weights | `--engram-config '{"cpu_offload":false,"table_memory":"disk"}'` |
| 3 | Head node OOM at ~4 min into load (repeats identically across 4 boots) | The b12x O_DIRECT loader's pinned bounce buffers add 10–16 GiB unreclaimable host memory during weight loading | `--load-format safetensors` (kernel-evictable page cache instead of pinned buffers) |
| 4 | `ValueError: Free memory on device cuda:0 less than desired GPU memory utilization` | The fork checks free memory at startup against gmu × total; busy head node has less free than clean workers | gmu 0.85 (not the fork default 0.92); run `echo 3 > /proc/sys/vm/drop_caches` before boot |
| 5 | `io_uring bounce initialization failed: Operation not permitted` | Docker's default seccomp profile blocks io_uring syscalls used by the b12x I/O layer | `--security-opt seccomp=unconfined` |
| 6 | `AssertionError: 129280 is not divisible by 3` | The fork divides raw vocabulary by TP size; V4.1's vocab (129,280) is not divisible by 3 | Use a single `--served-model-name main` (the alias path); the fork runtime-pads attention heads, o_groups, and drafter experts automatically |
| 7 | Spec decode (DSpark k=5) boot OOMs at every gmu/KV combination | Draft model adds ~3 GiB/rank; at TP3 there is no room for both the draft and a meaningful KV pool | Rejected on capacity; documented as the speed/capacity trade |
| 8 | `expandable_segments:True` produces NaN logits on >64-token prompts | GB10 unified memory + PyTorch's expandable segments allocator interact badly | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (mandatory) |

## Failures in the bring-up path itself

These are defects in the recipe's own scripts, not in the serving stack —
they were found by re-running the published quick start end to end. If
you are on an older copy, expect them.

| Symptom | Root cause | Fix |
|---|---|---|
| `./launch/ds41-flash-tp3-up: Permission denied` | The launcher was published without the executable bit, so the quick start could not start | Published with mode 755 |
| `preflight failed on <box>` on the first box, every time, with `MISSING: checkpoint not staged` | The checkpoint check expanded `$(source "$ENV_SRC" ...)` on the BOX, where `ENV_SRC` is unset, so the path was empty and the test could never pass | Resolve every value on the operator's side, inject it as a literal, and let the box use its own variables |
| `preflight` never warned about memory | The `free -g | awk` field reference was written `$$7` inside a double-quoted ssh payload, so `$$` expanded to the operator's shell PID on the sending side and the box received a literal `PID7` instead of `$7` — the comparison never matched | `\$7`, so the box compares field 7 |
| `preflight` accepted any image named `eugr/spark-vllm-b12x:nightly-20261015` | The check compared the tag string only; a moved tag passes | The preflight now asserts the manifest digest AND the sha256 of the image's RootFS layer list |
| `start` failed with `bash: start-tp3-fork.sh: No such file or directory` | The launcher called a script the recipe did not ship, and `stage-env` copied only `cluster.env` | The recipe now ships `launch/start-tp3-fork.sh` and `launch/tp3-kit/`, and `stage-env` copies both to every box |
| `stage-env` failed with `No such file or directory` | The remote target directory was never created | `stage-env` creates the layout on each box first |
