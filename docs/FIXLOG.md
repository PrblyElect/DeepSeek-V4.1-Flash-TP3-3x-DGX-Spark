# FIXLOG — ds41-flash-tp3 (V4.1-Flash on 3× DGX Spark, vLLM b12x)

Every failure mode encountered during bring-up, its root cause, and its
fix. Ordered by severity (boot-fatal → quality).

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Head node kernel-OOM during weight load; workers survive | NCCL connection buffers pin ~4.7 GiB unreclaimable host RAM per node; the head (carrying API server + relay + daemons) has ~2 GiB less margin than workers | `NCCL_BUFFSIZE=1048576`, `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO="^LL128"` — cuts pinned to 139 MiB |
| 2 | Kernel-OOM ~7 s into load with only 1.4 GiB process RSS | Engram tables staged into host memory (~31 GiB/rank) on top of 95 GiB weights | `--engram-config '{"cpu_offload":false,"table_memory":"disk"}'` |
| 3 | Head node OOM at ~4 min into load (repeats identically across 4 boots) | The b12x O_DIRECT loader's pinned bounce buffers add 10–16 GiB unreclaimable host memory during weight loading | `--load-format safetensors` (kernel-evictable page cache instead of pinned buffers) |
| 4 | `ValueError: Free memory on device cuda:0 less than desired GPU memory utilization` | The fork checks free memory at startup against gmu × total; busy head node has less free than clean workers | gmu 0.85 (not the fork default 0.92); run `echo 3 > /proc/sys/vm/drop_caches` before boot |
| 5 | `io_uring bounce initialization failed: Operation not permitted` | Docker's default seccomp profile blocks io_uring syscalls used by the b12x I/O layer | `--security-opt seccomp=unconfined` |
| 6 | `AssertionError: 129280 is not divisible by 3` | The fork divides raw vocabulary by TP size; V4.1's vocab (129,280) is not divisible by 3 | Use a single `--served-model-name main` (the alias path); the fork runtime-pads attention heads, o_groups, and drafter experts automatically |
| 7 | Spec decode (DSpark k=5) boot OOMs at every gmu/KV combination | Draft model adds ~3 GiB/rank; at TP3 there is no room for both the draft and a meaningful KV pool | Rejected on capacity; documented as the speed/capacity trade |
| 8 | `expandable_segments:True` produces NaN logits on >64-token prompts | GB10 unified memory + PyTorch's expandable segments allocator interact badly | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (mandatory) |
| 9 | RAG queries returning empty responses | WebUI's `CONTENT_EXTRACTION_ENGINE` not routed to the document extraction service | Set to `docling` with the correct service URL |
