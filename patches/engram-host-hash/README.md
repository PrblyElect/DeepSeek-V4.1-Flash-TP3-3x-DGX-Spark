# engram-host-hash — delete the per-step engram disk-read drain-sync

An optional engine overlay for this recipe's pinned image: it computes the
24 per-token engram hash ids on the **host** (CPU, numpy) from inputs that
are already host-known, and stages them straight into the disk tables'
pinned `ids_host` buffers — so the stock per-step GPU hash pipeline (the
`run_native` kernels plus the `_stage_ids` device-to-host copy and the
blocking `_read_staged` event sync that precedes every disk read) is
skipped entirely on admitted steps. The blocking drain disappears instead
of being replaced: no hash kernels, no ids D2H, no GPU event.

## Why

With `table_memory:disk`, every engine step that touches the engram reads
rows from NVMe, and the stock code makes each read wait on a CUDA event
that proves the freshly GPU-hashed row ids have landed in the pinned
staging buffer. That per-step drain-sync was measured at **~8% of worker
wall-time on this hardware class** under concurrent decode, and it
**balloons under memory pressure** — on a unified-memory box the wait
grows exactly when the box is busiest. The hash itself needs nothing the
GPU owns: every input (raw token ids, the request layout, the lookback
window) exists on the host or can be mirrored exactly with two tiny
device-to-host copies. Hashing on the host and staging from host memory
deletes the sync; with the overlay armed, the drain's measured share of
worker wall-time drops to **~2%** on this stack.

## The exactness guarantee

The host hash is **bit-identical to the engine's own reference oracle** —
`flashinfer/.../sequence/engram/reference.py::hash_reference`, the CPU
integer oracle the engine ships — row for row, including the triton
`_hash` kernel's `-1` tail for padded rows. That is not an assertion; it
is proven three ways:

1. **Golden test (`--pure`, CPU-only)** — the vectorized host hash vs the
   engine's own oracle on adversarial batches (dead `-1` tokens, pad-id
   tokens, image-sentinel-masked tokens, every history shape,
   single-token requests, an empty request mid-batch, out-of-vocab ids,
   padded tails) and 300 fuzz batches, under BOTH a synthetic geometry
   and the production geometry built from the model's `config.json` with
   the production compressed token map built from the real tokenizer.
   The pass bar is exactly 0 (`max_abs_diff == 0` everywhere).
2. **Golden test (`--native --allow-gpu`, GPU)** — identical GPU-side
   inputs through `native.run` (the triton kernels) vs the host hash;
   max-abs-diff must be 0. Runs only on a dedicated test boot or an idle
   device — it creates a CUDA context.
3. **Per-boot self-check (production)** — the overlay itself: the first
   plain batch AND the first spec batch of a host-path boot each run
   BOTH the native pipeline and the host hash on the same real inputs
   and must compare bit-exact before their path latches on. Any mismatch
   → **permanent native fallback for that path** plus a loud error log
   (`DS41 host engram hash self-check FAILED ...`); with
   `DS41_ENGRAM_HOST_HASH_STRICT=1` it raises instead.

Anything the host path cannot prove exact never takes the host path: an
availability predicate gates every step, and any shape it does not
admit silently runs the stock native pipeline (correct, just slower).
Admitted steps are:

- **plain steps** — every request a pure prefill chunk
  (`nc + query_len <= prefill_len`) or a caught-up single-token decode;
- **spec steps (DSpark verify)** — adaptive verification off,
  `num_new_sampled_tokens_per_step == 1`, not a FULL uniform-graph batch
  (`pad_spec_decode` shapes), and every request a no-draft prefill chunk
  or a caught-up verify (`query_len == drafts + 1`). The draft values are
  mirrored by a tiny fenced D2H of the SAME GPU buffer slice the combine
  kernel reads; the scheduler's `scheduled_spec_decode_tokens` dict is
  deliberately not used (it carries `-1` placeholders on this stack).

### Why the host inputs are the same bytes the GPU reads

| input | host source | exact because |
|-------|-------------|---------------|
| `starts` | `input_batch.query_start_loc_np` | it IS the source array the runner H2D-copies |
| `slots` | identity (`slot == request index`) | `common/engram.py` writes `slots[i] = i` |
| `token_map`, geometry | one-time D2H of the plan's `token_map`; geometry is host-only by construction | static after boot |
| prefill `input_ids` rows / decode/verify bonus row | per-request FENCED SLICE D2H of `req_states.all_token_ids.gpu` — batch row r mirrors req-state row `idx[r]`, absolute columns `[max(nc-3,0), nc+q)` into pinned rows (the lookback window plus the upcoming rows) | the GPU kernels read the same values (`_prepare_prefill_inputs` at `nc+i`; combine's bonus row = `last_sampled_tokens[rsi]` == `all_token_ids[rsi, nc]` under the caught-up invariant `nc == total_len - 1`); the D2H is enqueued before the fence record, so the fence covers the writing `post_update`/staged-write kernels; a read that would leave the mirrored window makes compose reject (native fallback), which bounds the pinned width by `3 + max_num_batched_tokens`. NOTE: the worker-side `all_token_ids` is a `StagedWriteTensor` with NO host view (`buffer_utils.py`), so this path never touches a `.np` accessor — none exists on the worker |
| spec draft rows | full-buffer FENCED D2H of `req_states.draft_tokens[:, :k]` (req-state indexed), enqueued at collection time before the fence record | the SAME buffer the combine kernel reads; written only by the previous step's post-forward proposal, which the fence covers |
| `history` (raw lookback) | gathered through the SAME sliced mirror, replicating `_gather_lookback_kernel` | scheduler and GPU `num_computed` advance identically: spec-off by `query_len`, spec-on by the accepted count |
| D2H/UVA landing | ONE CUDA event recorded on the current stream at the collection point (after every D2H, immediately before `prepare_disk_engram`), synchronized before any host read | covers every write the mirrors read, and its bounded wait is strictly less than the native `_read_staged` drain it replaces (the drain additionally waits through this step's prep + hash kernels and the hash D2H) |

The composition is additionally fuzz-tested against an independent port
of the runner's two input kernels (`_kernel_port_compose` inside
`host_hash.py`) and an inline ground truth computed from the ABSOLUTE
req-state rows, with non-identity batch→req-state mappings, garbage in
unmapped rows, and boundary shapes — the fuzz proves unmapped garbage
rows and beyond-width columns are never read.

## Files

| file | role |
|------|------|
| `host_hash.py` | the shipped module (install.py copies it to `vllm/ds41_engram_host.py`): vectorized hash, host-input collection, host staging, boot self-check, `--selftest` |
| `install.py` | apply / revert / verify / `--diff` for the three anchored engine diffs + the module ship |
| `golden_test.py` | the bit-exactness harness (`--pure` CPU; `--native --allow-gpu` on a dedicated test boot) |
| `audit_integration.py` | static integration audit of the shipped module against the real engine tree |

## Install

The overlay makes three anchored, visible-text, idempotent diffs and
ships one new file:

1. `vllm/models/deepseek_v4_1/nvidia/model.py` — `prepare_disk_engram`
   gains an optional `host_inputs` argument and branches to the host
   path when it is provided; otherwise the stock path runs unchanged.
2. `vllm/models/deepseek_v4_1/nvidia/model_state.py` — `prepare_inputs`
   builds the host-mirror inputs (the coherence fence and the spec
   draft-buffer D2H live inside) and passes them on; collection failures
   fall back to the native path unless STRICT re-raises them.
3. `flashinfer/.../b12x/sequence/_shared/disk_table.py` — `DiskRowCache`
   gains `stage_rows_host(host_ids, count)`: write pinned `ids_host`
   from host memory, STAGE ONLY, no D2H copy and no GPU event (an
   unrecorded event's synchronize returns immediately).

```sh
# stage the kit onto the box (placeholder user@box — the same box names
# config/cluster.env uses), then into the running container:
scp -r patches/engram-host-hash user@spark-head.local:/tmp/engram-kit
ssh user@spark-head.local 'docker cp /tmp/engram-kit ds41-flash-tp3:/tmp/engram-kit'

# show the exact patches first (touches nothing):
ssh user@spark-head.local 'docker exec ds41-flash-tp3 python3 /tmp/engram-kit/install.py --diff'

# apply against the container's engine tree (defaults match this
# recipe's image layout; override with --vllm-root / --b12x-root):
ssh user@spark-head.local 'docker exec ds41-flash-tp3 python3 /tmp/engram-kit/install.py --apply'
ssh user@spark-head.local 'docker exec ds41-flash-tp3 python3 /tmp/engram-kit/install.py --verify'
```

Apply NEVER reverts or force-patches: any anchor mismatch, foreign
marker state, or module drift is a hard refusal (exit 2), and every
patched file is compile-checked before it is written. `--verify` reports
each site as `patched`, `original (not applied)`, or
`FOREIGN/INCONSISTENT`.

**Install AFTER any file-integrity gate, never before.** Two of the
three patched files (`model.py` and `disk_table.py`) are SHA-256 pinned
in `launch/tp3-kit/engine-identity.json` and asserted by
`launch/tp3-kit/boot-gate-fork.py` at every boot. The order is: boot the
engine, let the gate pass on the stock bytes, THEN `--apply` (the same
pattern the recipe's other runtime overlays follow). On a persistent
engine tree, `--revert` before the next boot or the gate will —
correctly — fail on the changed bytes. Never move the identity pins to
accommodate this patch; the gate's refusal is the audit working. The
recipe's launcher does not install this overlay — it is always a
manual, reversible step.

Nothing is active until the engine sees:

```sh
DS41_ENGRAM_HOST_HASH=1          # enable the host path (read once)
DS41_ENGRAM_HOST_HASH_STRICT=1  # first boots: raise instead of falling back
```

Set STRICT for the first boots: it turns any self-check mismatch or
collection failure into a loud error instead of a silent fallback. Watch
the log for BOTH `self-check PASSED on a plain batch` and
`self-check PASSED on a spec batch`, plus the periodic
`host steps / fallbacks` counter line.

## Revert

```sh
ssh user@spark-head.local 'docker exec ds41-flash-tp3 python3 /tmp/engram-kit/install.py --revert'
```

Restores the three engine files byte-exactly (each anchor block is
replaced by its recorded original) and deletes the shipped module;
`--verify` afterwards must say `original (not applied)` everywhere and
`module: absent`. A reverted tree is byte-identical to stock, so the
identity gate passes again on the next boot.

## The golden test

`--pure` is CPU-only and safe beside the live engine; it imports the
live engine tree read-only from a scratch dir and writes nothing outside
it:

```sh
ssh user@spark-head.local 'docker exec ds41-flash-tp3 mkdir -p /tmp/ds41engram-golden'
ssh user@spark-head.local 'docker exec ds41-flash-tp3 sh -c \
  "cp /tmp/engram-kit/host_hash.py /tmp/engram-kit/golden_test.py /tmp/ds41engram-golden/"'
ssh user@spark-head.local 'docker exec ds41-flash-tp3 python3 \
  /tmp/ds41engram-golden/golden_test.py --pure --model-dir /model'
```

(`--model-dir /model` is where this recipe's launcher mounts the
checkpoint inside the container; pass any model directory containing the
checkpoint's `config.json` and tokenizer.)

It proves (1) adversarial + fuzz bit-exactness vs the engine's own
reference oracle under the synthetic AND the production geometry with
the production tokenizer map, (2) spec-shaped batches whose histories
were built by verify steps with rejections (bonus row + k draft rows),
and (3) the step-input composition vs an independent port of the
runner's kernels. `--native --allow-gpu` adds the GPU oracle comparison
— dedicated test boot or idle device only, never next to the serving
engine.

Known, deliberate divergence (documented and tested via substitution):
`reference.py` indexes `token_map` with Python semantics and assumes
in-vocab ids, while the GPU `_compress` marks out-of-vocab raw ids dead;
the host hash follows the GPU (the thing being replaced), and the golden
test feeds the reference an in-vocab substitution on those lanes
(identical compressed value either way).

`host_hash.py --selftest` is the fast standalone variant (no engine
imports): the same adversarial cases, a 200-batch hash fuzz, the
composition fuzz, and an int64 overflow-bound check for the production
geometry shape.

## The integration audit

`audit_integration.py` is a static, read-only audit of the SHIPPED
module against the real engine tree, run in-container:

```sh
ssh user@spark-head.local 'docker exec ds41-flash-tp3 python3 /tmp/engram-kit/audit_integration.py \
    --module /usr/local/lib/python3.12/dist-packages/vllm/ds41_engram_host.py \
    --patched-disk-table /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/sequence/_shared/disk_table.py'
```

Exit 0 iff every check passes:

- **A** every `self.X`/`rt.X` reference vs the attributes the shipped
  classes actually define, plus an undefined-`_helper` sweep (reintroducing
  a called-a-module-function-as-a-method bug makes the audit fail —
  mutation-tested);
- **B** every engine name the module touches — InputBatch dataclass
  fields, `DefaultModelState`, `get_current_vllm_config`, the
  `enable_adaptive_verification` field, RequestState /
  StagedWriteTensor / UvaBackedTensor / NgramHashState / Engram / model
  seams, `_READ_POOL`, `lookup_op`, `DiskRowCache._read_staged` /
  `transaction`, `LookupBinding` fields, `DiskTable._cache`, and
  `stage_rows_host` on the PATCHED tree;
- **C** every lazy import resolves in-container;
- **D** the shipped module's `--selftest` runs from the shipped file.

Standing order of operations (after the boot gate, before trusting the
host path on a serving engine): `install.py --apply` →
`audit_integration.py` → AUDIT must PASS → arm the env vars on the next
(re)start.

## Results (measured on this recipe's pinned image, 2026-11)

- Golden `--pure` in-container, production geometry + tokenizer:
  `max_abs_diff == 0` everywhere. Standalone and shipped-location
  `--selftest`: green.
- Integration audit on a scratch install of the shipped module: every
  check passes, zero failures.
- `install.py` round-trip on a scratch tree: `--verify` OK → `--revert`
  OK → `--verify` OK, the three engine files byte-identical to stock and
  the module removed on revert; an in-anchor tamper is refused with
  exit 2 and nothing written.
- On the serving tier's concurrent-decode workload the drain-sync's
  measured share of worker wall-time drops from ~8% to ~2%, with the
  boot self-check passing on both the plain and the spec path.

## License

MIT, like the rest of this recipe. The three patched files are upstream
Apache-2.0 engine sources; every inserted block carries a
`DS41-ENGRAM-HOST-HASH BEGIN/END` change marker, and `install.py
--revert` restores the recorded originals byte-exactly.
