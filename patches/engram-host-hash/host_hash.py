"""DS41 host-side Engram hash — compute the 24 per-token hash ids on the CPU
from host-known inputs and stage them straight into the disk tables' pinned
``ids_host`` buffers.

Replaces, on the host path, the per-step GPU pipeline

    run_native: _prepare_metadata + per-layer _hash kernels  (GPU)
    run_lookups: _stage_ids D2H + event record + _read_staged event sync
                 (the blocking drain that costs ~8% of worker wall-time
                  on this hardware class and amplifies under memory
                  pressure)

with a bit-exact numpy vectorization of the same equation (the CPU oracle
``flashinfer/.../sequence/engram/reference.py``; the GPU oracle is the
triton ``_hash`` kernel, whose semantics the vectorization also replicates
row-for-row, including the -1 tail for padded rows).

Semantics replicated exactly (see golden_test.py for the proofs):

- compression: ``token_map[t]`` when kept and ``0 <= t < vocab`` else ``-1``
  (triton ``_compress``); the keep mask is ``input_ids != 129264`` (the
  image sentinel).
- history compaction: raw lookback value kept iff ``0 <= raw < vocab`` and
  ``raw != 129264``, else ``-1`` (triton ``_prepare_metadata``).
- request id per token: largest r with ``starts[r] <= t`` (triton
  ``_request_ids_kernel`` binary search).
- source at lag L for token t: ``compressed[t-L]`` when ``t-L >= start``
  else ``history[req, 3 + (t - start - L)]`` (triton ``_source_token`` with
  MAX_ORDER=4; col in {0,1,2}); dead marker ``-1`` sticks (blocked flag is
  cumulative over lags 0..L).
- rolling value: ``mixed ^= (pad if blocked else source) * mult[L]``, and
  head group g (heads [8g, 8g+8)) emits ``mixed_through_lag(g+1) % prime
  + offset`` — identical to reference.py's ``rolling`` at lag = g+1.
- padded rows (t >= live tokens) are written as ``-1`` exactly like the
  GPU kernel's ``tl.where(live, result, -1)``.

SPEC EXTENSION (DSpark verify steps take the host path too): the draft
values the GPU combine kernel reads come from the persistent
``req_states.draft_tokens`` buffer — REQ-STATE indexed: the speculator
writes ``draft_tokens[idx_mapping[batch]]`` (gpu/model_runner.py:2569-2571)
and the combine kernel reads the ``req_state_idx`` row
(gpu/input_batch.py:460-463); slots come off ``free_indices.pop()`` (the
END), so batch row and req-state slot are essentially never identical.
The host path enqueues a tiny full-buffer D2H (max_reqs x k int64, garbage
in unassigned rows by design) plus per-request SLICED D2Hs of
``all_token_ids`` (the worker-side StagedWriteTensor has NO host view —
buffer_utils.py:132-176 — so the exact slices the composition reads, the
lookback window [nc-3, nc) and the upcoming rows [nc, nc+q), are copied
into pinned rows), fences both with one event recorded before any host
read, and indexes the mirrors by ``idx_mapping`` exactly like every other
req-state tensor; the composition fuzz proves unmapped garbage rows and
beyond-width columns are never read. The
scheduler's ``scheduled_spec_decode_tokens`` dict is NOT used: on this
stack it carries ``-1`` placeholders (the DraftTokensHandler copies real
drafts to the host only for structured-output batches,
``spec_decode/utils.py:116-145``). Adaptive-verification steps (GPU-side
draft reallocation desyncs ``query_start_loc_np``) and FULL uniform-graph
spec steps fall back to the native path.

CPU-mirror trust: ``num_computed_tokens_np`` is trusted as exact for
the requests the host path composes — spec-off accounting advances scheduler and GPU
identically, and under spec both advance by the accepted count (post_update
delta = query_len - num_rejected = num_sampled), so the caught-up invariant
``nc == total_len - 1`` holds and the bonus row is exactly
``all_token_ids[rsi, nc]``; the final-chunk sampler edge
(``first_logit_seq_pos == prefill_len - 1``, where the combine gate fails
and the row keeps its prompt value) and a finish-prefill-with-drafts span
are unreachable schedules here — the scheduler attaches drafts only after
a request's prompt is fully computed — and both fall back if ever seen.

Staging: the host writes the flattened (rows*24) int64 ids into each
table's ``ids_host`` pinned buffer (``stage_rows_host``, added to
disk_table.py by this kit — stage only, no GPU work), then mirrors the
native ``run_lookups`` shape exactly: stage all tables first (cheap host
copies), read concurrently on the engine's own thread pool + main thread
via ``_read_staged``, then the identical ``lookup_op`` calls. No D2H id
copy and no GPU event exist on this path, so the drain sync disappears
instead of being replaced.

This file is shipped verbatim into the engine tree as
``vllm/ds41_engram_host.py`` by install.py; it also runs standalone
(``python3 host_hash.py --selftest``) for the golden test.
"""

from __future__ import annotations

import os
import threading

import numpy as np

IMAGE_SENTINEL = 129264  # vllm image_sentinel_mask base id (== token_map dead rule)
NUM_HEADS = 24
MAX_ORDER = 4

_ENABLED_CACHE: bool | None = None
_STRICT_CACHE: bool | None = None


def host_hash_enabled() -> bool:
    """DS41_ENGRAM_HOST_HASH=1 gates the whole host path (read once)."""
    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        _ENABLED_CACHE = os.environ.get("DS41_ENGRAM_HOST_HASH", "") == "1"
    return _ENABLED_CACHE


def strict_mode() -> bool:
    """DS41_ENGRAM_HOST_HASH_STRICT=1 raises on any host/native mismatch."""
    global _STRICT_CACHE
    if _STRICT_CACHE is None:
        _STRICT_CACHE = os.environ.get("DS41_ENGRAM_HOST_HASH_STRICT", "") == "1"
    return _STRICT_CACHE


# ---------------------------------------------------------------------------
# Pure vectorized hash (no engine imports; numpy only)
# ---------------------------------------------------------------------------


def compress_tokens(tokens_raw, keep, token_map, vocab):
    """Replicates triton ``_compress``: token_map[t] iff kept and in vocab."""
    ids = np.asarray(tokens_raw, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    safe = (ids >= 0) & (ids < vocab)
    tm = token_map
    gathered = tm[np.where(safe, ids, 0)]
    return np.where(keep & safe, gathered, np.int64(-1))


def compress_history(history_raw, token_map, vocab):
    """Replicates ``_prepare_metadata``'s history column rule (raw -> -1/compressed)."""
    h = np.asarray(history_raw, dtype=np.int64)
    valid = (h >= 0) & (h < vocab) & (h != IMAGE_SENTINEL)
    tm = token_map
    gathered = tm[np.where(valid, h, 0)]
    return np.where(valid, gathered, np.int64(-1))


def request_ids(starts, live):
    """Largest r with starts[r] <= t, for t in [0, live); matches the kernel's
    binary search (empty requests are skipped by construction)."""
    s = np.asarray(starts, dtype=np.int64)
    t = np.arange(live, dtype=np.int64)
    return np.searchsorted(s, t, side="right") - 1


def hash_layer(
    tokens_raw,
    keep,
    starts,
    history_raw,
    token_map,
    vocab,
    multipliers,
    primes,
    offsets,
    padded_rows,
):
    """Vectorized per-layer hash; returns (padded_rows, 24) int64.

    Live rows [0, live) carry the hash; padded tail rows are -1 (identical
    to the GPU binding's hash_ids layout that run_lookups stages).
    """
    s = np.asarray(starts, dtype=np.int64)
    num_reqs = len(s) - 1
    live = int(s[num_reqs]) if num_reqs else 0
    n_out = int(padded_rows)
    if live == 0:
        return np.full((n_out, NUM_HEADS), -1, dtype=np.int64)

    comp = compress_tokens(tokens_raw[:live], keep[:live], token_map, vocab)
    hc = compress_history(history_raw[:num_reqs], token_map, vocab)
    req = request_ids(s, live)
    rel = np.arange(live, dtype=np.int64) - s[req]

    pad = int(token_map[2])
    mult = np.asarray(multipliers, dtype=np.int64)  # (4,)

    # S[t, lag]: the source token at each lag (dead marker -1 included).
    S = np.empty((live, MAX_ORDER), dtype=np.int64)
    t = np.arange(live, dtype=np.int64)
    for lag in range(MAX_ORDER):
        src_rel = rel - lag
        from_query = src_rel >= 0
        q_idx = np.clip(t - lag, 0, live - 1)
        h_col = np.clip(3 + src_rel, 0, 2)
        S[:, lag] = np.where(
            from_query, comp[q_idx], hc[req, h_col]
        )

    blocked = np.logical_or.accumulate(S == -1, axis=1)  # sticky over lags
    V = np.where(blocked, np.int64(pad), S)
    C = V * mult[None, :]
    M = np.bitwise_xor.accumulate(C, axis=1)  # M[:, k] == rolling through lag k

    out = np.empty((n_out, NUM_HEADS), dtype=np.int64)
    for g in range(3):  # head group g -> heads [8g, 8g+8) -> lag g+1
        m = M[:, g + 1][:, None]
        p = np.asarray(primes[g * 8 : (g + 1) * 8], dtype=np.int64)[None, :]
        o = np.asarray(offsets[g * 8 : (g + 1) * 8], dtype=np.int64)[None, :]
        out[:live, g * 8 : (g + 1) * 8] = m % p + o
    if n_out > live:
        out[live:] = -1
    return out


def hash_all_layers(
    tokens_raw,
    keep,
    starts,
    history_raw,
    token_map,
    vocab,
    layer_geometry,  # sequence of (multipliers(4,), primes(24,), offsets(24,))
    padded_rows,
):
    """(L, padded_rows, 24) int64 — layer i feeds prepared_engram_hashes[:, i].

    NOTE the axis order: the model's ``prepared_engram_hashes`` buffer is
    (tokens, L, 24); transpose(1, 0, 2) where it is uploaded (see
    run_prepare_host).
    """
    return np.stack(
        [
            hash_layer(
                tokens_raw,
                keep,
                starts,
                history_raw,
                token_map,
                vocab,
                m,
                p,
                o,
                padded_rows,
            )
            for (m, p, o) in layer_geometry
        ],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Pure step-input composition (the two runner kernels' admitted shapes)
# ---------------------------------------------------------------------------


def compose_step_inputs(
    padded_rows,
    starts,
    idx,
    nc,
    pl,
    all_rows,
    row_base,
    row_width,
    num_drafts,
    drafts_rows,
):
    """Compose the step's input ids host-side; (ids, ok).

    Replicates, for the schedules the host path admits, the combined effect
    of ``prepare_prefill_inputs`` (chunk rows from all_token_ids) and
    ``_combine_sampled_and_draft_tokens`` (bonus row = last accepted token,
    draft rows from the per-request draft buffer):

      A) prefill chunk   nc + query_len <= prefill_len:
             ids[qs:qe] = all_token_ids[rsi, nc : nc + query_len]
      B) caught-up decode (no drafts)  nc >= prefill_len, query_len == 1:
             ids[qs]    = all_token_ids[rsi, nc]   (== last_sampled_tokens)
      C) caught-up verify (spec)       nc >= prefill_len,
                                       query_len == num_drafts + 1:
             ids[qs]    = all_token_ids[rsi, nc]   (bonus row)
             ids[qs+1:qe] = drafts_rows[rsi, :num_drafts]

    ``all_rows`` is the SLICED pinned mirror of all_token_ids built by
    maybe_collect's gather D2H: BATCH-row r holds req-state row idx[r]'s
    absolute columns [row_base[r], row_base[r] + row_width[r]) — exactly
    the lookback window [nc-3, nc) plus the upcoming rows [nc, nc+q) the
    composition reads (chunk prompt continuation / the bonus row). A read
    whose absolute column leaves that window returns ok=False (native
    fallback), which is what makes the mirror's bounded width safe.

    ``drafts_rows`` is REQ-STATE indexed (the GPU draft buffer's rows are
    req-state slots — the speculator writes draft_tokens[idx_mapping[batch]]
    and the combine kernel reads the req_state_idx row), so drafts are
    gathered by ``idx[r]`` exactly like nc/pl.

    Any other per-request shape returns ok=False (native fallback); those
    shapes are unreachable schedules today (see the CPU-mirror trust note
    in the module docstring) and are deliberately not guessed at.
    """
    starts = np.asarray(starts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    nc = np.asarray(nc, dtype=np.int64)
    pl = np.asarray(pl, dtype=np.int64)
    row_base = np.asarray(row_base, dtype=np.int64)
    row_width = np.asarray(row_width, dtype=np.int64)
    num_reqs = len(starts) - 1
    ids = np.zeros(int(padded_rows), dtype=np.int64)
    spec = num_drafts is not None
    if spec:
        num_drafts = np.asarray(num_drafts, dtype=np.int64)

    for r in range(num_reqs):
        qs, qe = int(starts[r]), int(starts[r + 1])
        q = qe - qs
        if q <= 0:
            continue
        rsi, c, p = int(idx[r]), int(nc[r]), int(pl[r])
        base, width = int(row_base[r]), int(row_width[r])
        lo = c - base  # mirror-relative column of absolute column c
        nd = int(num_drafts[r]) if spec else 0
        if c + q <= p:  # A) prefill chunk
            hi = lo + q
            if lo < 0 or hi > width:
                return ids, False
            ids[qs:qe] = all_rows[r, lo:hi]
        elif not spec and c >= p and q == 1:  # B) caught-up decode
            if lo < 0 or lo >= width:
                return ids, False
            ids[qs] = all_rows[r, lo]
        elif spec and c >= p and q == nd + 1 and nd >= 0:  # C) verify
            if (lo < 0 or lo >= width) or rsi >= drafts_rows.shape[0]:
                return ids, False
            ids[qs] = all_rows[r, lo]
            if nd:
                # Draft rows are REQ-STATE indexed (rsi), matching the
                # combine kernel's draft_tokens_ptr + req_state_idx * stride.
                ids[qs + 1 : qe] = drafts_rows[rsi, :nd]
        else:
            return ids, False
    return ids, True


# ---------------------------------------------------------------------------
# Reference oracle (exact port of flashinfer engram/reference.py's loop, with
# the geometry unpacked so it also runs without the engine package).
# ---------------------------------------------------------------------------


def hash_reference_port(
    tokens, mask, starts, slots, history, token_map, multipliers, primes, offsets
):
    """The 35-line CPU oracle, verbatim semantics (Python int arithmetic).

    ``history`` here is the COMPRESSED history (as reference.py expects);
    ``tokens``/``mask``/``starts`` mirror the engine call path.
    """
    pad = token_map[2]
    compressed = [
        token_map[t] if keep else -1 for t, keep in zip(tokens, mask, strict=True)
    ]
    rows = []
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        cache = list(history[slots[request]]) + compressed[begin:end]
        for position in range(3, len(cache)):
            rolling, blocked, result = 0, False, []
            for lag in range(4):
                source = cache[position - lag]
                blocked |= source == -1
                rolling ^= (pad if blocked else source) * multipliers[lag]
                if lag:
                    for head in range((lag - 1) * 8, lag * 8):
                        result.append(
                            rolling % primes[head] + offsets[head]
                        )
            rows.append(result)
    return rows


# ---------------------------------------------------------------------------
# Engine-side runtime (imports are lazy so the pure functions stay standalone)
# ---------------------------------------------------------------------------


class HostEngramInputs:
    """One step's host-mirrored hash inputs (all exact-by-construction).

    input_ids     (padded_rows,) int64 — raw token ids, host mirror
    keep          (live,) bool         — input_ids != image sentinel
    starts        (num_reqs+1,) int64  — == input_batch.query_start_loc_np
    history_raw   (num_reqs, 3) int64  — raw lookback window ([-1 padding])
    live          int                  — starts[num_reqs]
    padded_rows   int                  — input_ids.shape[0] the model sees
    is_spec       bool                 — this step carries draft tokens
    """

    __slots__ = (
        "input_ids",
        "keep",
        "starts",
        "history_raw",
        "live",
        "padded_rows",
        "is_spec",
    )


class HostEngramRuntime:
    """Per-DeepseekV4Model cached host state (token map, geometry, pinned H2D
    mirrors, boot self-check latches, counters). Created lazily on first use."""

    def __init__(self, model):
        self.log = _engine_logger()
        eng = model.engram_hash
        eng._ensure_bindings()
        layout = eng.layout
        self.vocab = int(layout.caps[0].vocab_size)
        self.num_layers = len(layout.hash_plans)
        # token_map: static after boot; one-time D2H of the plan's tensor.
        token_map = eng._token_map
        self.token_map = (
            token_map.detach().cpu().numpy().astype(np.int64, copy=True)
            if not token_map.is_cpu
            else token_map.numpy().astype(np.int64, copy=True)
        )
        self.pad = int(self.token_map[2])
        geom = layout.geometry
        self.layer_geometry = [
            (
                np.asarray(geom.multipliers[i], dtype=np.int64),
                np.asarray(geom.primes[i], dtype=np.int64),
                np.asarray(geom.offsets[i], dtype=np.int64),
            )
            for i in range(self.num_layers)
        ]
        caps = layout.caps[0]
        max_tokens = int(caps.max_tokens)
        import torch

        self._torch = torch
        # prepared_engram_hashes is (max_tokens, L, 24) — pinned mirror in
        # the MODEL's axis order; compute() returns (L, n, 24) and is
        # transposed at the upload site.
        self.pinned_hashes = torch.empty(
            (max_tokens, self.num_layers, NUM_HEADS),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.pinned_hashes_np = self.pinned_hashes.numpy()
        self.checked_plain = False
        self.checked_spec = False
        self.disabled = False  # plain path permanently off
        self.spec_disabled = False  # spec path permanently off (plain may live)
        self.fallbacks = 0
        self.host_steps = 0
        self._fallback_logged = False
        self._lock = threading.Lock()

    def compute(self, host: HostEngramInputs):
        return hash_all_layers(
            host.input_ids,
            host.keep,
            host.starts,
            host.history_raw,
            self.token_map,
            self.vocab,
            self.layer_geometry,
            host.padded_rows,
        )

    def native_compare(self, model, host, hashes, input_ids, query_start_loc,
                       lookback, tag):
        """Run BOTH host and native on one real batch; require bit-exact.

        Fills ``hashes`` via the native pipeline (run_native), copies it to
        the host (boot-time sync), and compares against the host hash with
        axes matched: native (n, L, 24) vs host (L, n, 24).transpose(1, 0, 2).
        Returns True on bit-exact equality; on mismatch disables that path
        permanently + loud log (raise under STRICT).
        """
        torch = self._torch
        live = host.live
        model.engram_hash.num_tokens.fill_(live)
        model.engram_hash.run_native(
            input_ids,
            ~_image_mask(input_ids),  # module function: == image_sentinel_mask
            query_start_loc,
            lookback,
            hashes,
        )
        native_np = hashes.detach().cpu().numpy()  # (n, L, 24)
        host_np = self.compute(host)  # (L, n, 24)
        host_np = host_np.transpose(1, 0, 2)  # -> (n, L, 24)
        ok = native_np.shape == host_np.shape and np.array_equal(native_np, host_np)
        if not ok:
            if native_np.shape == host_np.shape:
                worst = int(np.abs(native_np.astype(np.int64) - host_np).max())
                detail = f"max abs diff {worst}"
            else:
                detail = f"shape native {native_np.shape} host {host_np.shape}"
            msg = (
                f"DS41 host engram hash SELF-CHECK FAILED on a {tag} batch "
                f"({detail}); falling back to native hashing permanently "
                "for that path."
            )
            self.log.error(msg)
            if strict_mode():
                raise RuntimeError(msg)
            return False
        self.log.info(
            "DS41 host engram hash self-check PASSED on a %s batch "
            "(bit-exact, %d rows, %d live, %d layers).",
            tag, host.padded_rows, live, self.num_layers,
        )
        return True


def _image_mask(input_ids):
    """== vllm's image_sentinel_mask (token_ids == 129264), torch-free."""
    return input_ids == IMAGE_SENTINEL


def _lazy_torch():
    import torch

    return torch


def _engine_logger():
    try:
        from vllm.logger import init_logger

        return init_logger("vllm.ds41_engram_host")
    except Exception:
        import logging

        logging.basicConfig(level=logging.INFO)
        return logging.getLogger("vllm.ds41_engram_host")


_RUNTIMES: dict[int, HostEngramRuntime] = {}


def _runtime(model) -> HostEngramRuntime:
    key = id(model)
    rt = _RUNTIMES.get(key)
    if rt is None:
        rt = HostEngramRuntime(model)
        _RUNTIMES[key] = rt
    return rt


# Pinned host mirrors for the collection D2Hs. Drafts: full buffer, sized
# once. all_token_ids slices: one grow-only (max_reqs, width) int32 buffer
# whose width is bounded by 3 + max_num_batched_tokens; only the first
# row_width[r] columns of row r are ever read.
_DRAFTS_PINNED: dict[tuple[int, int], "object"] = {}
_ALL_ROWS_PINNED: dict[int, "object"] = {}


def _pinned_drafts(max_reqs: int, num_spec_steps: int):
    torch = _lazy_torch()
    key = (max_reqs, num_spec_steps)
    buf = _DRAFTS_PINNED.get(key)
    if buf is None:
        buf = torch.empty(
            (max_reqs, max(num_spec_steps, 1)),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        _DRAFTS_PINNED[key] = buf
    return buf


def _pinned_all_rows(max_reqs: int, width: int):
    torch = _lazy_torch()
    buf = _ALL_ROWS_PINNED.get(max_reqs)
    if buf is None or buf.shape[1] < width:
        buf = torch.empty(
            (max_reqs, width),
            dtype=torch.int32,  # all_token_ids dtype
            device="cpu",
            pin_memory=True,
        )
        _ALL_ROWS_PINNED[max_reqs] = buf
    return buf


# ---------------------------------------------------------------------------
# Host-input collection (called from the patched model_state.prepare_inputs,
# i.e. at the END of prepare_inputs, immediately before prepare_disk_engram)
# ---------------------------------------------------------------------------


def maybe_collect(model_state, input_batch, req_states):
    """Build HostEngramInputs for this step, or None (native fallback).

    Availability contract (every clause is exact-by-construction; documented
    in the kit README):

    - DS41_ENGRAM_HOST_HASH=1 (checked once).
    - Plain steps (no drafts): every request a pure prefill chunk
      (nc + query_len <= prefill_len) or a caught-up single-token decode.
    - Spec steps (drafts present): adaptive verification OFF (config),
      num_new_sampled_tokens_per_step == 1, not a FULL uniform-graph batch
      (pad_spec_decode), and every request a no-draft prefill chunk or a
      caught-up verify (query_len == drafts + 1). Draft values are mirrored
      by a tiny D2H of the SAME GPU buffer slice the combine kernel reads.
    - UVA/D2H coherence fence: one CUDA event recorded on the current
      stream at this collection point (after the drafts D2H when present)
      and synchronized before any host read. It covers every write the
      mirrors read — the previous step's post_update (accepted tokens into
      all_token_ids via UVA) and its speculator proposal (draft buffer) —
      and its bounded wait is strictly less than the native _read_staged
      drain it replaces (the drain additionally waits through this step's
      prep + hash kernels and the hash D2H).
    """
    if not host_hash_enabled():
        return None
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        return None
    starts_np = input_batch.query_start_loc_np
    if starts_np is None or len(starts_np) < num_reqs + 1:
        return None
    starts = starts_np[: num_reqs + 1].astype(np.int64)
    live = int(starts[num_reqs])
    n = int(input_batch.input_ids.shape[0])
    if live <= 0 or live > n:
        return None

    drafts_per_req = input_batch.num_draft_tokens_per_req
    is_spec = drafts_per_req is not None and int(drafts_per_req.sum()) > 0
    torch = _lazy_torch()
    drafts_host = None
    num_drafts = None

    # idx_mapping (batch row -> req-state slot) is a plain host np array:
    # available at collection start, no fence needed. NOTE the draft buffer
    # is REQ-STATE indexed — the speculator writes
    # draft_tokens[idx_mapping[batch]] (gpu/model_runner.py:2569-2571) and
    # the combine kernel reads the req_state_idx row
    # (gpu/input_batch.py:460-463); slots come off free_indices.pop() (the
    # END), so batch row r is essentially never req-state slot r.
    idx = input_batch.idx_mapping_np[:num_reqs].astype(np.int64)

    if is_spec:
        if input_batch.uniform_decode_graph:
            return None  # pad_spec_decode shapes: native fallback
        try:
            from vllm.config import get_current_vllm_config

            sc = get_current_vllm_config().speculative_config
        except Exception:
            sc = None
        if sc is not None and getattr(sc, "enable_adaptive_verification", False):
            return None  # GPU draft reallocation desyncs query_start_loc_np
        if int(getattr(model_state, "num_new_sampled_tokens_per_step", 1)) != 1:
            return None
        k = int(drafts_per_req.max())
        gpu_drafts = req_states.draft_tokens  # (max_num_reqs, num_spec_steps)
        if k > gpu_drafts.shape[1] or int(idx.max()) >= gpu_drafts.shape[0]:
            return None
        pinned = _pinned_drafts(int(gpu_drafts.shape[0]), int(gpu_drafts.shape[1]))
        # FULL-buffer D2H (req-state rows, garbage in unassigned slots by
        # design): compose indexes it by idx_mapping exactly like nc/pl/
        # all_token_ids. Tiny — max_num_reqs * k * 8 bytes. Enqueued BEFORE
        # the fence record so the fence covers both this copy and the
        # all_token_ids slice copies below.
        pinned[: gpu_drafts.shape[0], :k].copy_(
            gpu_drafts[:, :k], non_blocking=True
        )
        drafts_host = pinned.numpy()
        num_drafts = np.asarray(drafts_per_req, dtype=np.int64)

    # all_token_ids on the WORKER is a StagedWriteTensor with NO host view
    # (its .gpu is a true GPU tensor in the non-UVA default; buffer_utils
    # :132-176) — the .np accessor exists only on the scheduler-side UVA
    # form. Mirror exactly the slices the composition reads with per-request
    # async D2H into pinned rows: absolute columns [max(nc-3,0), nc+q) —
    # the 3-token lookback window plus the upcoming rows (prompt
    # continuation for chunks, the bonus row for verify/decode; drafts come
    # from the draft buffer). Enqueued BEFORE the fence record, same
    # pattern as the drafts D2H; width stays bounded by 3 + query_len <=
    # 3 + max_num_batched_tokens.
    nc = req_states.num_computed_tokens_np[idx].astype(np.int64)
    pl = req_states.prefill_len.np[idx].astype(np.int64)
    all_gpu = req_states.all_token_ids.gpu  # GPU or UVA view; D2H works
    row_len = int(all_gpu.shape[1])
    qlen = starts[1 : num_reqs + 1] - starts[:num_reqs]
    row_base = np.maximum(nc - 3, 0)
    row_end = np.minimum(nc + qlen, row_len)
    row_width = row_end - row_base
    if int(row_width.min()) < 0 or int(row_width.max()) > n + 3:
        return None
    pinned_rows = _pinned_all_rows(int(all_gpu.shape[0]), n + 3)
    for r in range(num_reqs):
        w = int(row_width[r])
        if w <= 0:
            continue
        pinned_rows[r, :w].copy_(
            all_gpu[int(idx[r]), int(row_base[r]) : int(row_base[r]) + w],
            non_blocking=True,
        )

    # Coherence fence: one event AFTER every D2H above, synchronized before
    # the host reads the mirrors below.
    event = getattr(model_state, "_ds41_engram_fence", None)
    if event is None:
        event = torch.cuda.Event()
        model_state._ds41_engram_fence = event
    event.record(torch.cuda.current_stream())
    event.synchronize()

    rows_np = pinned_rows.numpy()

    ids, ok = compose_step_inputs(
        n, starts, idx, nc, pl, rows_np, row_base, row_width, num_drafts,
        drafts_host,
    )
    if not ok:
        return None  # shape the host path does not admit: native fallback

    # Raw lookback window (replicates _gather_lookback_kernel host-side):
    # position nc-3+j lives in mirror row r at column (nc-3+j) - row_base[r].
    history = np.full((num_reqs, 3), -1, dtype=np.int64)
    pos = nc[:, None] - 3 + np.arange(3, dtype=np.int64)[None, :]
    valid = pos >= 0
    rel = pos - row_base[:, None]
    r_rows = np.repeat(np.arange(num_reqs)[:, None], 3, axis=1)
    history[valid] = rows_np[r_rows[valid], rel[valid]]

    host = HostEngramInputs()
    host.input_ids = ids
    host.keep = ids[:live] != IMAGE_SENTINEL
    host.starts = starts
    host.history_raw = history
    host.live = live
    host.padded_rows = n
    host.is_spec = bool(is_spec)
    return host


# ---------------------------------------------------------------------------
# Host-path prepare (called from the patched model.prepare_disk_engram)
# ---------------------------------------------------------------------------


def run_prepare_host(
    model, engrams, hashes, input_ids, query_start_loc, lookback, host_inputs
):
    """Full host replacement for run_native + stage_disk + run_lookups.

    Returns True when the step ran on the host path; False -> the caller
    runs the stock native path. Never raises for availability reasons.

    Boot self-check: the FIRST plain batch and the FIRST spec batch each run
    BOTH the native pipeline and the host hash on the same real inputs and
    must compare bit-exact before their path latches on (mismatch ->
    permanent fallback for that path + loud log; STRICT -> raise).
    """
    try:
        rt = _runtime(model)
        if rt.disabled:
            return False
        if host_inputs.is_spec:
            if rt.spec_disabled:
                return False
            if not rt.checked_spec:
                if not rt.native_compare(
                    model, host_inputs, hashes, input_ids, query_start_loc,
                    lookback, "spec",
                ):
                    rt.spec_disabled = True
                    return False
                rt.checked_spec = True
        elif not rt.checked_plain:
            if not rt.native_compare(
                model, host_inputs, hashes, input_ids, query_start_loc,
                lookback, "plain",
            ):
                rt.disabled = True
                return False
            rt.checked_plain = True
        torch = rt._torch
        host_np = rt.compute(host_inputs)  # (L, n, 24)
        n = host_inputs.padded_rows
        live = host_inputs.live

        # GPU consumers: prepared_engram_hashes (n, L, 24) via pinned H2D
        # (async, stream-ordered before stage_disk's copies and lookup_op).
        rt.pinned_hashes_np[:n] = host_np.transpose(1, 0, 2)
        model.prepared_engram_hashes[:n].copy_(
            rt.pinned_hashes[:n], non_blocking=True
        )
        model.engram_hash.num_tokens.fill_(live)

        bindings = []
        host_ids = []
        for engram in engrams:
            bindings.append(
                engram.stage_disk(
                    hashes[:, engram.layer_hash_index], model.engram_hash.num_tokens
                )
            )
            # Per-table flat ids (n*24): the layer's full padded hash block,
            # byte-identical to what _stage_ids would have copied D2H.
            host_ids.append(
                torch.from_numpy(np.ascontiguousarray(host_np[engram.layer_hash_index]))
            )
        counts = [n] * len(bindings)
        run_lookups_host(bindings, host_ids, counts, clear_tail=False)
        with rt._lock:
            rt.host_steps += 1
            if rt.host_steps % 1000 == 0:
                rt.log.info(
                    "DS41 host engram hash: %d host steps, %d fallbacks",
                    rt.host_steps,
                    rt.fallbacks,
                )
        return True
    except Exception:
        rt = _RUNTIMES.get(id(model))
        if rt is not None:
            rt.fallbacks += 1
            if not rt._fallback_logged:
                rt._fallback_logged = True
                rt.log.exception(
                    "DS41 host engram path failed; falling back to native "
                    "hashing (this message logs once)"
                )
        if strict_mode():
            raise
        return False


def run_lookups_host(bindings, host_ids, token_counts, *, clear_tail=True):
    """Mirror of b12x.sequence.engram._impl.run_lookups with host staging.

    Native order, byte for byte: open every disk table's transaction and
    STAGE all tables first (here: host memcpy into the pinned ids_host —
    no GPU work at all), then read concurrently (the engine's own thread
    pool for tables[1:] + the main thread for tables[0]), then the same
    lookup_op calls with the same counts (n*24 staged ids including the
    -1 tails, n prepared tokens).
    """
    from b12x.sequence.engram._impl import _READ_POOL  # noqa: PLC0415
    from b12x.sequence.engram._kernels import lookup_op  # noqa: PLC0415

    bindings = tuple(bindings)
    counts = tuple(int(c) for c in token_counts)
    if len(bindings) != len(counts) or len(bindings) != len(host_ids):
        raise ValueError("one token count and host id array per binding")
    disk = [i for i, b in enumerate(bindings) if b.disk_table is not None]
    if len(disk) < 2:
        results = []
        for b, hid, n in zip(bindings, host_ids, counts):
            if b.disk_table is None:
                lookup_op(
                    b.plan.handle,
                    b.weight,
                    b.scale_bytes,
                    b.hash_ids,
                    b.num_tokens,
                    b.out,
                    n,
                    clear_tail,
                )
                results.append(b.out)
                continue
            b.disk_table._require_open()
            with b.disk_table._cache.transaction():
                b.disk_table._cache.stage_rows_host(hid, n * NUM_HEADS)
                b.disk_table._cache._read_staged(n * NUM_HEADS)
                lookup_op(
                    b.plan.handle,
                    b.weight,
                    b.scale_bytes,
                    b.hash_ids,
                    b.num_tokens,
                    b.out,
                    n,
                    clear_tail,
                )
            results.append(b.out)
        return tuple(results)

    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from contextlib import ExitStack  # noqa: PLC0415

    pool = _READ_POOL
    if pool is None:
        pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="b12x-engram-read"
        )
        from b12x.sequence.engram import _impl as _impl_module

        _impl_module._READ_POOL = pool
    with ExitStack() as stack:
        caches = {}
        for i in disk:
            bindings[i].disk_table._require_open()
            caches[i] = stack.enter_context(
                bindings[i].disk_table._cache.transaction()
            )
            caches[i].stage_rows_host(host_ids[i], counts[i] * NUM_HEADS)
        futures = [
            pool.submit(caches[i]._read_staged, counts[i] * NUM_HEADS)
            for i in disk[1:]
        ]
        try:
            caches[disk[0]]._read_staged(counts[disk[0]] * NUM_HEADS)
        finally:
            errors = [f.exception() for f in futures]
        for error in errors:
            if error is not None:
                raise error
        for b, n in zip(bindings, counts):
            lookup_op(
                b.plan.handle,
                b.weight,
                b.scale_bytes,
                b.hash_ids,
                b.num_tokens,
                b.out,
                n,
                clear_tail,
            )
    return tuple(b.out for b in bindings)


# ---------------------------------------------------------------------------
# Independent port of the runner's two input kernels, for the composition
# self-test / golden test. Written directly from the engine listings
# (gpu/input_batch.py: _prepare_prefill_inputs_kernel, lines 276-322, and
# _combine_sampled_and_draft_tokens_kernel, lines 410-468): prefill kernel
# fills the WHOLE query span of a request with num_computed < prefill_len
# from all_token_ids; combine overwrites the logits rows (last sampled token
# at logits_start when first_logit_seq_pos >= prefill_len, then the drafts at
# the span tail) for requests with seq_len > prefill_len.
# ---------------------------------------------------------------------------


def _kernel_port_compose(
    padded_rows, starts, idx, nc, pl, all_rows, row_base, row_width,
    num_drafts, drafts_rows,
):
    """(ids, ok) — the kernels' behavior on the shapes the host path admits,
    reading the SAME sliced all_token_ids mirror (batch row r == req-state
    row idx[r], absolute columns [row_base[r], row_base[r]+row_width[r])).

    Returns ok=False for exactly the shapes compose_step_inputs rejects
    (anything where the exact host-side source of a row cannot be proven).
    """
    starts = np.asarray(starts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    nc = np.asarray(nc, dtype=np.int64)
    pl = np.asarray(pl, dtype=np.int64)
    row_base = np.asarray(row_base, dtype=np.int64)
    row_width = np.asarray(row_width, dtype=np.int64)
    num_reqs = len(starts) - 1
    spec = num_drafts is not None
    if spec:
        num_drafts = np.asarray(num_drafts, dtype=np.int64)
    ids = np.zeros(int(padded_rows), dtype=np.int64)
    num_new_sampled = 1

    for r in range(num_reqs):
        qs, qe = int(starts[r]), int(starts[r + 1])
        q = qe - qs
        if q <= 0:
            continue
        rsi, c, p = int(idx[r]), int(nc[r]), int(pl[r])
        base, width = int(row_base[r]), int(row_width[r])
        lo = c - base
        nd = int(num_drafts[r]) if spec else 0
        num_logits = nd + num_new_sampled if (spec or c >= p) else 0
        seq_len = c + q

        if seq_len <= p:
            # combine early-returns; the prefill kernel (nc < prefill_len
            # here by construction) filled the span from all_token_ids.
            hi = lo + q
            if not (c < p and lo >= 0 and hi <= width):
                return ids, False
            ids[qs:qe] = all_rows[r, lo:hi]
            continue

        # Sampling/verify request. The host path admits ONLY caught-up
        # spans here (c >= p): finish-prefill sampling spans rely on the
        # combine gate / prefill-fill order in ways compose_step_inputs
        # deliberately refuses to guess, so reject them exactly as it does.
        if c < p:
            return ids, False
        first_logit_seq_pos = seq_len - num_logits
        logits_start = qe - num_logits
        if num_new_sampled > 0 and first_logit_seq_pos >= p:
            # GPU writes last_sampled_tokens[rsi]; the caught-up invariant
            # (nc == total_len - 1) makes it all_token_ids[rsi, nc]. Accept
            # only shapes where first_logit_seq_pos == nc (true for every
            # caught-up span: first = c + q - num_logits and admitted
            # shapes have q == num_logits).
            if first_logit_seq_pos != c or lo < 0 or lo >= width:
                return ids, False
            ids[logits_start] = all_rows[r, lo]
        if nd > 0:
            if nd > drafts_rows.shape[1] or rsi >= drafts_rows.shape[0]:
                return ids, False
            # Kernel reads the REQ-STATE row (draft_tokens_ptr +
            # req_state_idx * draft_tokens_stride, input_batch.py:460-463).
            ids[qe - nd : qe] = drafts_rows[rsi, :nd]
    return ids, True


# ---------------------------------------------------------------------------
# Adversarial self-test (pure; python3 host_hash.py --selftest)
# ---------------------------------------------------------------------------


def _selftest_geometry(rng):
    """Deterministic small geometry: 24 primes near a base, odd multipliers."""
    base = 1_000_003
    primes = []
    cur = base
    while len(primes) < NUM_HEADS:
        cur += 2
        if all(cur % p for p in range(3, int(cur**0.5) + 1, 2)):
            primes.append(cur)
    offsets = [i * 1000 for i in range(NUM_HEADS)]
    mult = [int(v) for v in rng.integers(1, 1 << 40, size=4) * 2 + 1]
    return mult, primes, offsets


def _selftest_case(name, tokens, mask, starts, slots, history_raw, token_map, vocab,
                  mult, primes, offsets, padded=None, quiet=False):
    """Compare vectorized hash_layer vs the reference port on one batch."""
    live = starts[-1]
    padded = live if padded is None else padded
    got = hash_layer(
        tokens, mask, starts, history_raw, token_map, vocab, mult, primes,
        offsets, padded,
    )
    # Reference consumes the compressed history the rule above produces.
    hc = compress_history(history_raw, token_map, vocab)
    # reference.py indexes token_map with Python semantics (token_map[-1] is
    # the last element) and assumes in-vocab inputs; the GPU _compress marks
    # out-of-vocab raw ids dead. Substitute an in-vocab id with keep=False on
    # those lanes — the compressed value (-1) is identical either way.
    toks_list = [int(t) for t in tokens[:live]]
    mask_list = [
        bool(m) and 0 <= int(t) < vocab for t, m in zip(tokens[:live], mask[:live])
    ]
    ref_tokens = [t if 0 <= t < vocab else 0 for t in toks_list]
    ref_rows = hash_reference_port(
        ref_tokens,
        mask_list,
        list(starts),
        list(slots),
        [list(row) for row in hc],
        list(token_map),
        list(mult),
        list(primes),
        list(offsets),
    )
    got_live = got[:live]
    ref = np.asarray(ref_rows, dtype=np.int64).reshape(live, NUM_HEADS)
    if not np.array_equal(got_live, ref):
        bad = np.argwhere(got_live != ref)[:5]
        raise AssertionError(
            f"selftest[{name}]: mismatch at {bad.tolist()}; "
            f"got {got_live[bad[0][0], bad[0][1]] if len(bad) else None} "
            f"want {ref[bad[0][0], bad[0][1]] if len(bad) else None}"
        )
    if padded > live and not bool((got[live:] == -1).all()):
        raise AssertionError(f"selftest[{name}]: padded tail not -1")
    if not quiet:
        print(f"  ok  {name:38s} rows={live} padded={padded}")


def selftest(verbose=True):
    rng = np.random.default_rng(0)
    mult, primes, offsets = _selftest_geometry(rng)
    vocab = 64
    # Identity-ish token map with a distinctive pad (index 2).
    token_map = np.arange(vocab, dtype=np.int64)
    sentinel = IMAGE_SENTINEL

    cases = []

    # 1. dead tokens (-1) interleaved: blocked must stick
    cases.append((
        "dead -1 tokens in query",
        [5, -1, 7, 9, 11, 3], [True] * 6, [0, 6], [0],
        np.full((1, 3), -1, dtype=np.int64),
    ))
    # 2. pad id as a real token (compressed == token_map[2])
    cases.append((
        "pad id token",
        [2, 2, 2, 2, 2, 2], [True] * 6, [0, 6], [0],
        np.full((1, 3), -1, dtype=np.int64),
    ))
    # 3. image-sentinel-masked tokens (keep=False lanes)
    cases.append((
        "sentinel-masked tokens",
        [5, sentinel, 7, sentinel, 9, 3], [t != sentinel for t in [5, sentinel, 7, sentinel, 9, 3]],
        [0, 6], [0], np.full((1, 3), -1, dtype=np.int64),
    ))
    # 4. history-prefilled first positions (3 real, partial, empty)
    for hn, hist in (
        ("history 3 real", np.array([[10, 11, 12]], dtype=np.int64)),
        ("history partial", np.array([[10, -1, -1]], dtype=np.int64)),
        ("history empty", np.array([[-1, -1, -1]], dtype=np.int64)),
        ("history sentinel row", np.array([[sentinel, 11, 12]], dtype=np.int64)),
    ):
        cases.append((
            hn, [5, 6, 7], [True] * 3, [0, 3], [0], hist,
        ))
    # 5. single-token requests (history supplies all lookback)
    cases.append((
        "single-token requests x3",
        [5, 6, 7], [True] * 3, [0, 1, 2, 3], [0, 1, 2],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
    ))
    # 6. empty request between two live ones
    cases.append((
        "empty request mid-batch",
        [5, 6, 7, 8, 9], [True] * 5, [0, 2, 2, 5], [0, 1, 2],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
    ))
    # 7. out-of-vocab raw ids in query and history
    cases.append((
        "out-of-vocab ids",
        [vocab + 5, 3, 4, 5, 6], [True] * 5, [0, 5], [0],
        np.array([[vocab + 9, 2, 3]], dtype=np.int64),
    ))
    # 8. multi-request mixed lengths + padded tail
    cases.append((
        "mixed lengths + padded tail",
        [1, 2, 3, 4, 5, 6, 7, 8], [True] * 8, [0, 3, 7, 8], [0, 1, 2],
        np.array([[9, 10, 11], [-1, 1, 2], [3, 4, sentinel]], dtype=np.int64),
        12,
    ))
    # 9. spec-step shape: bonus row + k drafts after a rejection-built history
    cases.append((
        "spec verify rows (bonus + drafts)",
        [10, 21, 22, 23, 11, 31, 32], [True] * 7, [0, 4, 7], [0, 1],
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        9,
    ))

    if verbose:
        print("host_hash adversarial selftest (vectorized vs reference port):")
    for item in cases:
        name, tokens, mask, starts, slots, hist = item[:6]
        padded = item[6] if len(item) > 6 else None
        toks = np.asarray(tokens, dtype=np.int64)
        mk = np.asarray(mask, dtype=bool)
        st = np.asarray(starts, dtype=np.int64)
        _selftest_case(
            name, toks, mk, st, slots, hist, token_map, vocab,
            mult, primes, offsets, padded=padded, quiet=not verbose,
        )

    # 10. composition: admitted shapes vs an independent kernel port
    if verbose:
        print("host_hash compose_step_inputs (vs independent kernel port):")
    worst_failed = _composition_fuzz(
        rng, vocab, _kernel_port_compose, quiet=not verbose
    )
    if worst_failed:
        raise AssertionError("composition fuzz found mismatches (see output)")

    # 11. randomized fuzz: 200 batches, random spans/sentinels/dead/history
    if verbose:
        print("host_hash fuzz (200 random batches):")
    for trial in range(200):
        nreq = int(rng.integers(1, 7))
        lens = rng.integers(0, 9, size=nreq)
        live_total = int(lens.sum())
        if live_total == 0:
            lens[0] = 1
            live_total = 1
        starts = np.zeros(nreq + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        toks = rng.integers(0, vocab, size=live_total, dtype=np.int64)
        sprinkle = rng.random(live_total)
        toks[sprinkle < 0.05] = sentinel
        toks[(sprinkle >= 0.05) & (sprinkle < 0.10)] = -1
        toks[(sprinkle >= 0.10) & (sprinkle < 0.15)] = vocab + 3
        mk = rng.random(live_total) > 0.1
        hist = rng.integers(-1, vocab + 2, size=(nreq, 3)).astype(np.int64)
        _selftest_case(
            f"fuzz#{trial}", np.asarray(toks), mk, starts, list(range(nreq)),
            hist, token_map, vocab, mult, primes, offsets,
            padded=live_total + int(rng.integers(0, 4)), quiet=True,
        )
    if verbose:
        print("  ok  fuzz x200 (spans, sentinels, dead, oov, history)")

    # 12. int64 product bound with the production geometry shape
    cv = 99_092
    prod_bound = ((1 << 63) - 1) // cv // 2 * 2 + 1
    if prod_bound * (cv - 1) >= (1 << 63):
        raise AssertionError("int64 product bound violated for production geometry")
    if verbose:
        print("  ok  int64 product bound (production geometry)")
    return 0


def _derive_sliced_mirror(abs_rows, idx, nc, qlen):
    """Build what maybe_collect's gather D2H produces, from absolute rows.

    Mirror row r (BATCH row) = abs_rows[idx[r], base : base+width] with
    base = max(nc-3, 0), width = min(nc+q, row_len) - base; columns beyond
    each row's width are filled with a leak marker (they must never be
    read). This mirrors the pinned buffer's row math exactly.
    """
    num_reqs = len(idx)
    row_len = abs_rows.shape[1]
    base = np.maximum(np.asarray(nc, dtype=np.int64) - 3, 0)
    width = np.minimum(np.asarray(nc, dtype=np.int64) + np.asarray(qlen, dtype=np.int64), row_len) - base
    alloc = int(max(width.max(), 1)) + 2
    GARBAGE_W = int(abs_rows.max()) + 12345  # appears in NO valid column
    mirror = np.full((num_reqs, alloc), GARBAGE_W, dtype=abs_rows.dtype)
    for r in range(num_reqs):
        w = int(width[r])
        if w > 0:
            mirror[r, :w] = abs_rows[int(idx[r]), int(base[r]) : int(base[r]) + w]
    return mirror, base, width, GARBAGE_W


def _expected_ids_absolute(abs_rows, idx, nc, pl, starts, num_drafts, drafts_rows):
    """Inline ground truth from the ABSOLUTE req-state rows, independent of
    both compose_step_inputs and the kernel port (admitted shapes only)."""
    num_reqs = len(starts) - 1
    spec = num_drafts is not None
    ids = np.zeros(int(starts[-1]), dtype=np.int64)
    ok = True
    for r in range(num_reqs):
        qs, qe = int(starts[r]), int(starts[r + 1])
        q = qe - qs
        if q <= 0:
            continue
        rsi, c, p = int(idx[r]), int(nc[r]), int(pl[r])
        nd = int(num_drafts[r]) if spec else 0
        if c + q <= p:
            if c + q > abs_rows.shape[1]:
                ok = False
            else:
                ids[qs:qe] = abs_rows[rsi, c : c + q]
        elif (not spec) and c >= p and q == 1:
            if c >= abs_rows.shape[1]:
                ok = False
            else:
                ids[qs] = abs_rows[rsi, c]
        elif spec and c >= p and q == nd + 1:
            if c >= abs_rows.shape[1]:
                ok = False
            else:
                ids[qs] = abs_rows[rsi, c]
                if nd:
                    ids[qs + 1 : qe] = drafts_rows[rsi, :nd]
        else:
            ok = False
    return ids, ok


def _composition_fuzz(rng, vocab, port_compose, quiet=False):
    """Fuzz compose_step_inputs on the SLICED mirror interface against (a)
    the independent kernel port and (b) an inline ground truth computed from
    the ABSOLUTE req-state rows — with NON-IDENTITY batch->req-state
    mappings (shuffled + pop-off-end, matching free_indices.pop()), ragged
    per-request q, garbage in unmapped rows AND beyond each mirror row's
    width, and the gather-path boundary shapes (nc < 3 lookback padding,
    exact-fit row ends, clipped row_len -> reject).
    """
    failures = 0
    trials = 200
    identity_trials = 0
    boundary_trials = 0
    for trial in range(trials):
        nreq = int(rng.integers(1, 6))
        max_prompt = 24
        # Req-state pool larger than the batch; garbage rows in between.
        max_rows = nreq + int(rng.integers(1, 5))
        mapping_kind = rng.integers(0, 3)
        if mapping_kind == 0 and trial > 0:
            idx = np.arange(nreq, dtype=np.int64)  # rare identity control
            identity_trials += 1
        elif mapping_kind == 1:
            # free_indices.pop() off the end: descending-slot assignment.
            idx = np.arange(max_rows - 1, max_rows - nreq - 1, -1, dtype=np.int64)
        else:
            idx = rng.permutation(max_rows)[:nreq].astype(np.int64)
        mapped = set(int(x) for x in idx)
        unmapped = [s for s in range(max_rows) if s not in mapped]
        GARBAGE_A, GARBAGE_D = vocab + 101, vocab + 202  # leak markers

        pl = rng.integers(2, max_prompt, size=nreq).astype(np.int64)
        acc = rng.integers(0, 10, size=nreq)  # accepted counts (rejections vary)
        # Boundary knobs: sometimes nc < 3 (fresh request: lookback -1s),
        # sometimes nc+q rides the absolute row end exactly or past it.
        boundary = rng.integers(0, 4)
        boundary_trials += boundary in (1, 2, 3)
        if boundary == 1:
            acc[:] = 0
            pl[:] = rng.integers(2, 4, size=nreq)  # nc = pl <= 3
        cols = int((pl + acc + 8).max()) + 4
        row_len = cols
        if boundary == 2:
            # exact fit: zero acceptance, row ends exactly at max(pl), and
            # chunks ride nc+q == pl == row_len for the max-pl request.
            acc[:] = 0
            row_len = int(pl.max())
        elif boundary == 3:
            # clip: the row ends at/below some caught-up nc — reads that
            # would cross it must reject (never read garbage past the end).
            row_len = int((pl + acc).max()) - int(rng.integers(0, 2))
        row_len = max(int(row_len), 4)
        abs_rows = rng.integers(0, vocab, size=(max_rows, max(cols, row_len))).astype(np.int64)
        abs_rows[unmapped, :] = GARBAGE_A  # never read if indexing is right
        draft_rows = rng.integers(0, vocab, size=(max_rows, 5)).astype(np.int64)
        draft_rows[unmapped, :] = GARBAGE_D

        specs = []
        lens = []
        nds = []
        ncs = []
        for r in range(nreq):
            if boundary == 2:
                # exact fit for EVERY request: all chunks, pl == row_len,
                # nc + q == row_len (mirror width ends exactly at the read).
                kind = 0
                pl[r] = row_len
                q = int(rng.integers(1, max(1, pl[r] - 1) + 1))
                nc_r = int(pl[r] - q)
                nd = 0
            else:
                kind = rng.integers(0, 3)
                if kind == 0:  # mid-prompt chunk (spec or plain): nc < pl
                    q = int(rng.integers(1, max(1, pl[r] - 1) + 1))
                    nc_r = int(rng.integers(0, max(1, pl[r] - q)))
                    nd = 0
                else:
                    # caught-up (verify or plain decode): nc == total_len - 1
                    nc_r = int(pl[r] + acc[r])
                    if kind == 1:
                        q = int(rng.integers(1, 6))
                        nd = q - 1
                    else:
                        q, nd = 1, 0
                    if boundary == 1:
                        # fresh request: nc <= 3 exercises the lookback -1s
                        nc_r = int(rng.integers(1, 4))
            specs.append((kind, q, nd))
            lens.append(q)
            nds.append(nd)
            ncs.append(nc_r)
        nds = np.asarray(nds, dtype=np.int64)
        nc = np.asarray(ncs, dtype=np.int64)
        starts = np.zeros(nreq + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        padded = int(starts[-1]) + int(rng.integers(0, 3))
        num_drafts = nds if any(n > 0 for n in nds) else None
        is_spec = num_drafts is not None
        if is_spec:
            # plain decode must not be labelled spec: give those drafts 0
            for r in range(nreq):
                if specs[r][0] == 2:
                    num_drafts[r] = 0
        # Derive the sliced mirror exactly as the gather D2H would build it
        # (bounded by the SAME row_len the absolute rows have).
        abs_rows = abs_rows[:, :row_len]
        qlen = np.asarray(lens, dtype=np.int64)
        mirror, base, width, GARBAGE_W = _derive_sliced_mirror(abs_rows, idx, nc, qlen)
        got_ids, ok = compose_step_inputs(
            padded, starts, idx, nc, pl, mirror, base, width,
            num_drafts if is_spec else None, draft_rows if is_spec else None,
        )
        want, want_ok = port_compose(
            padded, starts, idx, nc, pl, mirror, base, width,
            num_drafts if is_spec else None, draft_rows if is_spec else None,
        )
        exp_ids, exp_ok = _expected_ids_absolute(
            abs_rows, idx, nc, pl, starts,
            num_drafts if is_spec else None, draft_rows if is_spec else None,
        )
        if ok != want_ok or (ok and not np.array_equal(got_ids, want)):
            failures += 1
            if not quiet:
                print(f"    composition fuzz#{trial} PORT MISMATCH ok={ok} want_ok={want_ok}")
        if ok != exp_ok or (ok and exp_ok and not np.array_equal(got_ids[: len(exp_ids)], exp_ids)):
            failures += 1
            if not quiet:
                print(f"    composition fuzz#{trial} GROUND-TRUTH MISMATCH ok={ok} exp_ok={exp_ok}")
        # Garbage-never-read proofs: unmapped rows and beyond-width columns.
        if ok and (GARBAGE_A in got_ids or GARBAGE_D in got_ids or GARBAGE_W in got_ids):
            failures += 1
            if not quiet:
                print(f"    composition fuzz#{trial} GARBAGE LEAK (wrong row/column read)")
    if not quiet:
        print(
            f"  ok  composition fuzz x{trials} (sliced mirror; non-identity idx "
            f"shuffled+pop-off-end; ragged q; boundary nc<3/exact-fit/clip; "
            f"{boundary_trials} boundary trials, {identity_trials} identity controls)"
        )
    return failures


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
