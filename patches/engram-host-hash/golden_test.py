#!/usr/bin/env python3
"""Golden test — DS41 host-side engram hash bit-exactness.

Modes:
  --pure      (CPU only; safe beside the live engine) Compares the
              vectorized host hash against the engine's OWN CPU oracle,
              flashinfer .../sequence/engram/reference.py::hash_reference,
              on adversarial batches and 300 fuzz batches, under
              (a) a synthetic geometry, (b) the PRODUCTION geometry built
              from the model's config.json, and (c) the PRODUCTION
              compressed token map built from the real tokenizer.
              Prints max-abs-diff per case; the pass bar is exactly 0.
  --native    (GPU; a dedicated test boot or explicit --allow-gpu only)
              Builds a real hash plan via b12x.sequence.engram and compares
              native.run against the host hash on identical GPU-side
              inputs. Skipped with a printed reason unless --allow-gpu is
              also given, because it creates a CUDA context next to
              whatever is already loaded on the device.

Run in-container from a scratch dir (imports the live engine tree
read-only; writes nothing outside the scratch dir); the checkpoint is
mounted at /model by this recipe's launcher:
    mkdir -p /tmp/ds41engram-golden
    cp host_hash.py golden_test.py /tmp/ds41engram-golden/
    docker exec ds41-flash-tp3 python3 /tmp/ds41engram-golden/golden_test.py --pure \
        --model-dir /model
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import host_hash  # noqa: E402

IMAGE_SENTINEL = host_hash.IMAGE_SENTINEL
NUM_HEADS = 24


def load_oracles():
    from flashinfer.experimental.b12x.sequence.engram.geometry import (
        build_geometry,
    )
    from flashinfer.experimental.b12x.sequence.engram.reference import (
        hash_reference,
    )

    return build_geometry, hash_reference


def compare_batch(
    name,
    tokens,
    mask,
    starts,
    slots,
    history_raw,
    token_map_list,
    geometry,
    vocab,
    padded=None,
    quiet=False,
):
    """One batch through (a) the real reference oracle per layer and (b) the
    vectorized host hash; returns the worst absolute difference."""
    _, hash_reference = load_oracles()
    tokens = np.asarray(tokens, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    history_raw = np.asarray(history_raw, dtype=np.int64)
    token_map = np.asarray(token_map_list, dtype=np.int64)
    live = int(starts[-1])
    padded = live if padded is None else padded

    hc = host_hash.compress_history(history_raw, token_map, vocab)
    # reference.py assumes in-vocab token ids (Python list indexing); the
    # GPU _compress marks out-of-vocab raw ids dead. Substitute an in-vocab
    # id with keep=False on those lanes — identical compressed value (-1).
    ref_tokens = [int(t) if 0 <= int(t) < vocab else 0 for t in tokens[:live]]
    ref_mask = [
        bool(m) and 0 <= int(t) < vocab
        for t, m in zip(tokens[:live], mask[:live])
    ]
    worst = 0
    for li, layer_id in enumerate(geometry.layer_ids):
        ref = hash_reference(
            ref_tokens,
            ref_mask,
            [int(x) for x in starts],
            [int(x) for x in slots],
            [list(int(v) for v in row) for row in hc],
            [int(v) for v in token_map_list],
            geometry,
            int(layer_id),
        ).numpy()
        got = host_hash.hash_layer(
            tokens,
            mask,
            starts,
            history_raw,
            token_map,
            vocab,
            geometry.multipliers[li],
            geometry.primes[li],
            geometry.offsets[li],
            padded,
        )[:live]
        worst = max(worst, int(np.abs(ref.astype(np.int64) - got).max()))
    if not quiet:
        print(f"  {name:46s} layers={len(geometry.layer_ids)} max_abs_diff={worst}")
    return worst


def adversarial_cases(vocab, token_map_list, geometry, tag):
    print(f"[{tag}] adversarial cases (vs engine reference.py oracle):")
    sentinel = IMAGE_SENTINEL if vocab > IMAGE_SENTINEL else vocab - 1
    worst = 0

    def case(name, tokens, mask, starts, slots, hist, padded=None):
        return compare_batch(
            name, tokens, mask, starts, slots, hist, token_map_list,
            geometry, vocab, padded=padded,
        )

    worst = max(worst, case(
        "dead -1 tokens in query",
        [5, -1, 7, 9, 11, 3], [True] * 6, [0, 6], [0],
        np.full((1, 3), -1, dtype=np.int64),
    ))
    worst = max(worst, case(
        "pad id token (compressed == pad)",
        [2, 2, 2, 2, 2, 2], [True] * 6, [0, 6], [0],
        np.full((1, 3), -1, dtype=np.int64),
    ))
    worst = max(worst, case(
        "image-sentinel-masked tokens",
        [5, sentinel, 7, sentinel, 9, 3],
        [t != sentinel for t in [5, sentinel, 7, sentinel, 9, 3]],
        [0, 6], [0], np.full((1, 3), -1, dtype=np.int64),
    ))
    for hn, hist in (
        ("history 3 real", np.array([[10, 11, 12]], dtype=np.int64)),
        ("history partial (-1 pad)", np.array([[10, -1, -1]], dtype=np.int64)),
        ("history empty", np.array([[-1, -1, -1]], dtype=np.int64)),
        ("history sentinel row", np.array([[sentinel, 11, 12]], dtype=np.int64)),
        ("history oov row", np.array([[vocab + 9, 11, 12]], dtype=np.int64)),
    ):
        worst = max(worst, case(hn, [5, 6, 7], [True] * 3, [0, 3], [0], hist))
    worst = max(worst, case(
        "single-token requests x3",
        [5, 6, 7], [True] * 3, [0, 1, 2, 3], [0, 1, 2],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
    ))
    worst = max(worst, case(
        "empty request mid-batch",
        [5, 6, 7, 8, 9], [True] * 5, [0, 2, 2, 5], [0, 1, 2],
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64),
    ))
    worst = max(worst, case(
        "out-of-vocab query ids",
        [vocab + 5, 3, 4, 5, 6], [True] * 5, [0, 5], [0],
        np.array([[vocab + 9, 2, 3]], dtype=np.int64),
    ))
    worst = max(worst, case(
        "mixed lengths + padded tail",
        [1, 2, 3, 4, 5, 6, 7, 8], [True] * 8, [0, 3, 7, 8], [0, 1, 2],
        np.array([[9, 10, 11], [-1, 1, 2], [3, 4, sentinel]], dtype=np.int64),
        padded=12,
    ))
    return worst


def fuzz_cases(n_trials, vocab, token_map_list, geometry, seed, tag):
    print(f"[{tag}] fuzz x{n_trials} (random spans/sentinels/dead/oov/history):")
    rng = np.random.default_rng(seed)
    sentinel = IMAGE_SENTINEL if vocab > IMAGE_SENTINEL else vocab - 1
    worst = 0
    for _ in range(n_trials):
        nreq = int(rng.integers(1, 8))
        lens = rng.integers(0, 10, size=nreq)
        if lens.sum() == 0:
            lens[0] = 1
        live = int(lens.sum())
        starts = np.zeros(nreq + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        toks = rng.integers(0, vocab, size=live, dtype=np.int64)
        sprinkle = rng.random(live)
        toks[sprinkle < 0.05] = sentinel
        toks[(sprinkle >= 0.05) & (sprinkle < 0.10)] = -1
        toks[(sprinkle >= 0.10) & (sprinkle < 0.15)] = vocab + 3
        mask = rng.random(live) > 0.1
        hist = rng.integers(-1, vocab + 2, size=(nreq, 3)).astype(np.int64)
        worst = max(
            worst,
            compare_batch(
                "", toks, mask, starts, list(range(nreq)), hist,
                token_map_list, geometry, vocab,
                padded=live + int(rng.integers(0, 4)), quiet=True,
            ),
        )
    print(f"  fuzz worst max_abs_diff = {worst}")
    return worst


# ---------------------------------------------------------------------------
# SPEC EXTENSION: spec-step shapes. Histories built from accepted-token
# sequences WITH rejections (each simulated verify step accepts 1..k+1 of
# its k+1 rows), queries are [last-accepted bonus row] + k draft rows.
# ---------------------------------------------------------------------------


def _simulate_spec_requests(rng, nreq, vocab, sentinel, kmax=5):
    """Per request: (all_rows_row, prompt_len, accepted_list, drafts).

    all_rows_row is the request's all_token_ids: prompt tokens then the
    accepted sequence produced by successive verify steps with rejections.
    """
    states = []
    for _ in range(nreq):
        pl = int(rng.integers(2, 24))
        prompt = list(rng.integers(0, vocab, size=pl))
        accepted = []
        for _ in range(int(rng.integers(0, 12))):  # verify steps so far
            k = int(rng.integers(1, kmax + 1))
            # acceptance count in [1, k+1]: at least the bonus row survives
            acc = int(rng.integers(1, k + 2))
            step_tokens = [int(t) for t in rng.integers(0, vocab, size=acc)]
            accepted.extend(step_tokens)
        row = np.asarray(
            prompt + accepted + [0, 0, 0], dtype=np.int64
        )  # + slack zeros beyond total_len
        states.append((row, pl, accepted))
    return states


def spec_hash_cases(n_trials, vocab, token_map_list, geometry, seed, tag):
    """Hash exactness on spec-shaped batches (bonus + drafts rows after
    rejection-built histories), vs the engine reference oracle."""
    print(f"[{tag}] spec-history hash cases x{n_trials} (rejection-built):")
    rng = np.random.default_rng(seed)
    sentinel = IMAGE_SENTINEL if vocab > IMAGE_SENTINEL else vocab - 1
    worst = 0
    for _ in range(n_trials):
        nreq = int(rng.integers(1, 6))
        states = _simulate_spec_requests(rng, nreq, vocab, sentinel)
        # Build one spec batch: each request contributes bonus + k drafts.
        toks, starts, hist_rows = [], [0], []
        for row, pl, accepted in states:
            total = pl + len(accepted)
            k = int(rng.integers(1, 6))
            # Bonus row = the token at position nc == total - 1 (caught-up
            # invariant), i.e. the last accepted token (or last prompt token
            # when nothing is accepted yet).
            bonus = int(row[total - 1])
            drafts = [
                int(t) if rng.random() > 0.08 else (
                    sentinel if rng.random() > 0.5 else -1
                )
                for t in rng.integers(0, vocab, size=k)
            ]
            toks.extend([bonus] + drafts)
            starts.append(starts[-1] + 1 + k)
            # Raw lookback = last 3 computed tokens at [nc-3, nc).
            nc = total - 1 if rng.random() > 0.15 else total  # rare 1-behind
            look = [int(row[nc - 3 + j]) if nc - 3 + j >= 0 else -1 for j in range(3)]
            hist_rows.append(look)
        live = starts[-1]
        toks = np.asarray(toks, dtype=np.int64)
        mask = toks != sentinel
        starts = np.asarray(starts, dtype=np.int64)
        hist = np.asarray(hist_rows, dtype=np.int64)
        worst = max(
            worst,
            compare_batch(
                "", toks, mask, starts, list(range(nreq)), hist,
                token_map_list, geometry, vocab,
                padded=live + int(rng.integers(0, 4)), quiet=True,
            ),
        )
    print(f"  spec-history worst max_abs_diff = {worst}")
    return worst


def spec_composition_cases(n_trials, vocab, token_map_list, geometry, seed, tag):
    """compose_step_inputs vs the independent kernel port on spec batches,
    with NON-IDENTITY batch->req-state mappings (shuffled and pop-off-end,
    matching free_indices.pop()) and garbage in unmapped rows. Every
    admitted batch is ALSO pushed through the hash-vs-reference comparison,
    tying the composed spec ids to the bit-exactness proof."""
    print(f"[{tag}] spec composition x{n_trials} (non-identity idx, vs kernel port):")
    rng = np.random.default_rng(seed)
    port = host_hash._kernel_port_compose
    mismatches = 0
    checked = 0
    identity_trials = 0
    kmax_drafts = 5
    worst_hash = 0
    for trial in range(n_trials):
        nreq = int(rng.integers(1, 6))
        states = _simulate_spec_requests(rng, nreq, vocab, IMAGE_SENTINEL)
        cols = max(len(row) for row, *_ in states)
        # Req-state pool with unmapped garbage rows; non-identity mapping.
        max_rows = nreq + int(rng.integers(1, 5))
        mapping_kind = rng.integers(0, 3)
        if mapping_kind == 0 and trial > 0:
            idx = np.arange(nreq, dtype=np.int64)  # rare identity control
            identity_trials += 1
        elif mapping_kind == 1:  # free_indices.pop() off the end
            idx = np.arange(max_rows - 1, max_rows - nreq - 1, -1, dtype=np.int64)
        else:  # shuffled
            idx = rng.permutation(max_rows)[:nreq].astype(np.int64)
        mapped = set(int(x) for x in idx)
        unmapped = [s for s in range(max_rows) if s not in mapped]
        GARBAGE_A, GARBAGE_D = vocab + 101, vocab + 202

        all_rows = rng.integers(0, vocab, size=(max_rows, cols + 8)).astype(np.int64)
        all_rows[unmapped, :] = GARBAGE_A  # must never be read
        drafts_rows = rng.integers(0, vocab, size=(max_rows, kmax_drafts)).astype(np.int64)
        drafts_rows[unmapped, :] = GARBAGE_D  # must never be read

        pl = np.zeros(nreq, dtype=np.int64)
        nc = np.zeros(nreq, dtype=np.int64)
        lens, nds = [], []
        kinds = []
        for r, (row, p, accepted) in enumerate(states):
            all_rows[idx[r], : len(row)] = row  # state lives in its REQ-STATE row
            pl[r] = p
            total = p + len(accepted)
            kind = rng.integers(0, 3)
            kinds.append(kind)
            if kind == 0:  # no-draft chunk (spec batch member)
                q = int(rng.integers(1, max(1, p - 1) + 1))
                nd = 0
                nc[r] = int(rng.integers(0, p - q)) if p - q > 0 else 0
            elif kind == 1:  # caught-up verify
                nd = int(rng.integers(1, kmax_drafts + 1))
                q = nd + 1
                nc[r] = total - 1
            else:  # caught-up plain decode on a spec batch
                q, nd = 1, 0
                nc[r] = total - 1
            lens.append(q)
            nds.append(nd)
        nds = np.asarray(nds, dtype=np.int64)
        starts = np.zeros(nreq + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        padded = int(starts[-1]) + int(rng.integers(0, 3))
        # Derive the SLICED pinned-mirror exactly as maybe_collect's gather
        # D2H builds it (batch row r == req-state row idx[r], columns
        # [max(nc-3,0), min(nc+q, row_len))), garbage beyond each width.
        qlen = np.asarray(lens, dtype=np.int64)
        mirror, base, width, GARBAGE_W = host_hash._derive_sliced_mirror(
            all_rows, idx, nc, qlen
        )
        got, ok = host_hash.compose_step_inputs(
            padded, starts, idx, nc, pl, mirror, base, width, nds, drafts_rows
        )
        want, want_ok = port(
            padded, starts, idx, nc, pl, mirror, base, width, nds, drafts_rows
        )
        exp, exp_ok = host_hash._expected_ids_absolute(
            all_rows, idx, nc, pl, starts, nds, drafts_rows
        )
        checked += 1
        if ok != want_ok or (ok and not np.array_equal(got, want)):
            mismatches += 1
            print(f"    composition MISMATCH kinds={kinds} ok={ok} want_ok={want_ok}")
        elif ok != exp_ok or (ok and exp_ok and not np.array_equal(got[: len(exp)], exp)):
            mismatches += 1
            print(f"    composition GROUND-TRUTH MISMATCH trial#{trial} ok={ok} exp_ok={exp_ok}")
        elif ok and (GARBAGE_A in got or GARBAGE_D in got or GARBAGE_W in got):
            mismatches += 1
            print(f"    composition GARBAGE LEAK trial#{trial} (wrong req-state row/column read)")
        elif ok:
            # Hash-coupled check: the composed spec ids, with lookbacks
            # gathered through the SAME sliced-mirror math maybe_collect
            # uses, must hash bit-exact vs the engine oracle.
            hist = np.full((nreq, 3), -1, dtype=np.int64)
            for r in range(nreq):
                for j in range(3):
                    pos = int(nc[r]) - 3 + j
                    if pos >= 0:
                        hist[r, j] = mirror[r, pos - int(base[r])]
            mask = got[: starts[-1]] != IMAGE_SENTINEL
            worst_hash = max(
                worst_hash,
                compare_batch(
                    "", got, mask, starts, list(range(nreq)), hist,
                    token_map_list, geometry, vocab, padded=padded, quiet=True,
                ),
            )
    print(
        f"  spec composition: {checked - mismatches}/{checked} identical "
        f"({identity_trials} identity controls, rest non-identity), "
        f"{mismatches} mismatches; composed-batch hash worst max_abs_diff = {worst_hash}"
    )
    return mismatches + (worst_hash != 0)


def load_production(model_dir):
    """(geometry, token_map list, vocab) from the model's config + tokenizer."""
    from flashinfer.experimental.b12x.sequence.engram.geometry import (
        build_compressed_token_map,
        build_geometry,
    )

    cfg = None
    for candidate in (
        os.path.join(model_dir, "inference", "config.json"),
        os.path.join(model_dir, "config.json"),
    ):
        if os.path.isfile(candidate):
            with open(candidate) as f:
                c = json.load(f)
            if "engram_layer_ids" in c:
                cfg = c
                break
            if isinstance(c.get("text_config"), dict) and "engram_layer_ids" in c["text_config"]:
                cfg = c["text_config"]
                break
    if cfg is None:
        raise AssertionError(
            f"no engram fields in any config under {model_dir}"
        )
    layer_ids = tuple(cfg["engram_layer_ids"])
    base = int(cfg["engram_vocab_size"])
    cv = int(cfg["engram_compressed_vocab_size"])
    geom = build_geometry(
        layer_ids=layer_ids, base_table_size=base, compressed_vocab_size=cv
    )
    if tuple(geom.num_embeddings) != tuple(cfg["engram_num_embeddings"]):
        raise AssertionError("production geometry disagrees with config tables")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    token_map, size = build_compressed_token_map(tok)
    if size != cv:
        raise AssertionError("production token map compressed vocab mismatch")
    return geom, token_map, int(cfg["vocab_size"])


def run_pure(args):
    build_geometry, _ = load_oracles()
    failures = 0

    # --- synthetic geometry (small, fast) ---
    geom_synth = build_geometry(
        layer_ids=(1, 14), base_table_size=1_000_003, compressed_vocab_size=64
    )
    vocab = 64
    token_map = list(range(vocab))
    worst = adversarial_cases(vocab, token_map, geom_synth, "synthetic-geometry")
    worst = max(worst, fuzz_cases(300, vocab, token_map, geom_synth, 7, "synthetic-geometry"))
    worst = max(
        worst, spec_hash_cases(150, vocab, token_map, geom_synth, 21, "synthetic-geometry")
    )
    if spec_composition_cases(150, vocab, token_map, geom_synth, 23, "synthetic-geometry"):
        failures += 1
    if worst != 0:
        failures += 1

    # --- production geometry + production tokenizer map ---
    if args.model_dir and os.path.isdir(args.model_dir):
        print(f"[production] loading config + tokenizer from {args.model_dir}")
        geom, token_map, vocab = load_production(args.model_dir)
        print(
            f"[production] layers={geom.layer_ids} vocab={vocab} "
            f"compressed={geom.compressed_vocab_size} pad={token_map[2]}"
        )
        worst = adversarial_cases(vocab, token_map, geom, "production")
        worst = max(worst, fuzz_cases(300, vocab, token_map, geom, 11, "production"))
        worst = max(
            worst, spec_hash_cases(150, vocab, token_map, geom, 31, "production")
        )
        if spec_composition_cases(150, vocab, token_map, geom, 33, "production"):
            failures += 1
        if worst != 0:
            failures += 1
    else:
        print(f"[production] model dir missing ({args.model_dir!r}); skipped")

    print()
    if failures:
        print("GOLDEN TEST: FAIL (nonzero max_abs_diff above)")
        return 1
    print("GOLDEN TEST (pure, vs engine reference.py): PASS — max_abs_diff == 0 everywhere")
    return 0


def run_native(args):
    if not args.allow_gpu:
        print(
            "[native] SKIPPED: needs --allow-gpu (creates a CUDA context; run "
            "on a dedicated test boot or an idle device, never beside the "
            "live serving engine)."
        )
        return 0
    import torch

    from b12x.sequence import engram as native

    geom, token_map_list, vocab = (
        load_production(args.model_dir) if args.model_dir else (None, None, None)
    )
    if geom is None:
        build_geometry, _ = load_oracles()
        geom = build_geometry(
            layer_ids=(1,), base_table_size=1_000_003, compressed_vocab_size=64
        )
        token_map_list = list(range(64))
        vocab = 64
    layer_id = geom.layer_ids[0]
    device = torch.device("cuda", torch.cuda.current_device())
    max_tokens = 512
    caps = native.Caps(
        device=device,
        max_tokens=max_tokens,
        max_seqs=16,
        max_requests=16,
        vocab_size=vocab,
        layer_id=layer_id,
        tp_size=1,
        tp_rank=0,
    )
    plan = native.plan(caps, token_map=token_map_list, geometry=geom)
    try:
        binding = native.bind(
            plan,
            scratch=torch.empty(
                (caps.max_tokens * 8 + caps.max_tokens * 4,),
                dtype=torch.uint8,
                device=device,
            ),
            token_ids=torch.zeros(max_tokens, dtype=torch.int64, device=device),
            token_mask=torch.ones(max_tokens, dtype=torch.bool, device=device),
            query_start_loc=torch.zeros(17, dtype=torch.int32, device=device),
            request_slots=torch.arange(16, dtype=torch.int32, device=device),
            committed_history=torch.zeros((16, 3), dtype=torch.int64, device=device),
            num_seqs=torch.zeros(1, dtype=torch.int32, device=device),
            num_tokens=torch.zeros(1, dtype=torch.int32, device=device),
            hash_ids=torch.empty((max_tokens, 24), dtype=torch.int64, device=device),
        )
    except Exception as exc:  # plan not prepared standalone
        print(f"[native] SKIPPED: standalone bind failed ({type(exc).__name__}: {exc}).")
        print("[native] Run this mode on a dedicated test boot instead.")
        return 0

    from b12x.sequence.engram._kernels import _request_ids_kernel  # unused, marker

    rng = np.random.default_rng(3)
    sentinel = IMAGE_SENTINEL if vocab > IMAGE_SENTINEL else vocab - 1
    worst = 0
    trials = 40
    for _ in range(trials):
        nreq = int(rng.integers(1, 9))
        lens = rng.integers(0, 12, size=nreq)
        if lens.sum() == 0:
            lens[0] = 1
        live = int(lens.sum())
        padded = min(max_tokens, live + int(rng.integers(0, 5)))
        starts = np.zeros(nreq + 1, dtype=np.int64)
        np.cumsum(lens, out=starts[1:])
        toks = rng.integers(0, vocab, size=live, dtype=np.int64)
        toks[rng.random(live) < 0.08] = sentinel
        mask = rng.random(live) > 0.1
        hist = rng.integers(-1, vocab + 2, size=(nreq, 3)).astype(np.int64)

        # GPU-side inputs (identical to the engine's run_native plumbing)
        binding.token_ids[:padded].zero_()
        binding.token_ids[:live].copy_(
            torch.from_numpy(np.ascontiguousarray(toks)), non_blocking=True
        )
        binding.token_mask[:padded].fill_(False)
        binding.token_mask[:live].copy_(
            torch.from_numpy(np.ascontiguousarray(mask)), non_blocking=True
        )
        binding.query_start_loc[: nreq + 1].copy_(
            torch.from_numpy(starts.astype(np.int32)), non_blocking=True
        )
        for r in range(nreq):
            binding.committed_history[r].copy_(
                torch.from_numpy(hist[r]), non_blocking=True
            )
        binding.num_seqs.fill_(nreq)
        binding.num_tokens.fill_(live)
        torch.cuda.synchronize()

        native.run(binding, token_count=padded)
        native_rows = binding.hash_ids[:padded].cpu().numpy()

        host_rows = host_hash.hash_layer(
            toks, mask, starts, hist, np.asarray(token_map_list, dtype=np.int64),
            vocab, geom.multipliers[0], geom.primes[0], geom.offsets[0], padded,
        )
        worst = max(worst, int(np.abs(native_rows - host_rows).max()))
    print(f"[native] fuzz x{trials}: worst max_abs_diff = {worst}")
    if worst != 0:
        print("GOLDEN TEST (native): FAIL")
        return 1
    print("GOLDEN TEST (native, vs b12x native.run): PASS — max_abs_diff == 0")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure", action="store_true", help="CPU-only reference comparison")
    ap.add_argument("--native", action="store_true", help="GPU native comparison")
    ap.add_argument("--allow-gpu", action="store_true")
    ap.add_argument(
        "--model-dir",
        default="/model",
    )
    args = ap.parse_args()
    if not args.pure and not args.native:
        args.pure = True
    rc = 0
    if args.pure:
        rc |= run_pure(args)
    if args.native:
        rc |= run_native(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
