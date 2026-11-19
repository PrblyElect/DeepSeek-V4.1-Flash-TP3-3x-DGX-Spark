#!/usr/bin/env python3
"""Install/revert/verify the DS41 session-prefix pin v2 (2026-11).

UNIVERSAL (measured, kept): every request's freed cached blocks are PINNED
for VLLM_DS41_PIN_TTL_S seconds (budget VLLM_DS41_PIN_MAX_BLOCKS) — no
priority tag, no opt-in ("launch model, resume 3 chats" just works).
The allocator moves pinned blocks to the free-queue TAIL, so they are
evicted LAST. A hard-pressure allocation still evicts them (the defer
scan is bounded), so the lane can never deadlock on its own pins. The
next turn of the same session then hits the prefix cache: TTFT ~1 s
instead of a full re-prefill. An already-pinned block only REFRESHES its
deadline (the turn-end fast path — a resident session's turn-end never
drops a victim); at cap, new admissions recycle (expired purge once,
then earliest-deadline victim per insert).

THREE visible-text anchors:
  block_pool.py P1  get_new_blocks: bounded front-scan moves unexpired
                    pinned blocks to the tail before popleft_n.
  block_pool.py P3  free_blocks: when the one-shot marker is armed (set
                    by the manager around EVERY request's free), record
                    the freed CACHED blocks as pinned (cap + recycle).
  kv_cache_manager.py P2 free(): arm the one-shot marker when
                    request.priority <= VLLM_DS41_PIN_PRIO_MAX
                    (default 1: untagged 0 and high -1 arm; only
                    explicit batch >= 2 opts out), clear after the
                    coordinator free.

File claims: block_pool.py + kv_cache_manager.py — disjoint from
the scheduler kits and the engram kit
(model files). Composes with both.

Discipline: exact-once anchors, compile-before-write, atomic write,
byte-exact revert, pristine-reconstruction md5 pins on --verify.

Usage: install.py --apply|--revert|--verify|--diff [--vllm-root DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import py_compile
import sys
from pathlib import Path

DEFAULT_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
BP_REL = "v1/core/block_pool.py"
KM_REL = "v1/core/kv_cache_manager.py"

MARK = "DS41-PIN"

PIN_BP = "3620e06cf46cf24d4a9b315f34b20ebf"
PIN_KM = "85941f40d2bcd3f484cc005227816606"

# --- block_pool.py, anchor P1: get_new_blocks (the defer) ---
ORIG_P1 = """\
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
"""
PATCHED_P1 = """\
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # --- DS41-PIN DEFER BEGIN (revert: restore original block) ---
        # : pinned blocks (an interactive session's freed prefix) are
        # moved to the free-queue TAIL so they are evicted LAST. Bounded
        # front-scan; expired pins stop deferring lazily (no sweep thread);
        # under hard pressure the scan simply finds nothing to move and the
        # pins evict like normal blocks (the lane can never deadlock).
        try:
            import time as _tp

            _ds41_pinned = getattr(self, "ds41_pinned_ids", None)
            if _ds41_pinned:
                _ds41_now = _tp.monotonic()
                _ds41_until = self.ds41_pin_until
                _ds41_scan = num_blocks + 64
                _ds41_node = self.free_block_queue.fake_free_list_head.next_free_block
                _ds41_moved = 0
                while _ds41_scan > 0 and _ds41_node is not None and _ds41_moved < num_blocks:
                    _ds41_blk = _ds41_node
                    _ds41_node = _ds41_node.next_free_block
                    if _ds41_blk.block_id in _ds41_pinned:
                        if _ds41_until.get(_ds41_blk.block_id, 0) > _ds41_now:
                            self.free_block_queue.remove(_ds41_blk)
                            self.free_block_queue.append(_ds41_blk)
                            _ds41_moved += 1
                        else:
                            _ds41_pinned.discard(_ds41_blk.block_id)
                    _ds41_scan -= 1
        except Exception:
            pass
        # --- DS41-PIN DEFER END ---

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
"""

# --- block_pool.py, anchor P3: free_blocks (the mark) ---
ORIG_P3 = """\
        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
"""
PATCHED_P3 = """\
        # --- DS41-PIN MARK BEGIN (revert: restore original block) ---
        # v2 (UNIVERSAL): when the one-shot marker is armed (the
        # manager sets it around EVERY request's free), record the freed
        # CACHED blocks as pinned for the marker's TTL. An ALREADY-pinned
        # block only REFRESHES its deadline (fast path, no cap dance) — a
        # resident session's turn-end never drops anyone. At cap, NEW
        # admissions recycle: purge lazily-expired pins once, then drop the
        # earliest-deadline (least-recently-refreshed) victim per insert —
        # a big session arriving at cap makes its own room, and agent churn
        # cannot starve newer sessions out of the set.
        try:
            _ds41_mark = getattr(self, "ds41_pin_mark_until", None)
            if _ds41_mark is not None:
                _ds41_purged = False
                for _ds41_b in blocks_to_evict_last:
                    if _ds41_b.block_hash is None:
                        continue
                    _ds41_bid = _ds41_b.block_id
                    if _ds41_bid in self.ds41_pinned_ids:
                        self.ds41_pin_until[_ds41_bid] = _ds41_mark
                        continue
                    if len(self.ds41_pinned_ids) >= self.ds41_pin_cap:
                        if not _ds41_purged:
                            _ds41_purged = True
                            _ds41_now = __import__("time").monotonic()
                            for _ds41_x in [
                                _b
                                for _b, _u in self.ds41_pin_until.items()
                                if _u <= _ds41_now
                            ]:
                                self.ds41_pinned_ids.discard(_ds41_x)
                                self.ds41_pin_until.pop(_ds41_x, None)
                        if (
                            len(self.ds41_pinned_ids) >= self.ds41_pin_cap
                            and self.ds41_pin_until
                        ):
                            _ds41_victim = min(
                                self.ds41_pin_until,
                                key=self.ds41_pin_until.get,
                            )
                            self.ds41_pinned_ids.discard(_ds41_victim)
                            self.ds41_pin_until.pop(_ds41_victim, None)
                        if len(self.ds41_pinned_ids) >= self.ds41_pin_cap:
                            break
                    self.ds41_pinned_ids.add(_ds41_bid)
                    self.ds41_pin_until[_ds41_bid] = _ds41_mark
        except Exception:
            pass
        # --- DS41-PIN MARK END ---
        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
"""

# --- kv_cache_manager.py, anchor P2: free() (arm/disarm) ---
ORIG_P2 = """\
        boundary_blocks = self._pop_boundary_blocks(request)
        self.block_pool.free_blocks(boundary_blocks)
        self.coordinator.free(request.request_id)
"""
PATCHED_P2 = """\
        # --- DS41-PIN ARM BEGIN (revert: restore original block) ---
        # v2 (UNIVERSAL, measured and kept): EVERY request's session blocks
        # survive its free — no priority tag, no opt-in. Explicitly-batch
        # traffic may still opt out via X-Vllm-Priority >=
        # VLLM_DS41_PIN_PRIO_MAX (default 1). The coordinator's frees
        # (below) mark the freed cached blocks pinned (TTL + cap + recycle
        # via env); disarm after.
        try:
            import os as _osm
            import time as _tm

            _ds41_prio_max = int(_osm.environ.get("VLLM_DS41_PIN_PRIO_MAX", "1"))
            _ds41_ttl = float(_osm.environ.get("VLLM_DS41_PIN_TTL_S", "1800"))
            _ds41_cap = int(_osm.environ.get("VLLM_DS41_PIN_MAX_BLOCKS", "6500"))
            if _ds41_ttl > 0 and _ds41_cap > 0 and getattr(request, "priority", 0) <= _ds41_prio_max:
                pool = self.block_pool
                if not hasattr(pool, "ds41_pinned_ids"):
                    pool.ds41_pinned_ids = set()
                    pool.ds41_pin_until = {}
                    pool.ds41_pin_cap = _ds41_cap
                pool.ds41_pin_cap = _ds41_cap
                pool.ds41_pin_mark_until = _tm.monotonic() + _ds41_ttl
        except Exception:
            pass
        # --- DS41-PIN ARM END ---
        boundary_blocks = self._pop_boundary_blocks(request)
        self.block_pool.free_blocks(boundary_blocks)
        self.coordinator.free(request.request_id)
        # --- DS41-PIN DISARM BEGIN (revert: restore original block) ---
        try:
            self.block_pool.ds41_pin_mark_until = None
        except Exception:
            pass
        # --- DS41-PIN DISARM END ---
"""

BP_BLOCKS = [(ORIG_P1, PATCHED_P1, "P1"), (ORIG_P3, PATCHED_P3, "P3")]
KM_BLOCKS = [(ORIG_P2, PATCHED_P2, "P2")]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _md5_str(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _read(f: Path) -> str:
    return f.read_text(encoding="utf-8")


def _write(f: Path, text: str) -> None:
    tmp = f.parent / (f.name + ".ds41pin.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, f)


def _state(src: str, blocks) -> str:
    verdicts = []
    for (orig, patched, _nm) in blocks:
        n_patched = src.count(patched)
        n_orig = src.count(orig)
        if n_patched == 1:
            verdicts.append("patched")
        elif n_patched == 0 and n_orig == 1:
            verdicts.append("original")
        else:
            verdicts.append("foreign")
    if "foreign" in verdicts or len(set(verdicts)) > 1:
        return "foreign"
    return verdicts[0]


def _reconstruct(src: str, blocks) -> str:
    out = src
    for (orig, patched, _) in blocks:
        if patched not in out:
            raise ValueError("patched block missing; cannot reconstruct")
        out = out.replace(patched, orig, 1)
    return out


def _apply_file(f: Path, blocks, pin: str, action: str) -> tuple[bool, str]:
    src = _read(f)
    state = _state(src, blocks)
    if action == "apply":
        if state == "patched":
            return True, f"apply: already present (idempotent) {f.name}"
        if state == "foreign":
            return False, f"refuse: foreign marker/anchor state in {f.name}"
        new_src = src
        for (orig, patched, _) in blocks:
            new_src = new_src.replace(orig, patched, 1)
        try:
            compile(new_src, str(f), "exec")
        except SyntaxError as exc:
            return False, f"refuse: patched {f.name} does not compile: {exc}"
        before = _md5(f)
        _write(f, new_src)
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, f"POST-WRITE py_compile FAILED ({f.name} left patched): {exc}"
        ok_pin = _md5_str(_reconstruct(_read(f), blocks)) == pin
        return ok_pin, (
            f"apply: patched {f.name} (md5 {before} -> {_md5(f)}; compile ok; "
            f"baseline-reconstruction pin {'MATCH' if ok_pin else 'MISMATCH'})"
        )
    if action == "revert":
        if state == "original":
            ok = _md5(f) == pin
            return ok, f"revert: already original {f.name} (pin {'MATCH' if ok else 'MISMATCH'})"
        if state == "foreign":
            return False, f"refuse: cannot revert foreign state in {f.name}"
        orig_src = _reconstruct(src, blocks)
        try:
            compile(orig_src, str(f), "exec")
        except SyntaxError as exc:
            return False, f"refuse: reverted text does not compile: {exc}"
        before = _md5(f)
        _write(f, orig_src)
        py_compile.compile(str(f), doraise=True)
        ok = _md5(f) == pin
        return ok, f"revert: byte-exact round trip {f.name} {'CONFIRMED' if ok else 'PIN MISMATCH'}"
    if state == "patched":
        try:
            pristine = _reconstruct(src, blocks)
        except ValueError as exc:
            return False, f"{f.name}: damaged block: {exc}"
        ok = _md5_str(pristine) == pin
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as exc:
            return False, f"{f.name}: py_compile FAIL: {exc}"
        return ok, f"{f.name}: patched; reconstruction pin {'MATCH' if ok else 'MISMATCH'}; py_compile ok"
    if state == "original":
        ok = _md5(f) == pin
        return ok, f"{f.name}: original; pin {'MATCH' if ok else 'MISMATCH'}"
    return False, f"{f.name}: FOREIGN"


def _print_diff() -> int:
    import difflib

    for (orig, patched, nm), rel in [
        (BP_BLOCKS[0], BP_REL),
        (BP_BLOCKS[1], BP_REL),
        (KM_BLOCKS[0], KM_REL),
    ]:
        print(f"=== {rel} anchor {nm} ===")
        print(
            "".join(
                difflib.unified_diff(
                    orig.splitlines(keepends=True),
                    patched.splitlines(keepends=True),
                    fromfile=f"(original {nm})",
                    tofile=f"(patched {nm})",
                    n=2,
                )
            ),
            end="",
        )
    print(f"pins: {BP_REL} {PIN_BP}; {KM_REL} {PIN_KM}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vllm-root", default=DEFAULT_VLLM)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--diff", action="store_true")
    a = ap.parse_args()

    if a.diff:
        return _print_diff()

    bp = Path(a.vllm_root) / BP_REL
    km = Path(a.vllm_root) / KM_REL
    for f in (bp, km):
        if not f.is_file():
            print(f"refuse: missing {f}", file=sys.stderr)
            return 2

    action = "apply" if a.apply else "revert" if a.revert else "verify"
    ok = True
    for f, blocks, pin in ((bp, BP_BLOCKS, PIN_BP), (km, KM_BLOCKS, PIN_KM)):
        file_ok, note = _apply_file(f, blocks, pin, action)
        print(note)
        ok = ok and file_ok
    print(f"{action}:", "OK" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
