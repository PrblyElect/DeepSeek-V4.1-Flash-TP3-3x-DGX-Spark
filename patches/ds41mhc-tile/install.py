#!/usr/bin/env python3
"""Install/revert/verify the DS41 mHC PREFILL PROJECTION GEOMETRY.

ONE VARIABLE: the mHC prefill TF32 projection's tile geometry. Nothing else
in the numerics, dtype, or code path changes — the projection stays TF32
with the split-FP32 (high+low) residual contract that plain-TF32 truncation
failed (1e-6); only the M/N/K tiling and the stage depth move.

THE FINDING (verified directly from the live image):

  `_projection_default_config` ships a decode-shaped fallback for every
  hidden-5120 token count except exactly 4096:

      hidden != 4096 and tokens >= 4096  -> M32 N8  K256 stages1 ks1
      everything else                    -> M16 N8  K256 stages1 ks1
      hidden == 4096  and tokens == 4096 -> M64 N24 K64  stages2 ks4   (tuned)

  The lane's mHC plan capacity is `max_num_batched_tokens` = 8192
  (`B12xWorkload.max_tokens`; `_plan_for` falls back to the largest prepared
  capacity for every smaller M), so the 2048-token prefill steps run the
  M32/N8/K256/stages1 kernel: three N tiles over 24 mixes, one TMA stage,
  and `projection_k_splits=1`. The engine's own efficiency predicates REJECT
  that geometry (`prefill_column_tiles` requires tile_n >= 24 at
  max_tokens >= 2048; the tokens<2048 rows also fail `prefill_warp_layout`).
  The shipped default is a config the tuner itself would filter out.

THE CHANGE: promote the vendor's own measured 4096-token geometry to every
hidden-5120 prefill plan (tokens >= 2048). One line:

      if hidden_size == 5_120 and tokens == 4_096:      (pristine)
      if hidden_size == 5_120 and tokens >= 2_048:      (patched)

  That geometry is not invented here: the image itself ships it for
  hidden=5120 at exactly 4096 tokens, and it is legal at 2048/4096/8192
  under the engine's own predicates (proved by golden_test.py, which runs
  the real predicate algebra).

EVIDENCE AT THE DESK (a rank-0 profile trace, 8x2048-token chunks):

  The three mHC prefill kernels (projection / gram / finalize) run ~80 ms
  per 2048-token chunk in aggregate on this hardware class (project ~40%,
  gram ~35%, finalize ~20%; each projection launch in the ~0.4-0.8 ms
  class, 40 launches per chunk).
  This patch can only move the project's share; the gram/finalize are not
  tile-driven (their own arms are separate windows).

FILE CLAIM: flashinfer/experimental/b12x/norm/mhc/_tuning.py — DISJOINT from
every other kit shipped in this recipe (including patches/engram-host-hash/)
and from the vllm tree. Composes with all of them.

Discipline: exact-once anchor, compile-before-write, atomic write,
byte-exact revert, pristine + patched md5 pins, stale-pyc eviction.

Usage: install.py --apply|--revert|--verify|--diff [--flashinfer-root DIR]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import py_compile
import sys
import tempfile
from pathlib import Path

DEFAULT_FLASHINFER = "/usr/local/lib/python3.12/dist-packages/flashinfer"
TARGET_REL = "experimental/b12x/norm/mhc/_tuning.py"

# The anchor is the vendor's hidden-5120 / exactly-4096 case. One occurrence
# in the pinned file; `--apply` refuses on 0 or >1 (foreign tree).
ORIG_ANCHOR = "    if hidden_size == 5_120 and tokens == 4_096:\n"
PATCHED_ANCHOR = "    if hidden_size == 5_120 and tokens >= 2_048:\n"

# Byte pins of the pinned image's file (nightly-20261015, digest 03a95654...).
PIN_PRISTINE_MD5 = "5bc90d53fea977e71c1db2290c9ae7ae"
PIN_PRISTINE_SHA256 = (
    "35ca49f259f562a63725791c760cbe948ab4fa832df47427e53b34f0481a8339"
)
PIN_PATCHED_MD5 = "fc2733aeadd0c29ab8c8d208c4f3040d"
PIN_PATCHED_SHA256 = (
    "985d5c63985929882bba78da5d97f828eea20c7369aafa1f151950d6862d2ebb"
)


def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def _write_atomic(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".ds41mhc-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(text: str) -> tuple[str, str]:
    return _md5(text), _sha256(text)


def _state(text: str) -> str:
    n_orig = text.count(ORIG_ANCHOR)
    n_patched = text.count(PATCHED_ANCHOR)
    if n_patched == 1 and n_orig == 0:
        return "patched"
    if n_orig == 1 and n_patched == 0:
        return "original"
    return "foreign"


def _evict_pyc(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if not cache.is_dir():
        return
    for stale in cache.glob(f"{path.stem}.*.pyc"):
        try:
            stale.unlink()
        except OSError:
            pass


def _apply_file(path: Path, action: str) -> tuple[bool, str]:
    text = _read(path)
    state = _state(text)
    md5, sha = _identity(text)

    if action == "apply":
        if state == "patched":
            ok = (md5, sha) == (PIN_PATCHED_MD5, PIN_PATCHED_SHA256)
            return ok, (
                f"apply: already patched (idempotent) {path.name} "
                f"md5 {md5} ({'MATCH' if ok else 'PIN MISMATCH'})"
            )
        if state == "foreign":
            return False, (
                f"refuse: {path.name} matches neither the pristine nor the "
                f"patched anchor exactly once (md5 {md5}) — foreign tree"
            )
        if (md5, sha) != (PIN_PRISTINE_MD5, PIN_PRISTINE_SHA256):
            return False, (
                f"refuse: pristine {path.name} md5 {md5} sha256 {sha} != the "
                f"pinned image ({PIN_PRISTINE_MD5}) — re-anchor the kit first"
            )
        patched = text.replace(ORIG_ANCHOR, PATCHED_ANCHOR, 1)
        try:
            compile(patched, str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - defensive
            return False, f"refuse: patched source does not compile: {exc}"
        if _identity(patched) != (PIN_PATCHED_MD5, PIN_PATCHED_SHA256):
            return False, "refuse: patched text does not match the pinned bytes"
        _write_atomic(path, patched)
        _evict_pyc(path)
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:  # pragma: no cover - defensive
            return False, f"POST-WRITE py_compile FAILED (left patched): {exc}"
        ok = _identity(_read(path)) == (PIN_PATCHED_MD5, PIN_PATCHED_SHA256)
        return ok, (
            f"apply: patched {path.name} md5 {md5} -> {PIN_PATCHED_MD5} "
            f"({'MATCH' if ok else 'MISMATCH'}); +tokens>=2048 hidden5120 "
            f"prefill geometry M64/N24/K64/stages2/ks4"
        )

    if action == "revert":
        if state == "original":
            ok = (md5, sha) == (PIN_PRISTINE_MD5, PIN_PRISTINE_SHA256)
            return ok, (
                f"revert: already original {path.name} md5 {md5} "
                f"({'MATCH' if ok else 'PIN MISMATCH'})"
            )
        if state == "foreign":
            return False, f"refuse: cannot revert foreign state in {path.name}"
        original = text.replace(PATCHED_ANCHOR, ORIG_ANCHOR, 1)
        try:
            compile(original, str(path), "exec")
        except SyntaxError as exc:  # pragma: no cover - defensive
            return False, f"refuse: reverted source does not compile: {exc}"
        _write_atomic(path, original)
        _evict_pyc(path)
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:  # pragma: no cover - defensive
            return False, f"POST-WRITE py_compile FAILED (left reverted): {exc}"
        ok = _identity(_read(path)) == (PIN_PRISTINE_MD5, PIN_PRISTINE_SHA256)
        return ok, (
            f"revert: byte-exact round trip {path.name} md5 {md5} -> "
            f"{PIN_PRISTINE_MD5} ({'CONFIRMED' if ok else 'PIN MISMATCH'})"
        )

    # verify
    if state == "patched":
        pristine = text.replace(PATCHED_ANCHOR, ORIG_ANCHOR, 1)
        pin_ok = _identity(pristine) == (PIN_PRISTINE_MD5, PIN_PRISTINE_SHA256)
        own_ok = (md5, sha) == (PIN_PATCHED_MD5, PIN_PATCHED_SHA256)
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:  # pragma: no cover - defensive
            return False, f"{path.name}: py_compile FAIL: {exc}"
        return own_ok and pin_ok, (
            f"{path.name}: patched; patched pin "
            f"{'MATCH' if own_ok else 'MISMATCH'}; pristine reconstruction "
            f"{'MATCH' if pin_ok else 'MISMATCH'}; py_compile ok"
        )
    if state == "original":
        ok = (md5, sha) == (PIN_PRISTINE_MD5, PIN_PRISTINE_SHA256)
        return ok, (
            f"{path.name}: original; pin {'MATCH' if ok else 'MISMATCH'} "
            f"(kit not armed)"
        )
    return False, f"{path.name}: FOREIGN (neither anchor occurs exactly once)"


def _print_diff(path: Path) -> int:
    text = _read(path)
    state = _state(text)
    print(f"### {TARGET_REL}  state={state} md5={_md5(text)}")
    print(
        "".join(
            difflib.unified_diff(
                ORIG_ANCHOR.splitlines(keepends=True),
                PATCHED_ANCHOR.splitlines(keepends=True),
                fromfile="pristine",
                tofile="patched",
                n=3,
            )
        ),
        end="",
    )
    print(f"pins: pristine {PIN_PRISTINE_MD5}  patched {PIN_PATCHED_MD5}")
    return 0 if state != "foreign" else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flashinfer-root", default=DEFAULT_FLASHINFER)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--revert", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--diff", action="store_true")
    args = ap.parse_args(argv)

    target = Path(args.flashinfer_root) / TARGET_REL
    if not target.is_file():
        print(f"refuse: missing {target}", file=sys.stderr)
        return 2
    if args.diff:
        return _print_diff(target)

    action = "apply" if args.apply else "revert" if args.revert else "verify"
    ok, note = _apply_file(target, action)
    print(f"[ds41mhc] {note}")
    print(f"[ds41mhc] {action}:", "OK" if ok else "FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
