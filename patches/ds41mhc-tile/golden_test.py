#!/usr/bin/env python3
"""Golden test for the DS41 mHC prefill-geometry kit.

Runs the PINNED IMAGE'S OWN tuning algebra — `b12x/preparation/{types,tuning}.py`
loaded byte-exactly from the target tree, plus the live `_tuning.py` — with a
torch-free `._kernels` stub, and asserts:

  1. the live file is in the expected lifecycle state and carries its byte pin;
     the counterpart state reconstructs byte-exactly from it (round-trip);
  2. at the lane's real plan capacity (max_num_batched_tokens = 8192) and at
     2048/4096 tokens, hidden 5120 resolves to the vendor's tuned geometry
     M64/N24/K64/2 stages/4 k-splits instead of the M32|M16/N8/K256/1-stage
     fallback (three N tiles re-streaming A, no pipelining);
  3. that geometry passes the engine's correctness AND efficiency predicates
     (`ParameterSpace.validate`) — the tuner itself rejects the shipped
     default, which is the whole point of the change;
  4. the pristine file's 2048/8192 rows FAIL those same predicates (the
     negative control: the defect is real and this change is the fix);
  5. the decode/native rows (tokens 1/16/96/128), the 1024-token row and the
     hidden 4096/7168 families resolve identically in both files, i.e. the
     patch has no collateral behavior change;
  6. the smem_limit assumption cannot change the verdict: the whole matrix is
     run at 232448 B (GB10's 227 KiB opt-in) and at 65536 B and must agree.

Exit 0 = every claim holds; anything else fails the boot gate.

Usage:
  golden_test.py [--flashinfer-root DIR] [--expect patched|pristine] [--quiet]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

TARGET_REL = "experimental/b12x/norm/mhc/_tuning.py"
PREP_DIR = "experimental/b12x/preparation"
ORIG_ANCHOR = "    if hidden_size == 5_120 and tokens == 4_096:\n"
PATCHED_ANCHOR = "    if hidden_size == 5_120 and tokens >= 2_048:\n"

PIN_PRISTINE_MD5 = "5bc90d53fea977e71c1db2290c9ae7ae"
PIN_PATCHED_MD5 = "fc2733aeadd0c29ab8c8d208c4f3040d"

MIXES = 24
BUFFER_ALIGN = 1024

# (hidden, tokens, operation) -> (backend, tile_m, tile_n, tile_k, stages,
#                                 m_warps, n_warps, k_splits)
PATCHED_EXPECT = {
    (5120, 2048, "post_pre"): ("tf32_tma", 64, 24, 64, 2, 4, 1, 4),
    (5120, 4096, "post_pre"): ("tf32_tma", 64, 24, 64, 2, 4, 1, 4),
    (5120, 8192, "post_pre"): ("tf32_tma", 64, 24, 64, 2, 4, 1, 4),
    (5120, 8192, "pre"): ("tf32_tma", 64, 24, 64, 2, 4, 1, 4),
    (5120, 1, "post_pre"): ("native", 16, 8, 256, 1, 1, 1, 1),
    (5120, 16, "post_pre"): ("native", 16, 8, 256, 1, 1, 1, 1),
    (5120, 96, "post_pre"): ("native", 16, 8, 256, 1, 1, 1, 1),
}

# Rows that must resolve identically in the pristine and patched files.
NO_COLLATERAL = [
    (5120, 1, "post_pre"),
    (5120, 16, "post_pre"),
    (5120, 96, "post_pre"),
    (5120, 128, "post_pre"),
    (5120, 1024, "post_pre"),
    (4096, 2048, "post_pre"),
    (4096, 4096, "post_pre"),
    (4096, 8192, "post_pre"),
    (7168, 2048, "post_pre"),
    (7168, 8192, "post_pre"),
]

LEGAL_PATCHED = [
    (5120, 2048, "post_pre"),
    (5120, 4096, "post_pre"),
    (5120, 8192, "post_pre"),
    (5120, 8192, "pre"),
]

REJECTED_PRISTINE = [
    (5120, 2048, "post_pre"),
    (5120, 8192, "post_pre"),
]

SMEM_LIMITS = (232448, 65536)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    """Read the exact bytes as text.

    `Path.read_text(newline=...)` is Python 3.13+; the pinned image's
    interpreter is 3.12.3, so the newline handling is explicit here.
    """
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_preparation(b12x_root: Path):
    """Register a torch-free `b12x.preparation` from the image's own bytes."""
    for name in [n for n in sys.modules if n == "b12x" or n.startswith("b12x.")]:
        del sys.modules[name]
    preparation = b12x_root / PREP_DIR
    missing = [p.name for p in (preparation / "types.py", preparation / "tuning.py")
               if not p.is_file()]
    if missing:
        raise SystemExit(f"missing preparation modules under {preparation}: {missing}")
    package = types.ModuleType("b12x")
    package.__path__ = [str(b12x_root)]
    sys.modules["b12x"] = package
    prep = types.ModuleType("b12x.preparation")
    prep.__path__ = [str(preparation)]
    sys.modules["b12x.preparation"] = prep
    typ = _load_file("b12x.preparation.types", preparation / "types.py")
    tun = _load_file("b12x.preparation.tuning", preparation / "tuning.py")
    prep.types = typ
    prep.tuning = tun
    for name in ("FrozenMapping", "DeviceIdentity"):
        setattr(prep, name, getattr(typ, name))
    for name in ("Knob", "ParameterBinding", "ParameterSpace", "TuningContract"):
        setattr(prep, name, getattr(tun, name))
    return typ, tun


def load_tuning(package: str, typ, tuning_path: Path):
    """Import one `_tuning.py` as `<package>._tuning` with a stubbed `._kernels`."""
    host = types.ModuleType(package)
    host.__path__ = [str(tuning_path.parent)]
    sys.modules[package] = host
    kernels = types.ModuleType(f"{package}._kernels")
    kernels._MIXES = MIXES

    class MHCPrefillTf32ProjectTmaKernel:
        buffer_align_bytes = BUFFER_ALIGN

    kernels.MHCPrefillTf32ProjectTmaKernel = MHCPrefillTf32ProjectTmaKernel
    sys.modules[f"{package}._kernels"] = kernels
    host._kernels = kernels
    module = _load_file(f"{package}._tuning", tuning_path)
    device = typ.DeviceIdentity(
        vendor="nvidia", compute_capability=(12, 1), sm_count=48, product_name="GB10"
    )
    return module, device


def _query(typ, module, hidden: int, tokens: int, operation: str, smem: int):
    return module.MhcQuery(
        dtype="bfloat16",
        max_tokens=tokens,
        hidden_size=hidden,
        split_k=4 * hidden // 256,
        operation=operation,
        has_norm_weight=True,
        lagged_mix=True,
        expanded_residual=operation == "pre",
        smem_limit=smem,
        controls=typ.FrozenMapping(),
        codegen=typ.FrozenMapping(),
    )


def _geometry(config) -> tuple:
    return (
        config.backend,
        config.projection_tile_m,
        config.projection_tile_n,
        config.projection_tile_k,
        config.projection_num_stages,
        config.projection_num_m_warps,
        config.projection_num_n_warps,
        config.projection_k_splits,
    )


def resolve(typ, module, device, key, smem):
    """Default geometry + the engine's own predicate verdict for one row."""
    hidden, tokens, operation = key
    try:
        query = _query(typ, module, hidden, tokens, operation, smem)
        module.TUNING.validate_query(query, device)
        config = module.TUNING.default_config(query, device)
        module.TUNING.validate_config(query, config, device)
        space = module.TUNING.parameters(query, device)
        names = {knob.name for knob in space.knobs}
        assignment = {
            name: value for name, value in config.to_dict().items() if name in names
        }
        try:
            space.validate(assignment)
            verdict = None
        except Exception as exc:  # noqa: BLE001 - a predicate failure is a result
            verdict = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        return _geometry(config), verdict
    except Exception as exc:  # noqa: BLE001 - planning failure is a result too
        return None, f"PLAN-FAIL {type(exc).__name__}: {exc}".replace("\n", " ")


class Checker:
    def __init__(self, quiet: bool = False) -> None:
        self.failures: list[str] = []
        self.quiet = quiet
        self.passes = 0

    def check(self, ok: bool, label: str) -> None:
        if ok:
            self.passes += 1
            if not self.quiet:
                print(f"[ds41mhc] PASS {label}")
        else:
            self.failures.append(label)
            print(f"[ds41mhc] FAIL {label}", file=sys.stderr)

    def report(self, stage: str) -> int:
        if self.failures:
            print(
                f"[ds41mhc] golden: FAILED at {stage} — {len(self.failures)} of "
                f"{self.passes + len(self.failures)} checks",
                file=sys.stderr,
            )
            return 2
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--flashinfer-root", default="/usr/local/lib/python3.12/dist-packages/flashinfer"
    )
    ap.add_argument("--expect", choices=("patched", "pristine"), default="patched")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.flashinfer_root)
    target = root / TARGET_REL
    if not target.is_file():
        print(f"[ds41mhc] FAIL missing {target}", file=sys.stderr)
        return 2

    live_text = _read_text(target)
    live_md5 = _md5(live_text)
    expected_md5 = PIN_PATCHED_MD5 if args.expect == "patched" else PIN_PRISTINE_MD5

    check = Checker(args.quiet)
    check.check(
        live_md5 == expected_md5,
        f"live {Path(TARGET_REL).name} md5 {live_md5} == {expected_md5} "
        f"({args.expect})",
    )
    anchors = live_text.count(ORIG_ANCHOR) + live_text.count(PATCHED_ANCHOR)
    check.check(anchors == 1, f"exactly one geometry anchor occurs (found {anchors})")
    if check.report("pins"):
        return 2

    # Materialise both lifecycle states, pin-verified, whichever is live.
    if live_md5 == PIN_PATCHED_MD5:
        patched_text = live_text
        pristine_text = live_text.replace(PATCHED_ANCHOR, ORIG_ANCHOR, 1)
    else:
        pristine_text = live_text
        patched_text = live_text.replace(ORIG_ANCHOR, PATCHED_ANCHOR, 1)
    check.check(
        _md5(pristine_text) == PIN_PRISTINE_MD5,
        "pristine state reconstructs to its pinned md5 (byte-exact round trip)",
    )
    check.check(
        _md5(patched_text) == PIN_PATCHED_MD5,
        "patched state reconstructs to its pinned md5 (byte-exact round trip)",
    )
    if check.report("round-trip"):
        return 2

    with tempfile.TemporaryDirectory(prefix="ds41mhc-golden-") as scratch:
        scratch_path = Path(scratch)
        pristine_copy = scratch_path / "_tuning_pristine.py"
        patched_copy = scratch_path / "_tuning_patched.py"
        pristine_copy.write_text(pristine_text, encoding="utf-8")
        patched_copy.write_text(patched_text, encoding="utf-8")

        typ, _tun = install_preparation(root)
        pristine_module, device = load_tuning("mhc_pristine", typ, pristine_copy)
        patched_module, _device = load_tuning("mhc_patched", typ, patched_copy)

        rows = set(PATCHED_EXPECT) | set(NO_COLLATERAL) | set(LEGAL_PATCHED)
        matrices = {}
        for smem in SMEM_LIMITS:
            pristine_rows = {
                key: resolve(typ, pristine_module, device, key, smem) for key in rows
            }
            patched_rows = {
                key: resolve(typ, patched_module, device, key, smem) for key in rows
            }
            matrices[smem] = (pristine_rows, patched_rows)

            for key, expected in PATCHED_EXPECT.items():
                geometry = patched_rows[key][0]
                check.check(
                    geometry == expected,
                    f"smem={smem} patched geometry {key} == {expected} (got {geometry})",
                )

            for key in NO_COLLATERAL:
                check.check(
                    pristine_rows[key] == patched_rows[key],
                    f"smem={smem} no collateral change {key} "
                    f"(pristine {pristine_rows[key]})",
                )

            for key in LEGAL_PATCHED:
                geometry, verdict = patched_rows[key]
                check.check(
                    verdict is None,
                    f"smem={smem} engine predicates ACCEPT {key} "
                    f"{(geometry or ('?',))[0]}/{geometry[1:3] if geometry else '?'}"
                    + (f" [{verdict}]" if verdict else ""),
                )

            for key in REJECTED_PRISTINE:
                geometry, verdict = pristine_rows[key]
                check.check(
                    verdict is not None,
                    f"smem={smem} engine predicates REJECT the shipped default {key} "
                    f"{geometry[1:3] if geometry else '?'} (negative control)",
                )

        check.check(
            matrices[SMEM_LIMITS[0]] == matrices[SMEM_LIMITS[1]],
            f"smem robustness: identical matrices at {SMEM_LIMITS}",
        )

    if check.report("matrix"):
        return 2
    print(
        f"[ds41mhc] golden: PASS — {check.passes} checks ({args.expect}, "
        f"smem {SMEM_LIMITS}, rows {len(rows)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
