#!/usr/bin/env python3
"""Static integration audit of the SHIPPED ds41_engram_host module
(whole-module self-consistency discipline, applied at once).

Runs in-container, read-only on the live engine tree, against:
  A. the module's own classes — every ``self.X`` / ``rt.X`` reference vs the
     attributes the class actually defines (catches the
     called-a-module-function-as-a-method class of boot-killer);
  B. every ENGINE object attribute the module touches — verified against the
     real classes (imported where CPU-safe) or their installed source text
     (grep) where instantiation needs CUDA;
  C. every lazily-imported module inside function bodies — verified to
     resolve in-container;
  D. the shipped module's own ``--selftest`` executed FROM THE SHIPPED FILE.

Usage (in-container, from a scratch dir holding the patched engine copies):
  python3 audit_integration.py \
      --module ./vllm/ds41_engram_host.py \
      --patched-disk-table ./b12x/sequence/_shared/disk_table.py
Exit 0 iff every check passes.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import dataclasses
import importlib.util
import os
import subprocess
import sys

ROWS: list[tuple[str, str, str]] = []  # (area, check, result)


def row(area, check, ok, detail=""):
    ROWS.append((area, check, "PASS" if ok else "FAIL"))
    if not ok:
        print(f"  FAIL {area}: {check} {detail}")
    return ok


def load_module_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# A. self/rt attribute audit of the shipped module
# ---------------------------------------------------------------------------


def audit_self_references(src, path):
    tree = ast.parse(src, path)
    # Class scopes: collect assigned self attrs + methods per class.
    classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            assigned, methods = set(), set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.add(item.name)
                    for sub in ast.walk(item):
                        targets = []
                        if isinstance(sub, ast.Assign):
                            targets = sub.targets
                        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                            targets = [sub.target]
                        for t in targets:
                            if (
                                isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"
                            ):
                                assigned.add(t.attr)
            classes[node.name] = assigned | methods

    # __slots__ literals count as defined for classes that use them.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for t in item.targets:
                        if (
                            isinstance(t, ast.Name)
                            and t.id == "__slots__"
                            and isinstance(item.value, (ast.Tuple, ast.List))
                        ):
                            classes[node.name] |= {
                                e.value
                                for e in item.value.elts
                                if isinstance(e, ast.Constant)
                            }

    ok = True
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id == "self"
            ):
                if item.attr not in classes[node.name]:
                    ok = False
                    print(
                        f"  FAIL self-reference: {node.name}.self.{item.attr} "
                        f"at line {item.lineno} is never assigned/defined on "
                        f"{node.name}"
                    )
                else:
                    ROWS.append(
                        ("A", f"{node.name}.self.{item.attr}", "PASS")
                    )
    # rt.* references (locals aliasing HostEngramRuntime in functions).
    rt_known = classes.get("HostEngramRuntime", set())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for item in ast.walk(node):
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "rt"
                ):
                    if item.attr in rt_known:
                        ROWS.append(("A", f"rt.{item.attr}", "PASS"))
                    else:
                        ok = False
                        print(
                            f"  FAIL rt-reference: rt.{item.attr} at line "
                            f"{item.lineno} ({node.name}) not on "
                            f"HostEngramRuntime"
                        )

    # Undefined-helper sweep: every load of an underscore-prefixed name must
    # resolve to a module-level def/import/assign, a function parameter, a
    # local assignment, or a builtin. Catches calling a helper that was
    # renamed/removed (the _image_mask bug class, post-fix form).
    module_names = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, ast.Import):
            module_names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            module_names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Assign):
            module_names |= {
                t.id for t in node.targets if isinstance(t, ast.Name)
            }
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)

    def function_locals(fn):
        local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        if fn.args.vararg:
            local.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            local.add(fn.args.kwarg.arg)
        for sub in ast.walk(fn):
            targets = []
            if isinstance(sub, ast.Assign):
                targets = sub.targets
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                targets = [sub.target]
            elif isinstance(sub, (ast.For, ast.comprehension)):
                targets = [sub.target]
            elif isinstance(sub, ast.With):
                targets = [i.optional_vars for i in sub.items if i.optional_vars]
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                local.add(sub.name)
            elif isinstance(sub, ast.Import):
                local |= {a.asname or a.name.split(".")[0] for a in sub.names}
            elif isinstance(sub, ast.ImportFrom):
                local |= {a.asname or a.name for a in sub.names}
            for t in targets:
                if t is None:
                    continue
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        local.add(n.id)
        return local

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = module_names | function_locals(node)
            for item in ast.walk(node):
                if (
                    isinstance(item, ast.Name)
                    and isinstance(item.ctx, ast.Load)
                    and item.id.startswith("_")
                    and item.id not in scope
                ):
                    ok = False
                    print(
                        f"  FAIL undefined helper: {item.id} at line "
                        f"{item.lineno} ({node.name}) resolves to nothing "
                        f"module-level, local, or builtin"
                    )
    return ok


# ---------------------------------------------------------------------------
# B. engine-side names the module touches
# ---------------------------------------------------------------------------


def audit_engine_names(disk_table_patched_path):
    ok = True
    dist = "/usr/local/lib/python3.12/dist-packages"

    def srctext(rel):
        with open(os.path.join(dist, rel), encoding="utf-8") as f:
            return f.read()

    # InputBatch: real dataclass fields (CPU-safe import).
    try:
        from vllm.v1.worker.gpu.input_batch import InputBatch

        fields = {f.name for f in dataclasses.fields(InputBatch)}
        for name in (
            "query_start_loc_np",
            "idx_mapping_np",
            "num_draft_tokens_per_req",
            "num_reqs",
            "uniform_decode_graph",
            "input_ids",
        ):
            ok &= row("B", f"InputBatch.{name}", name in fields)
    except Exception as exc:
        ok = False
        row("B", "import InputBatch", False, str(exc))

    # DefaultModelState: class-level attribute.
    try:
        from vllm.v1.worker.gpu.model_states.default import DefaultModelState

        ok &= row(
            "B",
            "DefaultModelState.num_new_sampled_tokens_per_step",
            hasattr(DefaultModelState, "num_new_sampled_tokens_per_step"),
        )
    except Exception as exc:
        ok = False
        row("B", "import DefaultModelState", False, str(exc))

    # vllm.config: get_current_vllm_config import + spec field in source.
    try:
        from vllm.config import get_current_vllm_config  # noqa: F401

        row("B", "vllm.config.get_current_vllm_config import", True)
    except Exception as exc:
        ok = False
        row("B", "vllm.config.get_current_vllm_config import", False, str(exc))
    ok &= row(
        "B",
        "speculative config field enable_adaptive_verification",
        "enable_adaptive_verification" in srctext("vllm/config/speculative.py"),
    )

    # RequestState: source grep (instantiation needs CUDA).
    states_src = srctext("vllm/v1/worker/gpu/states.py")
    for name in (
        "self.num_computed_tokens_np",
        "self.prefill_len",
        "self.all_token_ids",
        "self.draft_tokens",
        "self.num_speculative_steps",
    ):
        ok &= row("B", f"RequestState source: {name}", name in states_src)

    # StagedWriteTensor/UvaBackedTensor surface (buffer_utils source).
    buf_src = srctext("vllm/v1/worker/gpu/buffer_utils.py")
    ok &= row("B", "UvaBackedTensor has .np (source)", "self.np = " in buf_src)
    ok &= row(
        "B",
        "StagedWriteTensor exposes .gpu, no .np (source)",
        "self.gpu = " in buf_src and "def np" not in buf_src,
    )

    # NgramHashState / Engram / model seams (source grep).
    engram_src = srctext("vllm/models/deepseek_v4_1/common/engram.py")
    for name in (
        "def _ensure_bindings",
        "self._token_map = token_map",
        '"num_tokens"',
        "def stage_disk",
        "self.layer_hash_index = layer_hash_index",
    ):
        ok &= row("B", f"common/engram.py source: {name}", name in engram_src)

    model_src = srctext("vllm/models/deepseek_v4_1/nvidia/model.py")
    for name in (
        '"prepared_engram_hashes"',
        "self.engram_hash = NgramHashState",
        "def prepare_disk_engram",
    ):
        ok &= row("B", f"nvidia/model.py source: {name}", name in model_src)

    # flashinfer live pieces (CPU-safe imports).
    try:
        from b12x.sequence.engram import _impl as engram_impl

        ok &= row("B", "engram._impl._READ_POOL", hasattr(engram_impl, "_READ_POOL"))
        from b12x.sequence.engram._kernels import lookup_op  # noqa: F401

        row("B", "engram._kernels.lookup_op import", True)
        from b12x.sequence._shared.disk_table import DiskRowCache

        ok &= row("B", "DiskRowCache._read_staged", hasattr(DiskRowCache, "_read_staged"))
        ok &= row("B", "DiskRowCache.transaction", hasattr(DiskRowCache, "transaction"))
    except Exception as exc:
        ok = False
        row("B", "flashinfer imports", False, str(exc))

    # LookupBinding fields.
    try:
        from b12x.sequence.engram._impl import LookupBinding

        lb = {f.name for f in dataclasses.fields(LookupBinding)}
        for name in ("weight", "scale_bytes", "hash_ids", "num_tokens", "out", "plan"):
            ok &= row("B", f"LookupBinding.{name}", name in lb)
        ok &= row("B", "LookupBinding.disk_table", "disk_table" in lb)
    except Exception as exc:
        ok = False
        row("B", "import LookupBinding", False, str(exc))

    # DiskTable._cache (engram/_disk.py source).
    ok &= row(
        "B",
        "engram/_disk.py DiskTable._cache",
        "self._cache = DiskRowCache(" in srctext(
            "flashinfer/experimental/b12x/sequence/engram/_disk.py"
        ),
    )

    # The PATCHED scratch disk_table must carry stage_rows_host.
    if disk_table_patched_path and os.path.isfile(disk_table_patched_path):
        patched_src = open(disk_table_patched_path, encoding="utf-8").read()
        ok &= row(
            "B",
            "patched disk_table defines stage_rows_host",
            "def stage_rows_host" in patched_src,
        )
    else:
        ok = False
        row("B", "patched disk_table path", False, disk_table_patched_path or "")
    return ok


# ---------------------------------------------------------------------------
# C. lazy imports inside function bodies
# ---------------------------------------------------------------------------


def audit_lazy_imports(src, path):
    tree = ast.parse(src, path)
    wanted = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for item in ast.walk(node):
                mods = []
                if isinstance(item, ast.Import):
                    mods = [a.name for a in item.names]
                elif isinstance(item, ast.ImportFrom) and item.module:
                    mods = [item.module]
                for m in mods:
                    if m not in wanted:
                        wanted.append(m)
    ok = True
    for m in wanted:
        try:
            __import__(m)
            row("C", f"lazy import resolves: {m}", True)
        except Exception as exc:
            ok = False
            row("C", f"lazy import resolves: {m}", False, str(exc))
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="shipped ds41_engram_host.py")
    ap.add_argument("--patched-disk-table", default="")
    args = ap.parse_args()

    src = open(args.module, encoding="utf-8").read()
    print(f"== A. self/rt attribute audit of {args.module} ==")
    ok = audit_self_references(src, args.module)
    print("== B. engine-side names ==")
    ok &= audit_engine_names(args.patched_disk_table)
    print("== C. lazy imports ==")
    ok &= audit_lazy_imports(src, args.module)
    print("== D. shipped-module selftest ==")
    proc = subprocess.run(
        [sys.executable, args.module, "--selftest"],
        capture_output=True, text=True,
    )
    ok &= row(
        "D", "shipped module --selftest exit 0", proc.returncode == 0,
        proc.stderr[-400:],
    )

    # Summary table.
    print("\n==== AUDIT TABLE ====")
    cur = None
    for area, check, result in ROWS:
        if area != cur:
            print(f"[{area}]")
            cur = area
        print(f"  {result:4s}  {check}")
    fails = sum(1 for _, _, r in ROWS if r == "FAIL")
    print(f"\n{len(ROWS)} checks, {fails} failures")
    print("AUDIT:", "PASS" if ok and fails == 0 else "FAIL")
    return 0 if ok and fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
