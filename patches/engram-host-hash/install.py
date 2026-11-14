#!/usr/bin/env python3
"""Install/revert/verify the DS41 host-side Engram hash patch kit.

Four changes, all visible-text, all idempotent, all refusing a tree whose
anchor blocks carry foreign edits:

  1. Ships host_hash.py into the engine as vllm/ds41_engram_host.py (new
     file; revert deletes it). The runtime path is inert unless
     DS41_ENGRAM_HOST_HASH=1 at engine time.
  2. vllm/models/deepseek_v4_1/nvidia/model.py :: prepare_disk_engram —
     gains an optional host_inputs argument and branches to the host path
     (run_prepare_host) when it is provided; otherwise the stock
     run_native + stage_disk + run_lookups path runs unchanged.
  3. vllm/models/deepseek_v4_1/nvidia/model_state.py :: prepare_inputs —
     builds the host-mirror inputs via maybe_collect (the fence event and
     the spec draft-buffer D2H live inside; returns None -> native
     fallback whenever exactness is not guaranteed) and passes them to
     prepare_disk_engram; DS41_ENGRAM_HOST_HASH_STRICT=1 re-raises
     collection failures instead of falling back silently.
  4. flashinfer .../b12x/sequence/_shared/disk_table.py :: DiskRowCache —
     adds stage_rows_host(host_ids, count): write pinned ids_host from
     host memory, STAGE ONLY (the host path then reads via _read_staged
     concurrently, mirroring the native run_lookups order) with no D2H
     copy and no GPU event (an unrecorded event's synchronize returns
     immediately).

Usage:
  install.py --apply   [--vllm-root DIR] [--b12x-root DIR]
  install.py --revert  [...]
  install.py --verify  [...]
  install.py --diff    (show the exact patches; touches nothing)

Defaults point at the ds41-flash-tp3 container layout. Apply NEVER reverts
or force-patches: any anchor mismatch, foreign marker state, or module
drift is a hard refusal (exit 2). After apply, run the strict boot:
DS41_ENGRAM_HOST_HASH=1 DS41_ENGRAM_HOST_HASH_STRICT=1 on the next boot;
the first real prepare batch runs BOTH host and native hashes and asserts
bit-exact equality (mismatch -> permanent native fallback + loud log, or a
raise under STRICT).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import py_compile
import shutil
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
MODULE_NAME = "ds41_engram_host.py"
MARK_BEGIN = "# --- DS41-ENGRAM-HOST-HASH BEGIN (revert: restore original block) ---"
MARK_END = "# --- DS41-ENGRAM-HOST-HASH END ---"

DEFAULT_VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
DEFAULT_B12X = "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x"

# ---------------------------------------------------------------------------
# Site 1: model.py — prepare_disk_engram host branch
# ---------------------------------------------------------------------------

MODEL_ORIG = '''\
    def prepare_disk_engram(self, input_ids, query_start_loc, lookback_token_ids):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        engrams = tuple(
            layer.engram
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if getattr(layer, "engram", None) is not None
        )
        try:
            for engram in engrams:
                engram.invalidate_disk_output()
            if self.engram_hash is None:
                raise RuntimeError("Disk Engram requires initialized hash state")
            hashes = self.prepared_engram_hashes[: input_ids.shape[0]]
            self.engram_hash.run_native(
                input_ids,
                ~image_sentinel_mask(input_ids),
                query_start_loc,
                lookback_token_ids,
                hashes,
            )
            # One batch lets the per-layer disk reads overlap on the host.
            bindings = [
                engram.stage_disk(
                    hashes[:, engram.layer_hash_index], self.engram_hash.num_tokens
                )
                for engram in engrams
            ]
            counts = [hashes.shape[0]] * len(bindings)
            # Enqueue the gathers before replay: graph waits can block every
            # CUDA work queue needed by a late side-stream lookup.
            engram_native.run_lookups(bindings, counts, clear_tail=False)
            for engram in engrams:
                engram.finish_disk(hashes.shape[0])
        except BaseException:
'''

MODEL_PATCHED = '''\
    def prepare_disk_engram(
        self, input_ids, query_start_loc, lookback_token_ids, host_inputs=None
    ):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Disk Engram preparation must run outside compile/capture"
            )
        engrams = tuple(
            layer.engram
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if getattr(layer, "engram", None) is not None
        )
        try:
            for engram in engrams:
                engram.invalidate_disk_output()
            if self.engram_hash is None:
                raise RuntimeError("Disk Engram requires initialized hash state")
            hashes = self.prepared_engram_hashes[: input_ids.shape[0]]
            host_done = False
            ''' + MARK_BEGIN + '''
            if host_inputs is not None:
                from vllm.ds41_engram_host import run_prepare_host

                host_done = run_prepare_host(
                    self,
                    engrams,
                    hashes,
                    input_ids,
                    query_start_loc,
                    lookback_token_ids,
                    host_inputs,
                )
            ''' + MARK_END + '''
            if not host_done:
                self.engram_hash.run_native(
                    input_ids,
                    ~image_sentinel_mask(input_ids),
                    query_start_loc,
                    lookback_token_ids,
                    hashes,
                )
                # One batch lets the per-layer disk reads overlap on the host.
                bindings = [
                    engram.stage_disk(
                        hashes[:, engram.layer_hash_index],
                        self.engram_hash.num_tokens,
                    )
                    for engram in engrams
                ]
                counts = [hashes.shape[0]] * len(bindings)
                # Enqueue the gathers before replay: graph waits can block every
                # CUDA work queue needed by a late side-stream lookup.
                engram_native.run_lookups(bindings, counts, clear_tail=False)
            for engram in engrams:
                engram.finish_disk(hashes.shape[0])
        except BaseException:
'''

# ---------------------------------------------------------------------------
# Site 2: model_state.py — fence + host-input collection at the call site
# ---------------------------------------------------------------------------

STATE_ORIG = '''\
        model_inputs["lookback_token_ids"] = window
        for model in self.disk_engram_models:
            model.prepare_disk_engram(
                input_batch.input_ids, input_batch.query_start_loc, window
            )
        return model_inputs
'''

STATE_PATCHED = '''\
        model_inputs["lookback_token_ids"] = window
        host_engram_inputs = None
        ''' + MARK_BEGIN + '''
        try:
            from vllm.ds41_engram_host import maybe_collect

            host_engram_inputs = maybe_collect(self, input_batch, req_states)
        except Exception:
            # STRICT mode propagates collection failures loudly (the whole
            # point of the flag); otherwise fall back to the native path.
            import os

            if os.environ.get("DS41_ENGRAM_HOST_HASH_STRICT", "") == "1":
                raise
            host_engram_inputs = None
        ''' + MARK_END + '''
        for model in self.disk_engram_models:
            model.prepare_disk_engram(
                input_batch.input_ids,
                input_batch.query_start_loc,
                window,
                host_inputs=host_engram_inputs,
            )
        return model_inputs
'''

# ---------------------------------------------------------------------------
# Site 3: disk_table.py — stage_rows_host (stage only, NO read)
# ---------------------------------------------------------------------------

DISK_ORIG = '''\
    def read_rows(self, ids: torch.Tensor, count: int) -> None:
        self._stage_ids(ids, count)
        self._read_staged(count)
'''

DISK_PATCHED = '''\
    def read_rows(self, ids: torch.Tensor, count: int) -> None:
        self._stage_ids(ids, count)
        self._read_staged(count)

    ''' + MARK_BEGIN + '''
    def stage_rows_host(self, host_ids: torch.Tensor, count: int) -> None:
        """Stage host-written IDs into the pinned buffer; no GPU work.

        The caller guarantees the first ``count`` int64 IDs in ``host_ids``
        (contiguous CPU int64) are fully written host-side with no GPU
        producer in flight for that region, and will call ``_read_staged``
        itself — the host path stages ALL tables first, then reads them
        concurrently, mirroring the native run_lookups order. No D2H copy
        and no event record happen here, so ``_ids_ready`` stays unrecorded
        on this path and the eventual ``_read_staged`` sync returns
        immediately (an unrecorded event's synchronize never blocks; a
        long-completed earlier record is equally cheap).
        """
        if self._transaction_thread != threading.get_ident():
            raise RuntimeError(
                "stage_rows_host requires an active disk row transaction"
            )
        count = operator.index(count)
        if not 0 <= count <= self.max_lookups:
            raise ValueError("disk lookup count exceeds batch capacity")
        if (
            host_ids.device.type != "cpu"
            or host_ids.dtype != torch.int64
            or not host_ids.is_contiguous()
        ):
            raise ValueError("host disk row IDs must be contiguous CPU int64")
        if host_ids.numel() < count:
            raise ValueError("host disk row IDs do not cover the requested count")
        self.ids_host[:count].copy_(host_ids.view(-1)[:count])
    ''' + MARK_END + '''
'''


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _diff(path: Path, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{path} (original)",
            tofile=f"{path} (patched)",
        )
    )


def _read(f: Path) -> str:
    return f.read_text(encoding="utf-8")


def _write(f: Path, text: str) -> None:
    tmp = f.with_suffix(f.suffix + ".ds41tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(f)


def _replace_anchor(f: Path, orig: str, patched: str, action: str) -> str:
    src = _read(f)
    # Patched-first ordering: the insert-style patch (disk_table.py) embeds
    # the original block verbatim, so "original present" alone never
    # distinguishes the two states.
    has_orig, has_patched = src.count(orig), src.count(patched)
    marks = src.count(MARK_BEGIN)
    if action == "apply":
        if has_patched == 1:
            return "already present"
        if has_patched > 1:
            raise RuntimeError(f"foreign marker state in {f} — refusing")
        if has_orig != 1:
            raise RuntimeError(
                f"anchor mismatch in {f}: original block count={has_orig} "
                f"(expected 1); foreign edits present — refusing"
            )
        if marks:
            raise RuntimeError(f"foreign marker state in {f} — refusing")
        _write(f, src.replace(orig, patched, 1))
        return "patched"
    if action == "revert":
        if has_patched == 1:
            _write(f, src.replace(patched, orig, 1))
            return "reverted"
        if has_patched == 0 and has_orig == 1 and marks == 0:
            return "already original"
        raise RuntimeError(
            f"revert mismatch in {f}: patched count={has_patched} "
            f"original count={has_orig} markers={marks} — refusing"
        )
    raise ValueError(action)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", default=DEFAULT_VLLM)
    ap.add_argument("--b12x-root", default=DEFAULT_B12X)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--diff", action="store_true", help="print patches, change nothing")
    a = ap.parse_args()

    vllm = Path(a.vllm_root)
    b12x = Path(a.b12x_root)
    model_py = vllm / "models/deepseek_v4_1/nvidia/model.py"
    state_py = vllm / "models/deepseek_v4_1/nvidia/model_state.py"
    disk_py = b12x / "sequence/_shared/disk_table.py"
    module_py = vllm / MODULE_NAME
    kit_module = KIT / "host_hash.py"
    files = [model_py, state_py, disk_py]

    if a.diff:
        print(f"=== kit module -> {module_py} (new file) ===")
        print(f"    source: {kit_module} (md5 {_md5(kit_module)})")
        print(f"=== {model_py} ===")
        print(_diff(model_py, MODEL_ORIG, MODEL_PATCHED), end="")
        print(f"=== {state_py} ===")
        print(_diff(state_py, STATE_ORIG, STATE_PATCHED), end="")
        print(f"=== {disk_py} ===")
        print(_diff(disk_py, DISK_ORIG, DISK_PATCHED), end="")
        return 0

    for f in [*files, kit_module]:
        if not f.is_file():
            print(f"refuse: missing {f}", file=sys.stderr)
            return 2

    SITES = (
        (model_py, MODEL_ORIG, MODEL_PATCHED),
        (state_py, STATE_ORIG, STATE_PATCHED),
        (disk_py, DISK_ORIG, DISK_PATCHED),
    )

    if a.apply:
        # Validate every anchor BEFORE writing anything (atomic refusal).
        if module_py.exists() and _md5(module_py) != _md5(kit_module):
            print(
                f"refuse: {module_py} exists and differs from the kit — "
                "remove it or sync the kit",
                file=sys.stderr,
            )
            return 2
        planned = []
        for f, orig, patched in SITES:
            src = _read(f)
            if src.count(patched) == 1:
                planned.append((f, src, "already present"))
                continue  # already applied; idempotent
            if src.count(patched) > 1 or src.count(orig) != 1 or src.count(MARK_BEGIN):
                print(
                    f"refuse: anchor mismatch in {f} "
                    f"(original count={src.count(orig)}, markers="
                    f"{src.count(MARK_BEGIN)}); foreign edits present",
                    file=sys.stderr,
                )
                return 2
            new_src = src.replace(orig, patched, 1)
            try:
                compile(new_src, str(f), "exec")  # syntax-check BEFORE writing
            except SyntaxError as exc:
                print(
                    f"refuse: patched {f} does not compile: {exc}",
                    file=sys.stderr,
                )
                return 2
            planned.append((f, new_src, "patched"))
        # Also compile-verify the shipped module's bytes before writing.
        try:
            compile(kit_module.read_text(encoding="utf-8"), str(kit_module), "exec")
        except SyntaxError as exc:
            print(f"refuse: kit module does not compile: {exc}", file=sys.stderr)
            return 2
        # All anchors good and all patched texts compile: write.
        if module_py.exists():
            print(f"module: already present (md5 {_md5(module_py)})")
        else:
            shutil.copyfile(kit_module, module_py)
            print(f"module: installed {module_py} (md5 {_md5(module_py)})")
        for f, new_src, state in planned:
            if state == "already present":
                print(f"{f.name}: already present")
                continue
            before = _md5(f)
            _write(f, new_src)
            py_compile.compile(str(f), doraise=True)
            print(f"{f.name}: patched")
            print(f"    md5 {before} -> {_md5(f)} (pre-compiled + py_compile ok)")
        print("apply complete. Boot gates: DS41_ENGRAM_HOST_HASH=1 (enable),")
        print("DS41_ENGRAM_HOST_HASH_STRICT=1 (raise on self-check mismatch).")
        return 0

    if a.revert:
        try:
            for f, orig, patched in SITES:
                before = _md5(f)
                print(f"{f.name}: {_replace_anchor(f, orig, patched, 'revert')}")
                py_compile.compile(str(f), doraise=True)
                print(f"    md5 {before} -> {_md5(f)} (py_compile ok)")
        except RuntimeError as exc:
            print(f"refuse: {exc}", file=sys.stderr)
            return 2
        if module_py.exists():
            module_py.unlink()
            print(f"module: removed {module_py}")
        else:
            print("module: absent")
        return 0

    # --verify
    ok = True
    for f, orig, patched in SITES:
        src = _read(f)
        if src.count(patched) == 1:
            state = "patched"
        elif src.count(orig) == 1 and src.count(patched) == 0 and not src.count(MARK_BEGIN):
            state = "original (not applied)"
        else:
            state = "FOREIGN/INCONSISTENT"
            ok = False
        try:
            py_compile.compile(str(f), doraise=True)
            comp = "py_compile ok"
        except py_compile.PyCompileError as e:
            comp = f"py_compile FAIL: {e}"
            ok = False
        print(f"{f}: {state}; {comp}")
    if module_py.exists():
        same = _md5(module_py) == _md5(kit_module)
        print(f"{module_py}: {'matches kit' if same else 'DIFFERS FROM KIT'}")
        ok = ok and same
    else:
        print(f"{module_py}: absent (host path will not engage)")
    print("verify:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
