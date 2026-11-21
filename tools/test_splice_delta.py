#!/usr/bin/env python3
"""test_splice_delta.py — synthetic proof of splice-delta.py's three
properties, with no network and no real weights:

  1. a complete tensor list rebuilds the build's shard byte-for-byte,
  2. a build header of a DIFFERENT length (an added __metadata__ key) is
     reproduced — the data section shifts and the patch follows it,
  3. an INCOMPLETE tensor list fails closed: exit 2, no output file left,
  4. a full run links the identical shard and the metadata, and a
     CORRUPTED local base shard is refused before it can be linked.

Run:  python3 tools/test_splice_delta.py     (exit 0 = all five hold)
"""
import contextlib
import os
import hashlib
import io
import json
import struct
import sys
import tempfile
from pathlib import Path

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "splice_delta", Path(__file__).resolve().parent / "splice-delta.py")
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

SHARD = "model-00001-of-00002.safetensors"
OTHER = "model-00002-of-00002.safetensors"


def write_safetensors(path: Path, tensors: dict, metadata: bool) -> None:
    header, offset = {}, 0
    for name, blob in tensors.items():
        header[name] = {"dtype": "F8_E4M3", "shape": [len(blob)],
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    if metadata:
        header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    pad = (-len(raw)) % 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw) + pad))
        f.write(raw + b" " * pad)
        for blob in tensors.values():
            f.write(blob)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ok = 0
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        base, build, dest = t / "base", t / "build", t / "out"
        base.mkdir(); build.mkdir()
        n = 4096
        t_a = bytes(i % 256 for i in range(n))             # edited
        t_a2 = bytes((i + 1) % 256 for i in range(n))      # its edited form
        t_b = bytes((255 - i) % 256 for i in range(n))     # edited
        t_b2 = bytes((254 - i) % 256 for i in range(n))    # its edited form
        t_c = bytes((i * 7) % 256 for i in range(n))       # untouched
        # case 1: same header length; case 2: the build adds __metadata__
        for case, meta in ((1, False), (2, True)):
            d = t / f"case{case}"
            b, k, o = d / "base", d / "build", d / "out"
            b.mkdir(parents=True); k.mkdir()
            write_safetensors(b / SHARD, {"a": t_a, "b": t_b, "c": t_c}, False)
            write_safetensors(k / SHARD, {"a": t_a2, "b": t_b2, "c": t_c}, meta)
            base_bytes = (b / SHARD).read_bytes()
            build_bytes = (k / SHARD).read_bytes()
            edited = {"a": t_a2, "b": t_b2}
            same_sha = "0" * 64  # the other shard is identical in both trees

            def fake_tree(repo, revision, _bb=build_bytes, _base=base_bytes):
                mine = _bb if repo == "build" else _base
                return {SHARD: {"sha256": hashlib.sha256(mine).hexdigest(),
                                "git_sha1": None, "size": len(mine)},
                        OTHER: {"sha256": same_sha, "git_sha1": None, "size": 0}}

            def fake_range(url, start, end, _bb=build_bytes):
                return _bb[start:end + 1]

            man = {"build": "synthetic", "repo": "build", "revision": "v",
                   "base_repo": "base", "base_revision": "v0",
                   "tensors": sorted(edited)}
            (d / "man.json").write_text(json.dumps(man))
            orig_tree, orig_range = sd.hub_tree, sd.fetch_range
            sd.hub_tree, sd.fetch_range = fake_tree, fake_range
            try:
                sys.argv = ["splice-delta.py", "--base-dir", str(b),
                            "--manifest", str(d / "man.json"),
                            "--dest-dir", str(o), "--only-shards", SHARD]
                with contextlib.redirect_stdout(io.StringIO()):
                    sd.main()
                same = (o / SHARD).read_bytes() == build_bytes
                print(f"[case {case}] rebuilt shard byte-identical to the build: "
                      f"{'PASS' if same else 'FAIL'}"
                      f"{' (header shifted by the added __metadata__)' if meta else ''}")
                ok += same

                # case 3 (once): an incomplete tensor list must fail closed
                if case == 1:
                    bad = dict(man, tensors=["a"])  # "b" is missing
                    (d / "bad.json").write_text(json.dumps(bad))
                    o2 = d / "out-bad"
                    sys.argv = ["splice-delta.py", "--base-dir", str(b),
                                "--manifest", str(d / "bad.json"),
                                "--dest-dir", str(o2), "--only-shards", SHARD]
                    code = 0
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), \
                                contextlib.redirect_stderr(io.StringIO()):
                            sd.main()
                    except SystemExit as exc:
                        code = exc.code
                    leftover = list(o2.glob("*")) if o2.exists() else []
                    fails_closed = code == 2 and not leftover
                    print(f"[case 3] incomplete tensor list: exit {code}, "
                          f"{len(leftover)} file(s) left behind — "
                          f"{'PASS (fails closed)' if fails_closed else 'FAIL'}")
                    ok += fails_closed
            finally:
                sd.hub_tree, sd.fetch_range = orig_tree, orig_range
        # case 4: the full path (hard-link + metadata) and the corrupted-base refusal
        d = t / "case4"
        b, k, o = t / "case4" / "base", t / "case4" / "build", t / "case4" / "out"
        b.mkdir(parents=True); k.mkdir()
        n = 4096
        t_a = bytes(i % 256 for i in range(n))
        t_a2 = bytes((i + 3) % 256 for i in range(n))
        t_c = bytes((i * 7) % 256 for i in range(n))
        write_safetensors(b / SHARD, {"a": t_a, "c": t_c}, False)
        write_safetensors(k / SHARD, {"a": t_a2, "c": t_c}, False)
        other_bytes = b"identical-in-both" * 512
        (b / OTHER).write_bytes(other_bytes)
        meta = {f: json.dumps({"file": f}).encode() for f in sd.METADATA}
        for f, blob in meta.items():
            (b / f).write_bytes(blob)
        base_bytes, build_bytes = (b / SHARD).read_bytes(), (k / SHARD).read_bytes()

        def fake_tree4(repo, revision):
            mine = build_bytes if repo == "build" else base_bytes
            return {SHARD: {"sha256": hashlib.sha256(mine).hexdigest(),
                            "git_sha1": None, "size": len(mine)},
                    OTHER: {"sha256": hashlib.sha256(other_bytes).hexdigest(),
                            "git_sha1": None, "size": len(other_bytes)},
                    **{f: {"sha256": None, "git_sha1": sd.git_blob_sha1(blob),
                           "size": len(blob)} for f, blob in meta.items()}}

        def fake_range4(url, start, end):
            return build_bytes[start:end + 1]

        man4 = {"build": "synthetic", "repo": "build", "revision": "v",
                "base_repo": "base", "base_revision": "v0", "tensors": ["a"]}
        (d / "man.json").write_text(json.dumps(man4))
        orig_tree, orig_range = sd.hub_tree, sd.fetch_range
        sd.hub_tree, sd.fetch_range = fake_tree4, fake_range4

        def run(outdir):
            sys.argv = ["splice-delta.py", "--base-dir", str(b),
                        "--manifest", str(d / "man.json"), "--dest-dir", str(outdir)]
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    sd.main()
                return 0, ""
            except SystemExit as exc:
                return exc.code, ""

        try:
            code, _ = run(o)
            files = sorted(x.name for x in o.iterdir()) if o.exists() else []
            full = (code == 0 and files == sorted([SHARD, OTHER, *sd.METADATA,
                                                   "_SPLICE_SOURCE.json"])
                    and (o / SHARD).read_bytes() == build_bytes
                    and os.path.samefile(b / OTHER, o / OTHER))
            print(f"[case 4a] full run: rebuilt + hard-linked + metadata + marker: "
                  f"{'PASS' if full else 'FAIL'}")
            ok += full

            (b / OTHER).write_bytes(other_bytes[:-1] + b"\x00")  # corrupt the local base
            o2 = d / "out-corrupt"
            code, _ = run(o2)
            refused = code == 2 and not (o2 / OTHER).exists()
            print(f"[case 4b] corrupted local base shard: exit {code}, not linked — "
                  f"{'PASS (refused)' if refused else 'FAIL'}")
            ok += refused
        finally:
            sd.hub_tree, sd.fetch_range = orig_tree, orig_range
    print(f"\n{ok}/5 properties hold")
    return 0 if ok == 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
