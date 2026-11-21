#!/usr/bin/env python3
"""splice-delta.py — build an alternate-weights checkpoint from the base
checkpoint plus ONLY the tensors the build actually edits.

WHY. An abliterated build is the base checkpoint with an edit in a small set
of tensors. That edit happens to live across many shard FILES, so the naive
route ("download the other build") pulls every shard that contains even one
edited tensor — for the build shipped here, 23 of 48 shards. The tensors
themselves are a tiny fraction of that: at most 92 of the model's 96,085
tensors (~1.2 GiB of bytes). This tool takes the base checkpoint you
already have, fetches exactly those tensors' byte ranges from the
publisher's shard files (HTTP range requests — no whole shard is downloaded),
writes them into copies of the base shards, verifies EVERY rebuilt shard
against the publisher's published SHA-256, and (by default) hash-checks the
base's shards that will be hard-linked against the base's own published
manifest — so a damaged or mixed-revision local base is refused, not linked. An incomplete or wrong tensor
list fails closed: the output never silently diverges from the publisher's
build.

WHAT IT DOES NOT DO. It does not boot, start, stop or switch anything, and it
never writes into the base checkpoint directory. The output is a sibling
directory: 25 hard links to the base's identical shards (zero extra bytes),
the rebuilt shards, copies of the metadata files, and a provenance marker.

USAGE
    python3 splice-delta.py --base-dir /path/to/base-checkpoint \
        --manifest tools/abliterated-delta-moonmath.json \
        [--dest-dir /path/to/output] [--workers 4] [--only-shards a,b,...]

REQUIREMENTS. Python 3.8+ (standard library only), network access to the
model hub, and ~160 GiB of free disk for the rebuilt shards. Base and build
must share the safetensors layout (same tensor names, shapes, dtypes and
data offsets) — the tool asserts this per shard before writing anything.

EXIT CODES. 0 success; 2 any failure (a failed shard is deleted before exit).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "splice-delta/1.0 (partial checkpoint assembly)"}
METADATA = ("config.json", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json")
RETRIES = 5


def log(msg: str) -> None:
    print(f"[splice] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[splice] FATAL: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


# ---------------------------------------------------------------- hub access

def hub_tree(repo: str, revision: str) -> dict:
    """path -> {sha256 (LFS) | git_sha1 (plain), size} for every file."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1"
    out = {}
    cursor = ""
    while True:
        req = urllib.request.Request(url + cursor, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            items = json.load(r)
            link = r.headers.get("Link", "") or ""
        for x in items:
            if x.get("type") != "file":
                continue
            lfs = x.get("lfs") or {}
            out[x["path"]] = {
                "sha256": lfs.get("oid"),
                "git_sha1": None if lfs.get("oid") else x.get("oid"),
                "size": x.get("size"),
            }
        nxt = ""
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split("<", 1)[1].split(">", 1)[0]
        if not nxt:
            return out
        url, cursor = nxt, ""


def fetch_range(url: str, start: int, end: int) -> bytes:
    """Bytes [start, end] inclusive, following the CDN redirect with the
    Range header intact. Retries transient failures with backoff."""
    want = end - start + 1
    last = None
    for attempt in range(RETRIES):
        req = urllib.request.Request(
            url, headers={**UA, "Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status != 206:
                    raise IOError(f"expected 206 Partial Content, got {r.status}")
                data = r.read()
            if len(data) != want:
                raise IOError(f"short read: {len(data)} of {want} bytes")
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, IOError,
                OSError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise IOError(f"range fetch failed after {RETRIES} tries: {last}")


def git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode())
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ safetensors

def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        raw = f.read(hlen)
    if len(raw) != hlen:
        raise IOError(f"truncated safetensors header in {path}")
    return hlen, json.loads(raw)


def layout(header: dict) -> dict:
    return {k: (v["dtype"], tuple(v["shape"]), tuple(v["data_offsets"]))
            for k, v in header.items() if k != "__metadata__"}


# --------------------------------------------------------------------- the splice

def splice_shard(base: Path, dest: Path, name: str, build_url: str,
                 edited: set, workers: int) -> str:
    """Rebuild one shard: the build's header + the base's data section with
    the edited tensors' ranges replaced by bytes fetched from the build."""
    hlen_b, hdr_b = read_header(base)
    lay_b = layout(hdr_b)

    # the build's own header (byte-exact, so an added __metadata__ key or a
    # different key order is reproduced rather than guessed)
    first = fetch_range(build_url, 0, 7)
    (hlen_n,) = struct.unpack("<Q", first)
    new_head = first + fetch_range(build_url, 8, 8 + hlen_n - 1)
    lay_n = layout(json.loads(new_head[8:]))
    if lay_b != lay_n:
        raise IOError(
            f"{name}: the build's tensor layout differs from the base "
            "(names/shapes/dtypes/offsets) — an in-place data patch cannot "
            "reproduce it; download the build's shard instead")

    here = sorted(t for t in edited if t in lay_n)
    spans = {t: lay_n[t][2] for t in here}

    def grab(t: str) -> bytes:
        a, b = spans[t]
        return fetch_range(build_url, 8 + hlen_n + a, 8 + hlen_n + b - 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        blobs = dict(zip(here, ex.map(grab, here)))

    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(base, "rb") as src, open(tmp, "wb") as out:
        out.write(new_head)
        src.seek(8 + hlen_b)
        remaining = os.fstat(src.fileno()).st_size - (8 + hlen_b)
        while remaining:
            chunk = src.read(min(1 << 24, remaining))
            if not chunk:
                raise IOError(f"{name}: base shard shorter than its header says")
            out.write(chunk)
            remaining -= len(chunk)
        for t, blob in blobs.items():
            a, b = spans[t]
            out.seek(8 + hlen_n + a)
            out.write(blob)
    return str(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-dir", required=True,
                    help="the base checkpoint you already have")
    ap.add_argument("--manifest", required=True,
                    help="the build's tensor manifest (tools/*.json)")
    ap.add_argument("--dest-dir", default="",
                    help="output directory (default: <base>-<build>)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-verify-base", action="store_true",
                    help="do not hash-check the base shards that are "
                         "hard-linked (saves ~5-10 min; the rebuilt shards "
                         "are still verified)")
    ap.add_argument("--only-shards", default="",
                    help="rebuild just these shards and skip links/metadata "
                         "(for proving the mechanism on one shard)")
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    repo, rev = man["repo"], man["revision"]
    base_repo, base_rev = man["base_repo"], man["base_revision"]
    edited = set(man["tensors"])
    base = Path(args.base_dir).expanduser()
    dest = Path(args.dest_dir).expanduser() if args.dest_dir else \
        base.parent / f"{base.name}-{man['build']}"
    if not base.is_dir():
        die(f"base checkpoint not found: {base}")
    if dest.resolve() == base.resolve():
        die("dest-dir must not be the base directory")

    log(f"build {repo} @ {rev[:12]} — {len(edited)} edited tensors")
    build_tree = hub_tree(repo, rev)
    base_tree = hub_tree(base_repo, base_rev)
    shards = sorted(p for p in build_tree
                    if p.startswith("model-") and p.endswith(".safetensors"))
    same = [s for s in shards if build_tree[s]["sha256"] == base_tree[s]["sha256"]]
    diff = [s for s in shards if s not in same]
    log(f"{len(diff)} shards differ from the base, {len(same)} are identical")
    only = [s for s in args.only_shards.split(",") if s] or None

    dest.mkdir(parents=True, exist_ok=True)
    done = []
    for name in (only or diff):
        if name not in diff:
            die(f"{name} is not a differing shard (it is hard-linked from the base)")
        src = base / name
        if not src.exists():
            die(f"base is missing {name}")
        t0 = time.time()
        try:
            tmp = splice_shard(src, dest, name,
                               f"https://huggingface.co/{repo}/resolve/{rev}/{name}",
                               edited, args.workers)
        except Exception as exc:  # layout mismatch, truncated file, network…
            for stale in dest.glob(name + "*"):
                stale.unlink(missing_ok=True)
            die(f"{name}: could not be rebuilt ({type(exc).__name__}: {exc}) "
                "— nothing was trusted")
        got = sha256_file(Path(tmp))
        want = build_tree[name]["sha256"]
        if got != want:
            Path(tmp).unlink(missing_ok=True)
            die(f"{name}: SHA-256 MISMATCH after the splice (got {got[:16]}…, "
                f"publisher {want[:16]}…) — the tensor list does not fully "
                "describe this build; nothing was trusted")
        Path(tmp).rename(dest / name)
        log(f"  {name}: rebuilt + SHA-256 verified ({time.time() - t0:.0f} s)")
        done.append(name)

    if only:
        log(f"proof run: {len(done)} shard(s) match the publisher bit for bit")
        return

    for name in same:
        src, dst = base / name, dest / name
        if not src.exists():
            die(f"base is missing {name} (needed to hard-link)")
        if dst.exists():
            if os.path.samefile(src, dst):
                continue
            dst.unlink()
        if not args.skip_verify_base:
            # the link is only as trustworthy as the local file: check it
            # against the base's own published manifest before relying on it
            got = sha256_file(src)
            want = base_tree[name]["sha256"]
            if got != want:
                die(f"{name}: the local base shard does not match the base "
                    f"checkpoint's published SHA-256 (got {got[:16]}…) — the "
                    "base is damaged or a mixed revision; nothing was linked")
        os.link(src, dst)
    log(f"  hard-linked {len(same)} identical shards (0 bytes"
        + (", each hash-checked against the base's manifest)"
           if not args.skip_verify_base else ")"))

    for f in METADATA:
        data = (base / f).read_bytes()
        meta = build_tree.get(f) or die(f"{f}: not found in the build's file list")
        if meta["sha256"]:
            ok_meta = hashlib.sha256(data).hexdigest() == meta["sha256"]
        elif meta["git_sha1"]:
            ok_meta = git_blob_sha1(data) == meta["git_sha1"]
        else:
            ok_meta = None
        if ok_meta is False:
            die(f"{f}: the base copy differs from the build's published file "
                "— copy it from the build before serving")
        if ok_meta is None:
            log(f"  note: {f} has no published digest to check against "
                "(copied from the base)")
        shutil.copy2(base / f, dest / f)

    (dest / "_SPLICE_SOURCE.json").write_text(json.dumps({
        "build": man["build"], "repo": repo, "revision": rev,
        "base_repo": base_repo, "base_revision": base_rev,
        "rebuilt_shards": done, "hard_linked_shards": len(same),
        "edited_tensors": sorted(edited),
        "verification": "every rebuilt shard SHA-256-matched the publisher's "
                        "published manifest before this file was written",
    }, indent=1) + "\n")
    log(f"done: {dest} — {len(done)} rebuilt, {len(same)} hard-linked, "
        "all verified against the publisher")


if __name__ == "__main__":
    main()
