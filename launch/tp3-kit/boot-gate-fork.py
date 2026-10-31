#!/usr/bin/env python3
"""boot-gate-fork.py — fail-the-boot state check for the FORK posture.

The b12x fork serves the NATURAL checkpoint (its loader does TP3 padding
at runtime; an early bring-up attempt died on 'MXFP4 index heads
must divide 64' — the fork's kernels want the ORIGINAL geometry). This
gate proves the model dir is the ORIGINAL pull before the engine starts:
48 shards at root (engram included), the 12 engram index entries PRESENT,
config untouched (heads 64, vocab 129280, engram_layer_ids [1,14]).

It ALSO proves the ENGINE is the audited build (engine-identity.json):
the image is patched at runtime by the bootstrap (the valid_object shim),
so a manifest digest alone cannot say what is executing. The build stamp
and the engine-critical file hashes can. A new eugr nightly therefore
FAILS this gate until it is re-audited — that is the adoption gate.
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/model"
DIST = "/usr/local/lib/python3.12/dist-packages"
HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY = os.path.join(HERE, "engine-identity.json")
ok = True


def check(cond, msg):
    global ok
    print(("  OK   " if cond else "  FAIL ") + msg)
    ok = ok and cond


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ model dir
def load_json(path, what, where="in the model dir"):
    """Read a JSON object, or fail with the gate's own line — not a traceback.

    This runs inside the container before the engine starts, so a malformed
    model dir must produce the designed `FAIL <msg>` output rather than a
    Python stack trace the operator has to decode.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        check(False, f"{what} present {where}")
    except (OSError, ValueError) as exc:
        check(False, f"{what} readable ({exc.__class__.__name__})")
    else:
        if isinstance(data, dict):
            return data
        check(False, f"{what} carries a JSON object (got {type(data).__name__})")
    raise SystemExit(1)


def list_model_dir():
    """The shard listing, guarded: a missing model root is a gate failure."""
    try:
        return sorted(f for f in os.listdir(ROOT)
                      if f.startswith("model-") and f.endswith(".safetensors"))
    except OSError as exc:
        check(False, f"model dir readable ({exc.__class__.__name__})")
        raise SystemExit(1)


check(os.path.exists(os.path.join(ROOT, "_DOWNLOAD_COMPLETE.json")),
      "snapshot complete (_DOWNLOAD_COMPLETE.json present)")

shards = list_model_dir()
check(len(shards) == 48, f"48 shards at root (got {len(shards)}) — engram included")
check(not os.path.isdir(os.path.join(ROOT, "engram-retained")),
      "no engram-retained/ (fork reads the files at root)")

cfg_path = os.path.join(ROOT, "config.json")
idx_path = os.path.join(ROOT, "model.safetensors.index.json")

cfg = load_json(cfg_path, "config.json")
tc = cfg.get("text_config") or {}
if not tc:
    check(False, "config.json carries a text_config block")
    raise SystemExit(1)
if not isinstance(tc, dict):
    check(False, f"config.json text_config carries an object (got {type(tc).__name__})")
    raise SystemExit(1)
check(tc.get("num_attention_heads") == 64, "config heads == 64 (original)")
check(tc.get("index_n_heads") == 32, "config index_n_heads == 32 (divides 64)")
check(tc.get("vocab_size") == 129280, f"config vocab_size == 129280 (original)")
check(tc.get("engram_layer_ids") == [1, 14],
      "config engram_layer_ids == [1, 14] (original; fork's choice how to serve)")

idx = load_json(idx_path, "model.safetensors.index.json")
wm = idx.get("weight_map") or {}
if not wm:
    check(False, "model.safetensors.index.json carries a weight_map")
    raise SystemExit(1)
if not isinstance(wm, dict):
    check(False, f"index weight_map carries an object (got {type(wm).__name__})")
    raise SystemExit(1)
eng = [n for n in wm if ".engram." in n]
check(len(eng) == 12, f"index carries the 12 engram entries (got {len(eng)})")
check("mtp.0.ffn.experts.128.w1.weight" not in wm,
      "no padded drafter expert (natural checkpoint)")

# ------------------------------------------------------------------ engine identity
if not os.path.exists(IDENTITY):
    check(False, f"engine-identity.json present in the kit ({IDENTITY})")
else:
    spec = load_json(IDENTITY, "engine-identity.json", where="in the kit")

    stamp = spec.get("build_stamp", {})
    spath = stamp.get("path", "")
    if not spath or not os.path.exists(spath):
        check(False, f"build stamp present ({spath or 'unset'})")
    else:
        vals = {}
        for line in open(spath):
            if ":" in line:
                k, _, v = line.partition(":")
                vals[k.strip()] = v.strip().strip('"')
        bad = [f"{k}={vals.get(k)!r} want {v!r}"
               for k, v in stamp.get("expect", {}).items() if vals.get(k) != v]
        check(not bad, "build stamp matches (vllm/flashinfer/build commits)"
              + ("" if not bad else " — " + "; ".join(bad)))

    # The MODEL contract: values the engine reads out of the checkpoint.
    # The code that consumes them is pinned by hash above, so this pins the
    # inputs. A silent numerical error (an activation limit that stops
    # being carried to the kernel) would otherwise show up as plausible
    # output and nothing else.
    for dotted, want in spec.get("model_contract", {}).items():
        if dotted == "why" or dotted.endswith(".why"):
            continue
        node = cfg
        for part in dotted.split("."):
            node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        check(node == want, f"model contract {dotted} == {want!r}"
              + ("" if node == want else f" — got {node!r}"))

    for rel, want in spec.get("files", {}).items():
        p = os.path.join(DIST, rel)
        got = sha256_file(p) if os.path.exists(p) else None
        check(got == want, f"engine file {rel}"
              + ("" if got == want else f" — sha256 {got} != {want}"))

    for rel, rule in spec.get("semantic", {}).items():
        p = os.path.join(DIST, rel)
        src = open(p).read() if os.path.exists(p) else ""
        shim = "def valid_object" in src
        try:
            subprocess.run(
                [sys.executable, "-c",
                 "from b12x._lib.cache_integrity import valid_object"],
                check=True, capture_output=True, timeout=120)
            resolvable = True
        except Exception:
            resolvable = False
        check(shim or resolvable, f"engine file {rel} ({rule})")

print("\nBOOT-GATE-FORK " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
