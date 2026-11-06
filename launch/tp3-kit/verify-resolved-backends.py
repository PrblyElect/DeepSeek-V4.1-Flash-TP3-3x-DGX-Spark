#!/usr/bin/env python3
"""verify-resolved-backends.py — check the engine RESOLVED the behaviour this
recipe was validated against.

WHY THIS EXISTS. The command line is not the behaviour. The dense-linear
kernel and the attention backend are chosen by the engine's own policy —
and the MoE backend's resolution still has to be proven even though the
launcher pins it — and a new upstream image can change them under the
recipe's command line. A reader who checks the command line and sees
nothing wrong has proved nothing about the kernels that actually ran.

The boot gate (boot-gate-fork.py) proves the ENGINE is the audited build:
the build stamp and the engine-critical file hashes. This script proves the
other half — that the audited build RESOLVED to the expected backends and
that the model's activation limit is still being carried to the kernel.

Two kinds of evidence are used, and both must agree:

  * the engine's own startup log — what it says it is using;
  * the engine's own /metrics and its own config resolution — what it
    actually did.

A missing marker is a FAIL, not a skip: if the log format changes upstream,
this fails loudly instead of silently passing.

USE. After the engine answers:

    python3 verify-resolved-backends.py --container <name>

Exit code 0 means every expectation held.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "engine-identity.json")

# Log markers that must be PRESENT. Each maps a manifest key to a substring
# the engine prints when it selected that behaviour.
#
# Every marker below was read out of a real engine's startup log, not
# guessed from upstream source: the fork is a fork, and the upstream file it
# was cut from is not always the file that prints the line. Each marker is
# value-bearing -- the expected value is interpolated -- so a marker that
# merely happens to appear in the log cannot satisfy a DIFFERENT backend.
#
# The file that prints each marker is hash-pinned in the manifest, so the
# marker cannot be reformatted out from under this check.
LOG_MARKERS = {
    "moe_backend": "Using '{v}' Mxfp4 MoE backend.",
    "dense_linear_kernel": "Using {v} for MXFP8 GEMM",
    # The indexer's readiness line changed shape when the FlashInfer
    # autotuner was shipped OFF: the "b12x ready ... N/N"
    # form only printed at the end of a tuning pass, and with tuning
    # stopped the reporter emits the heuristic-warmup form instead. Both
    # are the b12x readiness reporter naming its backend — value-bearing
    # either way — and the emitter file is hash-pinned, so the marker
    # cannot be reformatted out from under this check.
    "attention_backend": "{v} warming heuristics attention.dsa_indexer",
    "tp3_head_pad": "Padded DeepSeek V4.1 attention for TP3: {v}",
    "spec_decode": "speculative_config={v}",
    "load_format": "load_format={v}",
    "scheduler_block_size": "kv lcm block sizes {v}",
    "max_num_batched_tokens": "max_num_batched_tokens={v}",
}

# Log markers that must be ABSENT (see the manifest's `forbidden` block).
#
FORBIDDEN_MARKERS = {
    # The b12x MoE prefix marker was RETIRED: the b12x backend is
    # now the EXPECTED one (see the manifest), and the clamp hazard the
    # marker guarded moved to the tiny-decode env, asserted via
    # ENV_MARKERS below -- a missing env fails the check loudly. The
    # NvFp4 forbid stays: that backend remains wrong for this model.
    "nvfp4_moe_backend": "NvFp4 MoE backend",
}

# Which all-reduce ACTUALLY ENGAGED.
#
# The recipe sets VLLM_ENABLE_ROCE_ALLREDUCE=1 and
# VLLM_ROCE_ALLREDUCE_MAX_SIZE=2MB, and the image ships a one-shot RoCE
# collective (B12X_ROCENANTE) that needs BOTH RoCE rails of every link to
# reach the peer. The shipped topology is ONE CABLE PER PAIR OF BOXES
# (three cables, each device connected to the other two, both of each
# device's ports in use), so the collective cannot form: the engine logs the
# failure and falls through to PYNCCL. Nothing about that is visible from
# the recipe's command line -- the knob is set, the collective is absent,
# and the reduction is carried by a different path with no error signal.
#
# So the check states what happened rather than what was requested. The
# EXPECTED value lives in the manifest, so a reader who adds a second
# cable between the same pair of boxes gets a FAIL here until they update
# the expectation: the fabric change becomes a gated event instead of an
# unrecorded one.
ALLREDUCE_RE = re.compile(
    r"Using \[([^\]]*)\] all-reduce backends(?: \(in dispatch order\))?"
    r"(?: for group '([^']*)')?")
ALLREDUCE_KEYS = ("allreduce_backend", "allreduce_backend_by_group")


def expected_ar(expect: dict, group: str | None):
    """Expected all-reduce backend for one dispatch group (map first, scalar fallback)."""
    m = expect.get("allreduce_backend_by_group") or {}
    if group is not None and group in m:
        return m[group]
    return expect.get("allreduce_backend")


def norm_backends(raw):
    """Normalise a backend list to bare, unquoted names.

    The engine prints the ENGAGED backends as a Python list repr --
    ``Using ['PYNCCL'] all-reduce backends`` -- so the captured text carries
    the quotes. A manifest expectation is written without them, so comparing
    the raw text would fail a healthy lane on punctuation alone. Both sides
    are normalised the same way, and a name that differs only by quoting is
    treated as the same backend.
    """
    if raw is None:
        return None
    names = []
    for part in raw.split(","):
        name = part.strip().strip("'\"").strip()
        if name:
            names.append(name)
    return tuple(names)

# Metrics that must read exactly. Keyed by the cache_config_info label.
METRIC_KEYS = {
    "reported_block_size": "block_size",
    "enable_prefix_caching": "enable_prefix_caching",
    "sliding_window": "sliding_window",
}

# The metric family the labels above live in. Scoping the match to the
# family keeps a label of the same name in some other family from being
# read as this one.
METRIC_FAMILY = "vllm:cache_config_info"

# Expect keys that are checked against the engine's own resolved config
# rather than against the log or the metrics.
CONFIG_KEYS = ("swiglu_limit",)

# Expect keys checked against the container's own command line — the
# REQUESTED value of a flag, before the engine had a chance to resolve,
# round or reinterpret it. The value is the FLAG, not the whole marker:
# the command line is a shell string, so only a LITERAL number in it is
# evidence (see ENV_MARKERS).
CMD_MARKERS = {
    "requested_block_size": "--block-size",
    # No CLI flag exists for the tiny-decode kill switch; the entry exists
    # so the check loop runs for this key, and the ENV branch (above)
    # always resolves it. If the env were absent, the cmd fallback would
    # find no literal and fail loudly -- the intended direction.
    "tiny_decode_env": "--b12x-w4a8-tiny-decode",
}

# The container's ENVIRONMENT carries the resolved value of a knob the
# launcher forwards with `-e NAME="$NAME"`, and that is the authoritative
# source for a REQUESTED value.
#
# WHY NOT THE COMMAND LINE: the serve line is a shell string, so the
# command line holds the literal text "${BLOCK_SIZE:-256}". That text is
# IDENTICAL for every value the operator could set, so matching it would
# assert nothing at all — and a check that cannot fail is worse than no
# check. The expansion happens inside the container, when the bootstrap
# runs; the value it produced is in the environment.
#
# The command line is still consulted, but only for a LITERAL number: a
# launcher that bakes the value in (`--block-size 256`) is accepted too,
# because there the text IS the value.
ENV_MARKERS = {
    "requested_block_size": "BLOCK_SIZE",
    # The clamp check that replaced the retired b12x-MoE forbid: the env
    # must read exactly "0". Anything else -- including the image default
    # ("1", tiny-decode ON, clamp dropped) -- fails. There is no CLI flag
    # for this knob, so the env is the only honest source.
    "tiny_decode_env": "B12X_W4A8_TINY_DECODE",
}


def run(cmd, timeout=120, **kw):
    """Run a command, returning (rc, stdout+stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, f"{exc.__class__.__name__}: {exc}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def container_logs(container):
    rc, out = run(["docker", "logs", container], timeout=180)
    return out if rc == 0 else ""


def container_cmd(container):
    """The container's own command line, as the engine was actually started.

    This is the only place the REQUESTED value of a flag survives: the
    engine logs the block size it resolved and `/metrics` reports the
    minimum across the KV groups, but neither says what the command line
    asked for. A knob that is read but never forwarded is invisible to
    every other check here.
    """
    rc, out = run(["docker", "inspect", container, "--format",
                   "{{json .Config.Cmd}}"], timeout=60)
    return out if rc == 0 else ""


def container_env_value(container, name):
    """A forwarded knob's resolved value, read from the container's env.

    The serve line is a shell string, so the container's command line
    holds the literal "${BLOCK_SIZE:-256}" and cannot be parsed for a
    number -- the expansion happens inside the container, when the
    bootstrap runs. The resolved value is in the environment, which the
    launcher passes explicitly.

    Returns the integer, or None when the container does not carry it.
    """
    rc, out = run(["docker", "exec", container, "printenv", name], timeout=60)
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    rc, out = run(["docker", "inspect", container, "--format",
                   "{{range .Config.Env}}{{println .}}{{end}}"], timeout=60)
    if rc == 0:
        for line in out.splitlines():
            k, _, v = line.strip().partition("=")
            if k == name and v.strip().isdigit():
                return int(v.strip())
    return None


def container_port(container, default=8600):
    """The port the engine serves on, read from the container's own env.

    The serve line is a shell string, so the argv holds the literal
    "${PORT:-8600}" and cannot be parsed for a number; the resolved value
    is in the container's environment.
    """
    rc, out = run(["docker", "exec", container, "printenv", "PORT"], timeout=60)
    if rc == 0:
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                return int(line)
    rc, out = run(["docker", "inspect", container, "--format",
                   "{{range .Config.Env}}{{println .}}{{end}}"], timeout=60)
    if rc == 0:
        for line in out.splitlines():
            k, _, v = line.strip().partition("=")
            if k == "PORT" and v.strip().isdigit():
                return int(v.strip())
    return default


def container_metrics(container, port):
    """Read /metrics from INSIDE the container: no host port mapping needed."""
    code = (
        "import urllib.request;"
        f"print(urllib.request.urlopen('http://127.0.0.1:{port}/metrics',timeout=20)"
        ".read().decode('utf-8','replace'))"
    )
    rc, out = run(["docker", "exec", container, "python3", "-c", code], timeout=90)
    return out if rc == 0 else ""


def container_swiglu_limit(container):
    """The activation limit the ENGINE resolves from the checkpoint.

    This is the engine's own config path, not a re-read of the raw JSON, so
    it answers the question that matters: what did the running engine take
    as the clamped-SwiGLU limit.
    """
    code = (
        "from vllm.transformers_utils.config import get_config;"
        "print(get_config('/model', trust_remote_code=False).swiglu_limit)"
    )
    rc, out = run(["docker", "exec", container, "python3", "-c", code], timeout=180)
    if rc != 0:
        return None
    for line in out.splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            return float(line)
    return None


def metric_value(metrics_text, label, family=METRIC_FAMILY):
    """Read one label out of one metric family.

    Scoped to the family on purpose: an unscoped `label="..."` match would
    accept a same-named label from an unrelated family, and the engine's
    metric families do repeat label names across families.
    """
    pattern = rf'{re.escape(family)}\{{[^}}]*?\b{re.escape(label)}="([^"]*)"'
    m = re.search(pattern, metrics_text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--container", default=os.environ.get("DS41_CONTAINER", "ds41-flash-tp3"))
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()

    if not shutil.which("docker"):
        print("FAIL docker not found on PATH")
        return 1
    try:
        with open(args.manifest) as fh:
            spec = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"FAIL manifest readable ({exc.__class__.__name__})")
        return 1

    contract = spec.get("resolved_backends", {})
    expect = contract.get("expect", {})
    forbidden = contract.get("forbidden", {})

    port = args.port or container_port(args.container)
    logs = container_logs(args.container)
    metrics = container_metrics(args.container, port)
    limit = container_swiglu_limit(args.container)

    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)

    if not logs:
        check(False, f"engine log readable (docker logs {args.container})")
    if not metrics:
        check(False, f"engine /metrics readable (port {port})")

    # --- every expectation must be COVERED by a check ----------------------
    # A manifest key that no check looks at is a silently skipped
    # expectation: renaming a key in the manifest would turn its check
    # into a no-op instead of a failure. Uncovered keys fail here.
    covered = (set(LOG_MARKERS) | set(METRIC_KEYS) | set(CONFIG_KEYS)
               | set(CMD_MARKERS) | set(ENV_MARKERS) | set(ALLREDUCE_KEYS))
    for key in sorted(k for k in expect if k not in covered):
        check(False, f"expectation {key!r} is not covered by any check "
                     f"(add it to ALLREDUCE_KEYS, CMD_MARKERS, CONFIG_KEYS, "
                     f"ENV_MARKERS, LOG_MARKERS or METRIC_KEYS)")

    # --- the engine's own log: what it says it selected -------------------
    for key, template in LOG_MARKERS.items():
        want = expect.get(key)
        if want is None:
            continue
        marker = template.format(v=want)
        check(marker in logs, f"log: {key} == {want!r}"
              + ("" if marker in logs else f" — marker not found: {marker!r}"))

    # --- which all-reduce ACTUALLY engaged --------------------------------
    # Not a marker match on the requested backend: the engine prints the
    # backend it engaged, and that is the fact worth asserting.
    #
    # The engine prints one dispatch line PER GROUP (the TP group and the EP
    # group), and the groups are independent. Reading only the first line
    # would let a lane that engaged the collective on one group and fell
    # back on the other pass, so every line must agree.
    want_ar = expect.get("allreduce_backend")
    want_map = expect.get("allreduce_backend_by_group") or {}
    if want_ar is not None or want_map:
        want_norm = norm_backends(want_ar) if want_ar is not None else None
        # (backends, group) per dispatch line. The group is what makes a
        # mixed-lane failure actionable: 'tp:0' and 'ep:0' are independent
        # collectives, so "tp:0 engaged X" says which one moved.
        engaged = [(norm_backends(m.group(1)), m.group(2))
                   for m in ALLREDUCE_RE.finditer(logs)]
        if not engaged:
            check(False, f"log: allreduce_backend == {want_ar!r} — the dispatch "
                         f"line was not found in the engine log")
        else:
            wrong = []
            for g, grp in engaged:
                want = expected_ar(expect, grp) if want_map else want_ar
                if want is None:
                    continue
                if g != norm_backends(want):
                    wrong.append((g, grp, want))
            want_desc = (repr(want_map) if want_map else repr(want_ar))
            check(not wrong, f"log: allreduce_backend == {want_desc}"
                  + ("" if not wrong
                     else " — the engine engaged "
                          + ", ".join(
                              (f"group {grp!r}: " if grp else "")
                              + repr(", ".join(g)) + f" (expected {want!r})"
                              for g, grp, want in wrong)))
            if wrong:
                diag = re.search(r"RoCEnante initialization failed:[^\n]*", logs)
                if diag:
                    print("       the engine's own diagnostic: "
                          + diag.group(0).strip()[:200])
                print("       the one-shot RoCE collective needs BOTH RoCE rails "
                      "of every link; the shipped topology has one cable per "
                      "pair of boxes")

    # --- the container's own environment: what was FORWARDED --------------
    # and the command line: what was REQUESTED. A knob that was read but
    # never forwarded would pass every other check in this file.
    cmd = container_cmd(args.container)
    for key, flag in CMD_MARKERS.items():
        want = expect.get(key)
        if want is None:
            continue
        env_var = ENV_MARKERS.get(key)
        got = container_env_value(args.container, env_var) if env_var else None
        if got is not None:
            check(got == want, f"env: {key} == {want!r}"
                  + (f" (the container's {env_var})"
                     if got == want else f" — the container's {env_var} is {got!r}"))
            continue
        # The environment did not carry it. Fall back to the command line,
        # but ONLY a literal number: "${BLOCK_SIZE:-256}" is the same text
        # for every value and proves nothing.
        m = re.search(re.escape(flag) + r"\s+(\d+)", cmd)
        literal = int(m.group(1)) if m else None
        check(literal == want, f"cmd: {key} == {want!r}"
              + ("" if literal == want else
                 f" — no literal {flag} value in the container's command line, "
                 f"and {env_var} is not in its environment"))

    # --- the engine's own log: what it must NOT have selected -------------
    # The manifest is the source of truth for the markers; the defaults here
    # are the same strings, so the check still works against an older
    # manifest that predates the `forbidden` block.
    #
    # COVERAGE: a forbidden marker that sits in the manifest but in no
    # marker table is a silently DISABLED check. Adding one to the manifest
    # would then look like it pinned something and pin nothing at all --
    # the same class of defect as an uncovered `expect` key, so it fails
    # here. The `.why` keys are rationale, not markers, and are exempt.
    for key in sorted(k for k in forbidden
                      if not k.endswith(".why") and k not in FORBIDDEN_MARKERS):
        check(False, f"forbidden marker {key!r} is not covered by any check "
                     f"(add it to FORBIDDEN_MARKERS)")

    for key, default_marker in FORBIDDEN_MARKERS.items():
        if key.endswith(".why"):
            continue
        marker = forbidden.get(key, default_marker)
        if not marker:
            # An EMPTY marker would look for the empty string, which is
            # never in a log, so the check would pass for ANY log. That is
            # the same false-pass this whole file exists to prevent.
            check(False, f"forbidden marker {key!r} is empty — an empty marker "
                         f"is present in no log, so the check would pass "
                         f"whatever the engine did")
            continue
        check(marker not in logs, f"log: no {marker!r} (forbidden {key})"
              + ("" if marker not in logs else " — present in the log"))

    # --- the engine's own metrics -----------------------------------------
    for key, label in METRIC_KEYS.items():
        want = expect.get(key)
        if want is None:
            continue
        got = metric_value(metrics, label)
        # the manifest is JSON, so a number arrives as a number and the
        # metric arrives as a string: compare as text.
        check(got == str(want), f"metrics: {label} == {want!r}"
              + ("" if got == str(want) else f" — got {got!r}"))

    # --- the model's activation limit, as the engine resolved it ----------
    want_limit = expect.get("swiglu_limit")
    if want_limit is not None:
        check(limit is not None and abs(limit - float(want_limit)) < 1e-6,
              f"engine-resolved swiglu_limit == {want_limit!r}"
              + ("" if limit is not None else " — could not read it"))

    print("\nRESOLVED-BACKENDS " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
