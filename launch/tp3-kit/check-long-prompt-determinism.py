#!/usr/bin/env python3
"""check-long-prompt-determinism.py — the long-prompt correctness gate.

WHY THIS IS A SEPARATE GATE. The recipe's determinism gate compares SHORT
prompts. Short prompts never cross the chunked-prefill boundary, so they
cannot reach the failure mode that matters: upstream vLLM #62811 reports
that for DeepSeek-V4.1 the LAST chunk of a chunked prefill, when it runs as
a piecewise CUDA graph, produces the WRONG output — "wrong from the first
generated token" — with `enforce_eager=True` exact on the same request.
The upstream repro is an 8202-token prompt at `max_num_batched_tokens=8192`,
so the last chunk is 10 tokens. A short-prompt gate cannot see this class of
bug at all.

WHAT IS COMPARED, and why each one is there:

  1. THE GENERATED TOKENS, not just the text. The text is a lossy view; the
     token sequence is the thing the engine actually produced, and a
     first-token divergence is exactly upstream's signature.
  2. THE SAME REQUEST TWICE — a run-to-run divergence on the same request
     (a kernel race, a non-deterministic top-k) shows here.
  3. THE SAME REQUEST A THIRD TIME, WITH THE PREFIX CACHE WARM — upstream
     vLLM #64584 reports that a prefix-cache hit changes DS-V4.1 decode
     logprobs, because the sliding-window KV is rebuilt from a partial
     replay. A lane that serves a shared long prefix lives or dies on this.
  4. THE DECODE LOGPROBS, cold against cached. A logprob shift need not
     change the argmax token, so comparing token IDs alone cannot see the
     replay drift. The cold run's logprobs are compared against the cached
     run's, within a tolerance.

A FRESH NONCE is put at the head of the prompt on every run, so the first
call is genuinely cold even when the gate is run twice against the same
lane. Without it, a repeat run would compare two cached runs and the
cold-against-cached check would quietly stop testing anything.

The prompt is built to land on EXACTLY `max_num_batched_tokens + 10` tokens
(the upstream repro's shape: a 10-token last chunk), by taking token IDs
from the engine's own tokenizer and sending those IDs as the prompt.

WHEN TO RUN IT. This sends an 8k-token prefill three times. On a lane that
is serving production traffic that is real load: run it in a boot window,
against a lane you are accepting, not against a lane that is carrying
someone's work. The script prints the per-call latency so the cost is
visible.

    python3 check-long-prompt-determinism.py --base-url http://127.0.0.1:8600

Exit code 0 means every comparison held.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "engine-identity.json")

# A deterministic corpus. Numbered lines so the text is not a repeated
# n-gram: a repetitive prompt can tokenize and cache differently from a
# real one, and this gate is about the engine, not about the prose.
CORPUS = [
    "Line {n:05d}: the harbour lights came on before the fog had finished settling.",
    "Line {n:05d}: a cartographer marked the ridge, then crossed out the mark.",
    "Line {n:05d}: two hundred crates of salt went up the coast road each autumn.",
    "Line {n:05d}: the old pump worked until somebody sealed it with mortar.",
]


def post(url, payload, timeout=600):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build_prompt_ids(base, model, target_tokens, nonce):
    """Token IDs for a prompt of EXACTLY `target_tokens` tokens.

    The target has to be exact: the upstream repro's shape is a 10-token
    LAST chunk, so "close enough" is not enough — the gate has to land
    where the bug lives.

    The IDs come from the engine's own tokenizer and are then sent as the
    prompt, so the engine counts exactly what was asked for.

    A character-level search is deliberately NOT used. Byte-level BPE
    counts are not monotonic in the prefix length — a longer prefix can
    tokenize into FEWER tokens — so a binary search over characters has no
    correctness guarantee, and the walk that follows it can burn thousands
    of round trips before it lands or gives up. Selecting token IDs makes
    the count exact by construction, and the engine's own prompt count is
    asserted afterwards.
    """
    lines = [f"nonce {nonce}"]
    i = 0
    while True:
        lines.append(CORPUS[i % len(CORPUS)].format(n=i))
        i += 1
        if i % 64 == 0 and len("\n".join(lines)) > target_tokens * 8:
            break
    text = "\n".join(lines)

    data = post(f"{base}/tokenize", {"model": model, "prompt": text})
    ids = data.get("tokens") or []
    if len(ids) < target_tokens:
        raise SystemExit(f"corpus too short: tokenized to {len(ids)} tokens, "
                         f"need {target_tokens}")
    return ids[:target_tokens]


def complete(base, model, prompt_ids, max_tokens):
    t0 = time.time()
    data = post(f"{base}/v1/completions", {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "logprobs": 1,
    })
    dt = time.time() - t0
    ch = (data.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    return {
        "text": ch.get("text", ""),
        "tokens": lp.get("tokens"),
        "token_logprobs": lp.get("token_logprobs"),
        "prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
        "seconds": dt,
    }


def logprob_drift(a, b):
    """The largest absolute difference between two token_logprobs lists.

    None when the two runs are not comparable (a missing list, or two runs
    that generated a different number of tokens).
    """
    la = a.get("token_logprobs")
    lb = b.get("token_logprobs")
    if not la or not lb or len(la) != len(lb):
        return None
    return max(abs(x - y) for x, y in zip(la, lb))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8600")
    ap.add_argument("--model", default="")
    ap.add_argument("--mnbt", type=int, default=0,
                    help="max_num_batched_tokens; 0 = read it from the manifest")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--last-chunk", type=int, default=10,
                    help="tokens in the final prefill chunk (upstream used 10)")
    ap.add_argument("--logprob-tol", type=float, default=5e-2,
                    help="largest accepted logprob drift between the cold and "
                         "the cached run")
    ap.add_argument("--nonce", default="",
                    help="prompt nonce; the default is fresh on every run so "
                         "the first call is genuinely cold")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)

    model = args.model
    if not model:
        try:
            data = json.loads(urllib.request.urlopen(f"{base}/v1/models", timeout=30).read().decode())
            model = (data.get("data") or [{}])[0].get("id", "")
        except Exception:
            model = ""
    if not model:
        print("FAIL could not determine the served model name")
        return 1

    mnbt = args.mnbt
    if not mnbt:
        try:
            with open(args.manifest) as fh:
                spec = json.load(fh)
            mnbt = int(spec.get("resolved_backends", {}).get("expect", {})
                       .get("max_num_batched_tokens", 8192))
        except (OSError, ValueError, TypeError):
            mnbt = 8192
    target = mnbt + args.last_chunk

    nonce = args.nonce or time.strftime("%Y%m%dT%H%M%S")
    print(f"model={model}  max_num_batched_tokens={mnbt}  "
          f"target prompt={target} tokens  nonce={nonce}")

    prompt_ids = build_prompt_ids(base, model, target, nonce)
    check(len(prompt_ids) == target,
          f"prompt is {target} token IDs (got {len(prompt_ids)})")
    if len(prompt_ids) <= mnbt:
        print("FAIL the prompt does not cross the chunked-prefill boundary; "
              "this gate would pass without testing anything")
        return 1

    a = complete(base, model, prompt_ids, args.max_tokens)
    b = complete(base, model, prompt_ids, args.max_tokens)
    c = complete(base, model, prompt_ids, args.max_tokens)

    print(f"  timings: {a['seconds']:.1f}s / {b['seconds']:.1f}s / {c['seconds']:.1f}s")

    check(a["prompt_tokens"] == target,
          f"engine counted the prompt as {target} (got {a['prompt_tokens']})")

    # A comparison of two EMPTY sequences always passes, so the thing being
    # compared has to be shown to exist before it is compared.
    if a["tokens"] is not None:
        check(len(a["tokens"]) > 0, "the engine returned the generated token sequence")
        print(f"  generated: {' '.join(str(t) for t in a['tokens'][:12])}"
              f"{' ...' if len(a['tokens']) > 12 else ''}")
    else:
        check(len(a["text"]) > 0, "the engine returned generated text")
        print(f"  generated: {a['text'][:80]!r}")

    # 1: the same request twice must produce the same tokens.
    if a["tokens"] is not None and b["tokens"] is not None:
        check(a["tokens"] == b["tokens"],
              "run 1 and run 2 produced the same token sequence"
              + ("" if a["tokens"] == b["tokens"]
                 else f"\n         run1={a['tokens'][:12]}\n         run2={b['tokens'][:12]}"))
    else:
        check(a["text"] == b["text"], "run 1 and run 2 produced the same text")

    # 2: the third call, with the prefix cache warm from the first.
    if a["tokens"] is not None and c["tokens"] is not None:
        check(a["tokens"] == c["tokens"],
              "run 1 and the prefix-cached run 3 produced the same token sequence"
              + ("" if a["tokens"] == c["tokens"]
                 else f"\n         run1={a['tokens'][:12]}\n         run3={c['tokens'][:12]}"))
    else:
        check(a["text"] == c["text"], "run 1 and the prefix-cached run 3 produced the same text")

    # 3: the decode logprobs, cold against cached. A logprob shift need not
    # change the argmax token, so the token comparison above cannot see it.
    drift = logprob_drift(a, c)
    if drift is None:
        check(False, "the engine returned comparable decode logprobs for the "
                     "cold and the cached run"
                     + (" — one of the runs carried no token_logprobs"
                        if not a.get("token_logprobs") or not c.get("token_logprobs")
                        else " — the two runs generated a different number of tokens"))
    else:
        check(drift <= args.logprob_tol,
              f"decode logprobs agree within {args.logprob_tol} "
              f"(largest drift {drift:.6g})"
              + ("" if drift <= args.logprob_tol
                 else " — a prefix-cache hit changed the decode logprobs"))

    drift23 = logprob_drift(b, c)
    if drift23 is not None:
        check(drift23 <= args.logprob_tol,
              f"the two cached runs agree within {args.logprob_tol} "
              f"(largest drift {drift23:.6g})")

    # The engine's own view of the cache, if it exposes one. /metrics is
    # PROMETHEUS TEXT, not JSON -- parsing it as JSON always fails, and a
    # failure swallowed here would silently drop the note.
    try:
        m = urllib.request.urlopen(f"{base}/metrics", timeout=30).read().decode("utf-8", "replace")
        hit = re.search(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+(\S+)', m)
        if hit:
            print(f"  note: prefix cache queries counter reads {hit.group(1)}")
    except Exception:
        pass

    print("\nLONG-PROMPT-DETERMINISM " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
