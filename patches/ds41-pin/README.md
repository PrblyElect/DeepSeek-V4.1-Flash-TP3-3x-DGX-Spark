# ds41-pin — the session-prefix pin (resumed sessions keep their first token)

An optional scheduler patch for this recipe's pinned image: when a chat
session's cached prefix blocks would be evicted to make room for new
prefills, the pin **marks them for keep-instead-of-reuse** and recycles
other memory first. A conversation you come back to — at any size up to
the whole pool — answers from cache instead of re-prefilling its entire
history.

## Why

The KV pool this recipe ships (~5.1M tokens) is large enough that whole-
pool fills are a supported shape — and a fill of that size evicts
essentially every cached session. Without the pin, a resumed session
after such a fill pays its full prefill again before the first token
(a ~350K-token session: a ~2-minute wait on this stack). With the pin,
the freed blocks of every request are marked; the recycler prefers
unmarked memory, and resumed sessions keep first-token times at the
cache-hit class after whole-pool eviction storms — measured at pool
scale, all sessions, no priority model and no per-request opt-in.

## What it changes

Three anchored insertions in the pinned image's KV-cache recycling path
(`vllm/v1/core/block_pool.py` and `vllm/v1/core/kv_cache_manager.py` —
byte-pinned below), each pure bookkeeping: a pin-set on free, a preference in the
recycler's victim choice, and a cap-driven re-pin on access. No numerics,
no kernel changes; the model's outputs are untouched by construction.

Safety properties, same pattern as this recipe's other kits:

- **md5-pinned both ways**: the installer pins the pristine and patched
  bytes of each target file; any foreign tree is refused; `--revert`
  restores byte-exact.
- **A CPU-only golden** proves the patched recycling logic's shape
  (it runs in-container at boot; the installer itself proves the
  lifecycle: refuse a foreign tree, apply, verify, byte-exact revert).
- The launcher installs the kit only when `DS41_PIN=1`; the patch itself
  has no runtime switch (unset knobs run the measured defaults).

## Apply / revert (inside the container, or via the launcher's kit mount)

```sh
python3 install.py --vllm-root /usr/local/lib/python3.12/dist-packages/vllm --apply
python3 golden_test.py --vllm-root /usr/local/lib/python3.12/dist-packages/vllm
python3 install.py --vllm-root /usr/local/lib/python3.12/dist-packages/vllm --revert
```

Measured on this hardware class (on the fully-kitted stack): whole-pool
eviction storms (the pool filled past a set of idle sessions) leave
resumed sessions at the cache-hit first-token class; acceptance,
corruption, and NaN batteries clean on the kit's own carding boot. On
the stack this recipe ships, the warm-resume behavior (cache-hit class,
~1 s) is carded directly; pool-filling eviction storms need fill shapes
beyond the shipped stack's memory ceiling (see the README's memory
section) — the pin is bookkeeping and adds no memory cost of its own. Enabling costs a small
bookkeeping overhead at free time (bounded, measured in the
sub-millisecond-per-step class).

Knobs (in-container env, all optional, sensible defaults): the pin
TTL, the per-session block cap, and the priority ceiling; unset values
run the measured defaults.
