# VALIDATION — ds41-flash-tp3

Acceptance gates for the shipped configuration. Run after every boot.

**Where each gate runs.** Gates 1–5 and gate 9 are plain HTTP and run
from anywhere that can reach the head box. Gates 6–8 shell out to the
local `docker` binary and must run **on the head box**, in the staged
directory (`~/ds41-flash-tp3/`), where the container actually lives:

```sh
ssh SPARK_HEAD 'cd ~/ds41-flash-tp3 && python3 tp3-kit/verify-resolved-backends.py --container ds41-flash-tp3'
```

Gate 9 is the exception to "from anywhere": it is a plain HTTP call,
but it must be run **from the checkout** — its command names the kit
file by its checkout-relative path
(`python3 launch/tp3-kit/check-long-prompt-determinism.py …`). From the
staged directory the same file is `~/ds41-flash-tp3/tp3-kit/…`.

The kit is staged to the boxes by `ds41-flash-tp3-up stage-env`; it is
also in the recipe checkout under `launch/tp3-kit/`, which is the copy
to read but not the copy to run against a remote container.

## 1. Health

```sh
curl -s -o /dev/null -w "%{http_code}" http://HEAD_IP:8600/health
# expected: 200
```

## 2. Deterministic generation (temp=0) — short prompts

Send the same prompt three times; outputs must be byte-identical:

```sh
for i in 1 2 3; do
  curl -s http://HEAD_IP:8600/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"main","messages":[{"role":"user","content":"What is 17*23? Just the number."}],"max_tokens":30,"temperature":0}' | \
    jq -r '.choices[0].message.content'
done
# expected: identical output ×3 (e.g. "391")
```

**What this gate does not cover.** These prompts are short — they never
cross the chunked-prefill boundary. The failure mode that matters for
this model lives in the *last chunk* of a chunked prefill when that chunk
runs as a piecewise CUDA graph, so gate 9 is the one that reaches it.
Run both; this gate alone is not evidence that the long-prompt path is
correct.

## 3. Strict JSON compliance

```sh
curl -s http://HEAD_IP:8600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"main","messages":[{"role":"user","content":"Return a JSON object with key answer and value 42. Nothing else."}],"max_tokens":50,"temperature":0}' | \
  jq -r '.choices[0].message.content'
# expected: exactly {"answer":42} (reasoning stripped by parser)
```

## 4. Tool calling

```sh
curl -s http://HEAD_IP:8600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"main","messages":[{"role":"user","content":"What is 3+5? One word."}],"max_tokens":30,"temperature":0,"tools":[{"type":"function","function":{"name":"calc","description":"calculator","parameters":{"type":"object","properties":{"expr":{"type":"string"}}}}}],"tool_choice":"auto"}' | \
  jq -r '.choices[0].message.content'
# expected: a direct answer (no tool call needed for simple math); no error
```

## 5. Speed benchmark

```sh
time curl -s http://HEAD_IP:8600/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"main","messages":[{"role":"user","content":"Write a detailed paragraph about ocean currents."}],"max_tokens":250,"temperature":0}' | \
  jq '.usage.completion_tokens'
# expected: 250 tokens in ~8.5–9 s (~29 tok/s)
```

## 6. KV pool (from boot log)

```sh
docker logs ds41-flash-tp3 2>&1 | grep "GPU KV cache size"
# expected: roughly 1.6M-2.4M tokens (the pool is boot-labile), and a
# concurrency figure of roughly 3-4.7x at a 524,288-token context
```

## 7. Image identity

```sh
# the manifest digest must be the pinned one ...
docker image inspect eugr/spark-vllm-b12x:nightly-20261015 \
  --format '{{range .RepoDigests}}{{println .}}{{end}}' \
  | grep -F 'sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595'

# ... and the content identity (the layer list) must match too
docker image inspect eugr/spark-vllm-b12x:nightly-20261015 \
  --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | sha256sum
# expected: 41e4f9a3005bca761eb883c6565c1a3667cfbb014d49ec3d97735c7cfe146616
```

Both must hold on every box. The digest proves which manifest was
pulled; the layer-list hash proves which bytes are on disk — an image
imported rather than pulled can be content-identical while carrying no
manifest digest, and the engine-identity gate
(`launch/tp3-kit/engine-identity.json`) proves what executes inside
the container.

## 8. Resolved backends and the activation limit

The command line is not the behaviour: the engine chooses its MoE
backend, dense-linear kernel and attention backend itself. This gate
checks what it resolved to, against the contract in
`launch/tp3-kit/engine-identity.json`:

```sh
# on the head box, in the staged directory (the gate reads the LOCAL
# container, so it cannot be run against a remote one)
cd ~/ds41-flash-tp3
python3 tp3-kit/verify-resolved-backends.py --container ds41-flash-tp3
# expected: RESOLVED-BACKENDS PASS
```

It fails if the MoE backend is the b12x one (that path drops the model's
clamped-SwiGLU limit on small-batch decode, with no error signal), if the
engine resolved a different activation limit, if the TP=3 head padding is
not in effect, if the resolved block sizes do not match the contract, or
if the all-reduce transport that actually engaged is not the one the
manifest expects.

### The all-reduce expectation

One expectation is about the transport rather than a kernel, and it needs
a word of its own because it is the one that a **physical** change can
break.

The launcher sets `VLLM_ENABLE_ROCE_ALLREDUCE=1`, and the image ships a
one-shot RoCE collective (`B12X_ROCENANTE`) that needs both RoCE rails of
a link to reach the peer. The topology this recipe ships has one cable
per leg, so that collective cannot form: the engine logs the failure and
engages `PYNCCL` instead. The gate asserts the transport that **actually
engaged**, on every group the engine reports, against
`resolved_backends.expect.allreduce_backend` in the manifest.

On the shipped topology the expected value is `PYNCCL`, and the published
figures were all measured on that path.

If you re-cable a second rail per leg, this gate will **fail**, and it is
meant to: the engine will engage the collective, and the expectation now
describes a lane that no longer exists. Update
`launch/tp3-kit/engine-identity.json`
(`resolved_backends.expect.allreduce_backend`) to `B12X_ROCENANTE` at the
same time, and the gate then asserts the new transport. The failure
message prints the engine's own diagnostic so the reason is in front of
you rather than inferred.

Every expectation is read from the engine's own startup log, its own
`/metrics` and its own resolved config — and the one expectation that is
about what was *requested* rather than what was resolved is read from the
container's own environment, where the launcher forwards it. A missing
marker is a **failure, not a skip**: if the engine changes what it
prints, the gate says so instead of quietly passing. Each file that
prints one of those markers is hash-pinned in
`launch/tp3-kit/engine-identity.json`, so the format cannot change under
the gate unnoticed.

Run it after the engine is up — the in-container identity gate runs
*before* it, and cannot see this.

## 9. Long-prompt determinism (crosses the chunked-prefill boundary)

```sh
python3 launch/tp3-kit/check-long-prompt-determinism.py --base-url http://HEAD_IP:8600
# expected: LONG-PROMPT-DETERMINISM PASS
```

This builds a prompt of exactly `max_num_batched_tokens + 10` tokens —
the shape in which the last chunk of a chunked prefill is small enough to
run as a piecewise CUDA graph — asks for it three times, and compares the
generated **token sequence** rather than the text.

The prompt is built from token IDs taken from the engine's own tokenizer
and sent as those IDs, so the count is exact by construction; the gate
asserts the engine's own prompt count. A fresh nonce at the head of the
prompt keeps the first call genuinely cold even when the gate is run
twice against the same lane.

The first call is cold and the second and third hit the prefix cache, so
a replay divergence shows up between the first and the later ones and not
between the later ones. The decode **logprobs** are compared as well as
the token IDs: at temperature 0 a logprob shift need not change the
argmax token, so comparing tokens alone cannot see a replay drift. The
measured drift is printed on every run — record it in the FIXLOG on the
first run against a lane you are accepting, so the allowance
(`--logprob-tol`, default 0.05) can be tightened to what the lane
actually produces rather than left as a guess.

It sends an 8k-token prefill three times. Run it against a lane you are
accepting, not against one carrying production traffic.

## Pass criteria

All nine gates must pass. A failed gate is diagnosed forward from the
FIXLOG, never rolled back automatically.
