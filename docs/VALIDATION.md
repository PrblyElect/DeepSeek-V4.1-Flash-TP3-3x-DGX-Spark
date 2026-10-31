# VALIDATION — ds41-flash-tp3

Acceptance gates for the shipped configuration. Run after every boot.

## 1. Health

```sh
curl -s -o /dev/null -w "%{http_code}" http://HEAD_IP:8600/health
# expected: 200
```

## 2. Deterministic generation (temp=0)

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
# concurrency figure of roughly 3-4.5x at a 524,288-token context
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

## Pass criteria

All seven gates must pass. A failed gate is diagnosed forward from the
FIXLOG, never rolled back automatically.
