# ds41-w3 — the coded expert-scales overlay (weights-side memory back)

An optional kernel overlay for this recipe's pinned image: five Python
kernel files copied over the b12x MoE shared-kernel directory at boot
(`flashinfer/experimental/b12x/moe/_shared/kernels/`). It packs the MoE
expert scales into a coded form on load — **buying back roughly 2.5 GiB
per rank of weights-side memory at stock-equivalent quality and speed**
(a full adoption battery (capability, degeneration, arithmetic, copy-fidelity, NaN); rollback = set `DS41_W3=0`).

## Why it matters on this hardware

GB10 is unified-memory: weights-side savings are host-memory savings.
On the stack this recipe ships without this overlay, four concurrent deep
fills exhausted headroom and a watch on free memory stopped the lane (the
README's memory section). With the overlay armed, the same class of
shapes rides a materially wider margin — eight concurrent deep fills
complete end to end. **This kit is what keeps the lane up under load, by
leaving it that margin.**

## What it changes

Five files, copied verbatim (md5-pinned in `OVERLAY-MD5`):

| file | role |
|---|---|
| `dynamic.py` | the fused routed low-precision MoE dynamic kernel (coded-scale paths) |
| `w4a8_phase1.py` / `w4a8_phase2.py` | the two-phase W4A8 expert paths |
| `tiny_decode.py` | the small-batch decode path (unchanged clamp semantics) |
| `w3_coded_common.py` | the shared coded-scale primitives |

No scheduler files, no engine patches, no config — the overlay touches
only the MoE kernel layer. The five files are self-contained (no imports
outside the package's own modules); `python3 -m py_compile` clean.

## Safety

- **Install is a copy, md5-pinned**: the launcher refuses a half-staged
  kit (`ABSENT` check per file) and prints each install line; `OVERLAY-MD5`
  lets any reader verify their staged copy.
- **Quality gates at adoption**: capability, degeneration, arithmetic,
  copy-fidelity, NaN — clean, plus the corruption battery.
- **The knob is fail-closed**: `DS41_W3` defaults to `1` (the measured
  posture); `DS41_W3=0` skips the copy entirely and the engine runs the
  stock kernel files — byte-stock behavior.

## Verify / apply manually (inside the container)

```sh
cd /path/to/this/kit
md5sum -c OVERLAY-MD5   # (macOS: md5 -r *.py and compare by eye)
# the launcher does this at boot; manually:
# cp <the five files> /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/_shared/kernels/
```

Measured on this hardware class: ~2.5 GiB/rank of weights-side memory
returned; expert-path speed stock-equivalent; the memory story is the
entire point. See the README's memory section for the headroom results
with the overlay armed.
