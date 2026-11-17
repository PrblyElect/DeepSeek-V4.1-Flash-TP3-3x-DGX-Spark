# ds41mhc-tile — promote the vendor's own mHC prefill tile to 2048-token steps

An optional kernel-level patch for this recipe's pinned image: a
**one-line change to flashinfer's mHC tuning default** that promotes the
tile geometry the vendor already ships for hidden-5120 models at exactly
4096 prefill tokens (`M64/N24/K64`, two TMA stages, four K-splits) to
**every plan of 2048 tokens or more** — replacing the decode-shaped
fallback (`M32/N8/K256`, one stage, three N-tiles with the A-operand
re-streamed per tile) that 2048-token steps otherwise run.

## Why

The mHC plan capacity is the engine's batched-token budget, so a lane
running 2048-token prefill steps (the decode-under-prefill share's shipped
setting) compiles its mHC plans from the *fallback* branch — a tile the
engine's **own tuner predicates reject** for this shape class (the
predicate algebra demands wider N-tiles at prefill token counts). The
good geometry already exists in the same file, one branch away, shipped
by the vendor for the 4096 case.

Measured on this hardware class (a rank-0 profile of 8×2048-token
chunks, whole-cycle normalized): the projection kernel runs **~2×** with
the promotion — its share of a prefill chunk drops from ~5% to ~3%
(the honest prize is a few percent of the chunk, not of the wall). The
trace's launch geometry change (grid `(64,3,1)`→`(32,1,4)`, block
96→160) is the independent proof the patch armed, visible in any kernel
profile without trusting the timing.

## Safety

- **One line, md5-pinned both ways**: the installer pins the pristine
  and patched bytes of the single target file; any foreign tree is
  refused; `--revert` restores byte-exact.
- **A torch-free golden** (~50 checks) runs the image's real predicate
  algebra over both lifecycle states: it proves the shipped default
  FAILS the engine's own efficiency predicates at these token counts,
  that the promoted tile PASSES them, and that no collateral geometry
  changes. Run it inside the container against the pinned image before
  trusting any boot.
- Quality: acceptance batteries (capability, long-form degeneration,
  arithmetic, copy-fidelity, NaN) measured clean on the patched boot —
  the patch changes a tiling, not the numerics contract (the TF32 +
  split-FP32 dtype path is untouched).

## Apply / revert (inside the container, or via your bootstrap)

```sh
python3 install.py --flashinfer-root /usr/local/lib/python3.12/dist-packages/flashinfer --apply
python3 golden_test.py --flashinfer-root /usr/local/lib/python3.12/dist-packages/flashinfer --expect patched
python3 install.py --flashinfer-root /usr/local/lib/python3.12/dist-packages/flashinfer --revert
```

The kit is self-contained; the md5 pins bind it to the pinned image
digest this recipe ships. If a future image changes the target file, the
installer refuses (fail-closed) rather than guessing.
