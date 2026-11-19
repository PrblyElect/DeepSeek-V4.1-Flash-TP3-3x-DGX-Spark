#!/usr/bin/env python3
"""Golden/behavioral test for the DS41 session-prefix pin .

FAILS ON OLD CODE by construction (the patched methods carry the DS41-PIN
markers; the stock ones do not).

Runs in-container (--vllm-root) or against a patched-tree copy
(--copy-root pointing at a dir containing v1/core/{block_pool.py,
kv_cache_manager.py}). Behavioral cases drive the PATCHED methods on a
stub pool built from the REAL FreeKVCacheBlockQueue + KVCacheBlock
classes (pure data structures, no CUDA):
  1  the DEFER moves a pinned front block to the tail (get_new_blocks
     hands out the non-pinned block first, the pinned one last)
  2  expired pins stop deferring (lazy expiry)
  3  the MARK records freed CACHED blocks when armed, respects the cap
  4  non-cached blocks are not recorded
  5  the ARM sets the marker attrs (kv_cache_manager.free on a stub)
  5b UNTAGGED (priority 0) ALSO arms — the universal pin, no opt-in
  5c explicitly-batch (priority > PIN_PRIO_MAX) does NOT arm
  3c cap RECYCLING: at cap, expired pins purge and the earliest-deadline
     victim is dropped per insert (a big session makes its own room)
  3d/3e refresh fast-path: an already-pinned block only refreshes its
     deadline — no victims even at exactly-cap; a third-party oldest
     pin survives a foreign refresh (the blocker-discriminator case)
  6  wiring: all three markers present in both patched sources
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


def load(copy_root: str | None, vllm_root: str | None):
    base = Path(copy_root) if copy_root else Path(vllm_root)
    sys.path.insert(0, str(base.parent))
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    import vllm.v1.core.block_pool as bp_mod

    return base, KVCacheBlock, FreeKVCacheBlockQueue, bp_mod


def make_stub_pool(Queue, KVBlock, n=4):
    """A minimal object exposing what the patched methods touch."""
    blocks = [KVBlock(block_id=i) for i in range(n)]
    q = Queue(blocks)
    for b in blocks:
        b.ref_cnt = 0
        b._block_hash = None
        b._block_hash_num_tokens = None
        b.is_null = False
        b.pool = None
    class _StubPool:
        pass

    pool = _StubPool()
    pool.free_block_queue = q
    pool.blocks = blocks
    pool.enable_caching = True
    pool.boundary_checkpoints = None
    pool.metrics_collector = None
    pool.ds41_pinned_ids = set()
    pool.ds41_pin_until = {}
    pool.ds41_pin_cap = 6500
    pool.ds41_pin_mark_until = None
    def _count_free():
        n = 0
        node = q.fake_free_list_head.next_free_block
        while node is not None and node is not q.fake_free_list_tail:
            n += 1
            node = node.next_free_block
        return n

    pool.get_num_free_blocks = _count_free
    pool._maybe_evict_cached_block = lambda block: True
    pool._notify_reuse = lambda blocks: None
    pool._reuse_watchers = {}
    return pool, blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vllm-root", default=None)
    g.add_argument("--copy-root", default=None)
    a = ap.parse_args()
    base, KVBlock, Queue, bp_mod = load(a.copy_root, a.vllm_root)

    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    from vllm.v1.core.block_pool import BlockPool

    # 1-2: the defer, driven via the real method on a stub pool
    pool, blocks = make_stub_pool(Queue, KVBlock, n=3)
    blocks[0]._block_hash = b"h0"  # front block cached + pinned
    blocks[1]._block_hash = None
    pool.ds41_pinned_ids = {0}
    pool.ds41_pin_until = {0: 1e18}  # far future
    got = BlockPool.get_new_blocks(pool, 1)
    check("1 defer: pinned front block NOT handed out first", got[0].block_id != 0, str([b.block_id for b in got]))
    pool2, _ = make_stub_pool(Queue, KVBlock, n=3)
    pool2.blocks[0]._block_hash = b"h0"
    pool2.ds41_pinned_ids = {0}
    pool2.ds41_pin_until = {0: 1.0}  # long expired
    got2 = BlockPool.get_new_blocks(pool2, 1)
    check("2 expired pin stops deferring", got2[0].block_id == 0, str([b.block_id for b in got2]))

    # 3-4: the mark, via the real free_blocks on a stub
    pool3, blocks3 = make_stub_pool(Queue, KVBlock, n=3)
    blocks3[0]._block_hash = b"cached"
    blocks3[1]._block_hash = None
    for b in blocks3:
        b.ref_cnt = 1
    pool3.ds41_pin_mark_until = 1e18
    BlockPool.free_blocks(pool3, [blocks3[0], blocks3[1], blocks3[2]])
    check("3 mark: cached freed block recorded", 0 in pool3.ds41_pinned_ids and pool3.ds41_pin_until.get(0, 0) > 0)
    check("4 mark: non-cached block NOT recorded", 1 not in pool3.ds41_pinned_ids)
    # cap respected
    pool4, blocks4 = make_stub_pool(Queue, KVBlock, n=3)
    for b in blocks4:
        b._block_hash = b"x"
    for b in blocks4:
        b.ref_cnt = 1
    pool4.ds41_pin_cap = 1
    pool4.ds41_pin_mark_until = 1e18
    BlockPool.free_blocks(pool4, list(blocks4))
    check("3b mark: cap respected (1 of 3)", len(pool4.ds41_pinned_ids) == 1)

    # 3c: recycling at cap — expired purged first, then earliest-deadline victim
    pool4b, blocks4b = make_stub_pool(Queue, KVBlock, n=3)
    for b in blocks4b:
        b._block_hash = b"x"
        b.ref_cnt = 1
    pool4b.ds41_pinned_ids = {7}
    pool4b.ds41_pin_until = {7: 1.0}  # long expired
    pool4b.ds41_pin_cap = 1
    pool4b.ds41_pin_mark_until = 1e18
    BlockPool.free_blocks(pool4b, [blocks4b[0]])
    check("3c recycle: expired pin purged, new block takes the slot",
          7 not in pool4b.ds41_pinned_ids and len(pool4b.ds41_pinned_ids) == 1 and blocks4b[0].block_id in pool4b.ds41_pinned_ids)
    # victim path: a fresh (unexpired) pin is dropped when nothing is expired
    pool4c, blocks4c = make_stub_pool(Queue, KVBlock, n=3)
    for b in blocks4c:
        b._block_hash = b"x"
        b.ref_cnt = 1
    pool4c.ds41_pinned_ids = {9}
    pool4c.ds41_pin_until = {9: 1e18}  # far future — must be victim-dropped
    pool4c.ds41_pin_cap = 1
    pool4c.ds41_pin_mark_until = 1e18
    BlockPool.free_blocks(pool4c, [blocks4c[0]])
    check("3c recycle: earliest-deadline victim dropped for the newcomer",
          9 not in pool4c.ds41_pinned_ids and blocks4c[0].block_id in pool4c.ds41_pinned_ids and len(pool4c.ds41_pinned_ids) == 1)

    # 3d refresh fast-path: re-free of ALREADY-PINNED blocks at cap drops no
    # victims and refreshes deadlines (a resident session's turn-end is safe)
    pool4d, blocks4d = make_stub_pool(Queue, KVBlock, n=2)
    for b in blocks4d:
        b._block_hash = b"x"
        b.ref_cnt = 1
    pool4d.ds41_pinned_ids = {0, 1}
    pool4d.ds41_pin_until = {0: 1e18, 1: 1e18}
    pool4d.ds41_pin_cap = 2  # exactly at cap
    pool4d.ds41_pin_mark_until = 5e18  # the new (later) deadline
    BlockPool.free_blocks(pool4d, list(blocks4d))
    check("3d refresh at cap: no victims, deadlines refreshed",
          pool4d.ds41_pinned_ids == {0, 1}
          and pool4d.ds41_pin_until.get(0) == 5e18 and pool4d.ds41_pin_until.get(1) == 5e18
          and len(pool4d.ds41_pinned_ids) == 2)

    # 3e refresh fast-path DISCRIMINATOR: a third-party pin with the OLDEST
    # deadline is NOT in the freed batch — the pre-fix code dropped it (min()
    # victim); the fast path must leave it untouched while refreshing block 0.
    pool4e, blocks4e = make_stub_pool(Queue, KVBlock, n=1)
    blocks4e[0]._block_hash = b"x"
    blocks4e[0].ref_cnt = 1
    pool4e.ds41_pinned_ids = {0, 9}
    pool4e.ds41_pin_until = {0: 2e18, 9: 1e18}  # 9 = oldest, NOT in batch
    pool4e.ds41_pin_cap = 2  # exactly at cap
    pool4e.ds41_pin_mark_until = 5e18  # the new (later) deadline
    BlockPool.free_blocks(pool4e, [blocks4e[0]])
    check("3e third-party oldest pin SURVIVES a foreign refresh",
          9 in pool4e.ds41_pinned_ids and pool4e.ds41_pin_until.get(9) == 1e18
          and pool4e.ds41_pin_until.get(0) == 5e18
          and pool4e.ds41_pinned_ids == {0, 9})

    # 5: the ARM (kv_cache_manager.free on a stub manager + stub request)
    import vllm.v1.core.kv_cache_manager as km_mod

    class StubReq:
        request_id = "r1"
        priority = -1
        boundary_checkpoint = None
        boundary_checkpoint_blocks = None

    class StubCoord:
        def __init__(self, pool):
            self.freed = False
            self.mark_at_free = "never"
            self._pool = pool

        def free(self, rid):
            self.freed = True
            self.mark_at_free = getattr(self._pool, "ds41_pin_mark_until", None)

    pool5, _ = make_stub_pool(Queue, KVBlock, n=1)
    pool5.free_blocks = lambda blks: BlockPool.free_blocks(pool5, blks)
    mgr = types.SimpleNamespace(
        block_pool=pool5,
        coordinator=StubCoord(pool5),
        _pop_boundary_blocks=lambda req: [],
        _boundary_allocations={},
        _boundary_readers={},
    )
    km_mod.KVCacheManager.free(mgr, StubReq())
    check("5 arm: marker LIVE during free, disarmed after",
          mgr.coordinator.freed and isinstance(mgr.coordinator.mark_at_free, float)
          and pool5.ds41_pin_mark_until is None and hasattr(pool5, "ds41_pinned_ids"))

    # 5b: UNTAGGED (priority 0) arms — the universal pin
    class UntaggedReq(StubReq):
        priority = 0

    pool5b, _ = make_stub_pool(Queue, KVBlock, n=1)
    pool5b.free_blocks = lambda blks: BlockPool.free_blocks(pool5b, blks)
    mgr_b = types.SimpleNamespace(
        block_pool=pool5b,
        coordinator=StubCoord(pool5b),
        _pop_boundary_blocks=lambda req: [],
        _boundary_allocations={},
        _boundary_readers={},
    )
    km_mod.KVCacheManager.free(mgr_b, UntaggedReq())
    check("5b UNTAGGED arm: priority 0 pins too",
          mgr_b.coordinator.freed and isinstance(mgr_b.coordinator.mark_at_free, float)
          and pool5b.ds41_pin_mark_until is None)

    # 5c: explicitly-batch (priority 2 > PIN_PRIO_MAX 1) opts out
    class BatchReq(StubReq):
        priority = 2

    pool5c, _ = make_stub_pool(Queue, KVBlock, n=1)
    pool5c.free_blocks = lambda blks: BlockPool.free_blocks(pool5c, blks)
    mgr_c = types.SimpleNamespace(
        block_pool=pool5c,
        coordinator=StubCoord(pool5c),
        _pop_boundary_blocks=lambda req: [],
        _boundary_allocations={},
        _boundary_readers={},
    )
    km_mod.KVCacheManager.free(mgr_c, BatchReq())
    check("5c batch opt-out: priority 2 does NOT arm",
          mgr_c.coordinator.freed and mgr_c.coordinator.mark_at_free is None
          and pool5c.ds41_pin_mark_until is None)

    # 6: wiring in the patched sources
    bp_src = (base / "v1/core/block_pool.py").read_text(encoding="utf-8")
    km_src = (base / "v1/core/kv_cache_manager.py").read_text(encoding="utf-8")
    check(
        "6 wiring: all three markers present",
        "DS41-PIN DEFER" in bp_src and "DS41-PIN MARK" in bp_src and "DS41-PIN ARM" in km_src,
    )
    check("6b env knobs present",
          "VLLM_DS41_PIN_TTL_S" in km_src and "VLLM_DS41_PIN_MAX_BLOCKS" in km_src
          and "VLLM_DS41_PIN_PRIO_MAX" in km_src and "_ds41_prio_max" in km_src)

    print("golden:", "PASS" if not failures else f"FAIL ({failures})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
