"""ds41 W3 coded-scale helpers shared by the dynamic and phase kernels."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass.cutlass_dsl import Int32, Int64
from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    get_ptr_as_int64,
    ld_shared_u32,
)


_DS41_W3 = __import__("os").environ.get("DS41_W3", "").strip() not in ("", "0")



@cute.jit
def _w4a8_stage_coded_sfb_half(
    sfb_u8: cute.Tensor,
    smem_base: Int32,
    codes_tile_w: Int64,
    bases_off_w: Int64,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """Stage one coded SFB half into a 512B slot: 256B codes + 128B bases.

    24 cp.async-16B transfers (the stock half stages 32); the consumer
    decodes code(row) + base(row) from the slot (base region at +256).
    """
    for i in cutlass.range_constexpr((24 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(16):
            cp_async4_shared_global(
                smem_base + (idx << Int32(4)),
                get_ptr_as_int64(sfb_u8, codes_tile_w + Int64(idx * 4)),
            )
        elif idx < Int32(24):
            b = idx - Int32(16)
            cp_async4_shared_global(
                smem_base + Int32(256) + (b << Int32(4)),
                get_ptr_as_int64(sfb_u8, bases_off_w + Int64(b * 4)),
            )


@cute.jit
def _w4a8_stage_coded_sfb_full(
    sfb_u8: cute.Tensor,
    smem_base: Int32,
    codes_tile_w: Int64,
    bases_off_w: Int64,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
):
    """Stage one coded SFB tile into a 1KB slot: 512B codes + 256B bases."""
    for i in cutlass.range_constexpr((48 + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(32):
            cp_async4_shared_global(
                smem_base + (idx << Int32(4)),
                get_ptr_as_int64(sfb_u8, codes_tile_w + Int64(idx * 4)),
            )
        elif idx < Int32(48):
            b = idx - Int32(32)
            cp_async4_shared_global(
                smem_base + Int32(512) + (b << Int32(4)),
                get_ptr_as_int64(sfb_u8, bases_off_w + Int64(b * 4)),
            )


@cute.jit
def _ds41_bases_w13(
    sfb_u8: cute.Tensor,
    row_counts: cute.Tensor,
    b_w13_u32: cute.Tensor,
    expert: Int32,
    rp_tile: Int32,
    rp_nt: Int32,
    rp_half: Int32,
    pkt: Int32,
    kt_count: Int32,
) -> Int64:
    """Word offset (u32 words) of a half-tile's bases. E-free: the expert's
    codes offset is recovered from rp_tile (it folds the expert), and each
    expert's bases ride after its own codes."""
    nt13 = b_w13_u32.shape[0] // 256
    cb_w = nt13 * kt_count * 128
    rows_w = nt13 * 64
    e_cb_w = Int64(rp_tile) * Int64(128) - (
        Int64(rp_nt) * Int64(kt_count) + Int64(pkt)
    ) * Int64(128)
    # defensive clamps (predicated sites pass garbage indices; the
    # selected-away result may be wrong, never a fault)
    expert_c = max(Int32(0), min(Int32(expert), Int32(row_counts.shape[0]) - Int32(1)))
    off = (
        e_cb_w
        + Int64(expert_c) * Int64(rows_w)
        + Int64(cb_w)
        + Int64(min(max(rp_nt, Int32(0)), nt13 - Int32(1))) * Int64(64)
        + Int64(min(max(rp_half, Int32(0)), Int32(1))) * Int64(32)
    )
    # the plane tensor is the flat U32 repacked plane, so shape[0] is words,
    # not bytes.  A half-stage reads 128 B of bases = 32 words.
    extent = Int64(sfb_u8.shape[0])
    return min(max(off, Int64(0)), extent - Int64(32))


@cute.jit
def _ds41_bases_w2(
    sfb_u8: cute.Tensor,
    row_counts: cute.Tensor,
    b_down_u32: cute.Tensor,
    expert: Int32,
    rp_tile: Int32,
    rp_nt: Int32,
    rp_half: Int32,
    pkt: Int32,
    kt_count: Int32,
) -> Int64:
    """The w2 analogue of `_ds41_bases_w13`."""
    nt2 = b_down_u32.shape[0] // 256
    cb_w = nt2 * kt_count * 128
    rows_w = nt2 * 64
    e_cb_w = Int64(rp_tile) * Int64(128) - (
        Int64(rp_nt) * Int64(kt_count) + Int64(pkt)
    ) * Int64(128)
    # defensive clamps (predicated sites pass garbage indices; the
    # selected-away result may be wrong, never a fault)
    expert_c = max(Int32(0), min(Int32(expert), Int32(row_counts.shape[0]) - Int32(1)))
    off = (
        e_cb_w
        + Int64(expert_c) * Int64(rows_w)
        + Int64(cb_w)
        + Int64(min(max(rp_nt, Int32(0)), nt2 - Int32(1))) * Int64(64)
        + Int64(min(max(rp_half, Int32(0)), Int32(1))) * Int64(32)
    )
    # same units as _ds41_bases_w13: shape[0] is u32 words.  A full-stage
    # reads 256 B of bases = 64 words.
    extent = Int64(sfb_u8.shape[0])
    return min(max(off, Int64(0)), extent - Int64(64))


@cute.jit
def _ds41_decode_sfb_word(
    sfb_slot: Int32,
    row: Int32,
    slot_rows: cutlass.Constexpr,
) -> Uint32:
    """Reconstruct one stock SFB u32 word from a coded smem slot.

    Slot layout: [codes: rows x uint16][bases: rows x uint8] (inside the
    stock-sized slot). Integer only; the base-is-row-minimum invariant makes
    the unmasked per-lane add carry-free (test_ref_parity pins it).
    """
    w16 = ld_shared_u32(sfb_slot + ((row >> Int32(1)) << Int32(2)))
    code = (w16 >> Uint32((row & Int32(1)) * 16)) & Uint32(0xFFFF)
    brow = ld_shared_u32(
        sfb_slot + Int32(slot_rows * 2) + ((row >> Int32(2)) << Int32(2))
    )
    b = (brow >> Uint32((row & Int32(3)) * 8)) & Uint32(0xFF)
    d0 = code & Uint32(0xF)
    d1 = (code >> Uint32(4)) & Uint32(0xF)
    d2 = (code >> Uint32(8)) & Uint32(0xF)
    d3 = (code >> Uint32(12)) & Uint32(0xF)
    spread = d0 | (d1 << Uint32(8)) | (d2 << Uint32(16)) | (d3 << Uint32(24))
    return b * Uint32(0x01010101) + spread


@cute.jit
def _ds41_expert_shift_w13(
    sfb_u8: cute.Tensor,
    row_counts: cute.Tensor,
    b_w13_u32: cute.Tensor,
    expert: Int32,
    kt_count: Int32,
) -> Int64:
    """Words between the flat-tile index and the real per-expert position.

    The plane is `[codes][bases]` per expert, so a flat tile index is short by
    exactly one expert's bases region per expert. Same `nt13` derivation as
    `_ds41_bases_w13`, so the two sides cannot disagree.
    """
    nt13 = b_w13_u32.shape[0] // 256
    expert_c = max(Int32(0), min(Int32(expert), Int32(row_counts.shape[0]) - Int32(1)))
    return Int64(expert_c) * Int64(nt13 * 64)


@cute.jit
def _ds41_expert_shift_w2(
    sfb_u8: cute.Tensor,
    row_counts: cute.Tensor,
    b_down_u32: cute.Tensor,
    expert: Int32,
    kt_count: Int32,
) -> Int64:
    """The w2 analogue of `_ds41_expert_shift_w13`."""
    nt2 = b_down_u32.shape[0] // 256
    expert_c = max(Int32(0), min(Int32(expert), Int32(row_counts.shape[0]) - Int32(1)))
    return Int64(expert_c) * Int64(nt2 * 64)
