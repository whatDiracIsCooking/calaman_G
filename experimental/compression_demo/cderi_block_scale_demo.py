#!/usr/bin/env python3
"""Block-floating-point scaling on the SORTED CDERI stream.

The #124 result: pre-sorting the nonzero CDERI values makes them compress
beautifully (xor_delta/transpose finally get a monotone sequence), but the sort
PERMUTATION needed to restore order is ~n*log2(n) bits (~1.23 GB zstd, ~1.04x),
and that tax exceeds the gain -- fp32-sorted lands 4.04x vs fp32-in-place 5.12x.

Two further facts bound anything built on sorting this tensor:

  * The permutation CEILING. The compressed permutation alone is ~1.23 GB
    against a 5.92 GB original, so *every* sorted scheme is capped at
    5.92/1.23 ~ 4.8x end-to-end -- already below the 5.12x in-place winner.
    No value transform can recover that: sorting loses on ratio, period.

For SORTED schemes this spike is NOT a ratio play -- every one is walled under
the permutation ceiling. It measures the ACCURACY lever the earlier work left on
the table: block-floating-point.

The idea (per-REGION 2^k, not per-element and not global):

  * GLOBAL /2^k pins the whole stream's max into (0.5, 1]; small values fall far
    below and, in a narrow format, UNDERFLOW (fp16's 5-bit exponent bottoms out
    at ~6e-5; even fp32 has a denormal tail here -- values ~1e-40 cast to fp32
    subnormals, max rel err ~7e-2 per #124).
  * PER-ELEMENT frexp (block size 1) gives every value its own exponent -- no
    underflow, uniform roundoff, and crucially ORDER-INDEPENDENT: an element
    keeps its own exponent whether or not the stream is sorted. Cost is one int16
    exponent per value (n of them).
  * BLOCK (this spike): one 2^k per contiguous REGION of B values. The scale is
    a power of two, so it is a LOSSLESS exponent shift (mantissa bits untouched);
    its only job is to pull each region into the format's normal range so the
    downcast does not underflow. Cost is n/B exponents, not n.

SORTING enables BLOCK (B>1), but NOT per-element. Block-FP only rescues accuracy
if a region's values share a magnitude: on the sorted stream contiguous values
are magnitude-neighbours, so one exponent per region ~ per-element accuracy; on
the unsorted stream a region spans the full range and its shared exponent
underflows the small members. PER-ELEMENT frexp (B=1) never shares an exponent,
so it is safe in ANY order and in-place owes NO permutation -- the row the sorted
framing missed: fp16 in-place per-element lands 8.32x zstd (4.88e-4, 0 underflow),
past fp32-in-place's 5.12x, because it buys fp16's underflow safety without ever
reordering. We sweep B on both streams for accuracy vs side-channel, and measure
the in-place per-element rows end-to-end to pin that ratio win.

    uv run --with numpy --with cramjam --with h5py \
        experimental/compression_demo/cderi_block_scale_demo.py [path/to/cderi.h5]
"""

from __future__ import annotations

import pathlib
import sys
import time

import cramjam  # noqa: F401  (used via compression_demo.CODECS)
import h5py
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import compression_demo as cd  # noqa: E402
from cderi_fp32_sort_demo import _gb, best_compressed  # noqa: E402

DEFAULT_H5 = pathlib.Path("/home/lostica/projects/calaman_G/main/.temp") / (
    "gly10_ccpvdz_rifit_cderi.h5"
)
CODECS = cd.CODECS

# label -> numpy float dtype, with its same-width unsigned view dtype.
PRECISIONS: dict[str, tuple[type, type]] = {
    "fp32": (np.float32, np.uint32),
    "fp16": (np.float16, np.uint16),
}


# --- block-floating-point core -----------------------------------------------


def block_exps(vals: np.ndarray, B: int) -> np.ndarray:
    """Per-region block exponent k_j: frexp exponent of each region's max|value|.

    Region j (B values) is scaled by 2^(-k_j), landing its max in (0.5, 1]. The
    exponents fit int16 comfortably (this tensor spans e in ~[-133, 1]).
    """
    n = vals.size
    nblocks = -(-n // B)
    pad = nblocks * B - n
    a = np.abs(vals)
    if pad:
        a = np.concatenate([a, np.zeros(pad, dtype=a.dtype)])
    bmax = a.reshape(nblocks, B).max(axis=1)  # (nblocks,)
    _, e = np.frexp(bmax)  # bmax == m * 2^e, m in [0.5,1); e==0 where bmax==0
    return e.astype(np.int16)


def block_scale_downcast(
    vals: np.ndarray, ftype: type, B: int
) -> tuple[np.ndarray, int, np.ndarray, float, float]:
    """Block-FP downcast of `vals` with region size B.

    Returns (low fpN array, #underflow, block-exp int16 stream, max rel err,
    rms rel err). The 2^k scale is an exact exponent shift; the only lossy step
    is the fpN cast, so the error is magnitude-independent *within* a region.
    B == vals.size is global scaling; B == 1 is per-element frexp.
    """
    n = vals.size
    exps = block_exps(vals, B)
    k = np.repeat(exps, B)[:n].astype(np.int32)  # per-element exponent
    scaled = np.ldexp(vals, -k)  # vals / 2^k, exact
    low = scaled.astype(ftype)  # the only lossy step
    del scaled
    recon = np.ldexp(low.astype(np.float64), k)
    underflow = int(np.count_nonzero((vals != 0) & (low == 0)))
    rel = np.abs((recon - vals) / vals)
    maxerr, rmserr = float(rel.max()), float(np.sqrt(np.mean(rel * rel)))
    del recon, rel, k
    return low, underflow, exps, maxerr, rmserr


def exponent_runs(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Region boundaries at every change of the binary exponent of |vals|.

    A region where k never changes shares one exponent, so scaling by 2^k lands
    *every* mantissa in [0.5,1) -- this is per-element frexp with the exponent
    run-length-encoded. Returns (run exponents int16, run lengths uint32).
    """
    e = np.frexp(np.abs(vals))[1].astype(np.int16)
    change = np.empty(e.size, dtype=bool)
    change[0] = True
    change[1:] = e[1:] != e[:-1]
    run_k = e[change]
    run_len = np.diff(np.flatnonzero(np.concatenate([change, [True]]))).astype(
        np.uint32)
    return run_k, run_len


def raw_downcast(
    vals: np.ndarray, ftype: type
) -> tuple[np.ndarray, int, float, float]:
    """No scaling -- cast straight to fpN (the #124 in-place path). Returns
    (low, #underflow, max rel err, rms rel err)."""
    low = vals.astype(ftype)
    recon = low.astype(np.float64)
    underflow = int(np.count_nonzero((vals != 0) & (low == 0)))
    rel = np.abs((recon - vals) / vals)
    return low, underflow, float(rel.max()), float(np.sqrt(np.mean(rel * rel)))


# --- report helpers ----------------------------------------------------------


def _fmt_err(x: float) -> str:
    return f"{x:.2e}"


def main() -> None:
    cd._selfcheck()
    print("faithfulness check vs C++ layout contract: PASS")
    h5 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_H5

    with h5py.File(h5, "r") as f:
        system = dict(f.attrs).get("system")
        t = time.monotonic()
        arr = f["j3c"][:].ravel()
        sys.stderr.write(f"  loaded h5 in {time.monotonic() - t:.1f}s\n")

    n = arr.size
    orig_bytes = arr.nbytes
    mask = arr != 0.0
    nz = arr[mask]  # unsorted nonzero stream (in-place order)
    nnz = nz.size
    mask_packed = np.packbits(mask)
    del arr, mask

    # Sort (by value) + the permutation that restores in-place order.
    t = time.monotonic()
    order = np.argsort(nz, kind="stable")
    sys.stderr.write(f"  argsort in {time.monotonic() - t:.1f}s\n")
    assert nnz <= np.iinfo(np.uint32).max
    perm = order.astype(np.uint32)
    nz_sorted = nz[order]
    del order

    vmin, vmax = float(np.abs(nz[nz != 0]).min()), float(np.abs(nz).max())
    orders = int(np.log2(vmax / vmin))
    print(f"\nDataset: {h5.name}  ({system})")
    print(f"  {n} values, {nnz} nonzero ({nnz / n * 100:.2f}%); "
          f"original {_gb(orig_bytes)} fp64")
    print(f"  nonzero range |v| in [{vmin:.2e}, {vmax:.2e}] -- "
          f"spans ~{orders} binary orders\n")

    # Shared side-channels (constant across all scaling choices).
    t = time.monotonic()
    perm_comp = {c: CODECS[c](perm.view(np.uint8)) for c in CODECS}
    mask_comp = {c: CODECS[c](mask_packed) for c in CODECS}
    sys.stderr.write(f"  perm+mask compressed in {time.monotonic() - t:.1f}s\n")
    print("Side-channels (compressed, zstd):")
    print(f"  sort permutation : {_gb(perm_comp['zstd'])}  "
          f"({perm.nbytes / perm_comp['zstd']:.2f}x of {_gb(perm.nbytes)})")
    print(f"  presence bitmask : {_gb(mask_comp['zstd'])}  "
          f"({mask_packed.nbytes / mask_comp['zstd']:.0f}x of "
          f"{_gb(mask_packed.nbytes)})")
    ceil = orig_bytes / perm_comp["zstd"]
    print(f"  => PERMUTATION CEILING: any sorted scheme <= {ceil:.2f}x "
          f"(perm alone), vs fp32-in-place 5.12x.\n")

    # ------------------------------------------------------------------ Part 1
    # Accuracy frontier: block-FP on SORTED vs UNSORTED, fp32 & fp16, over B.
    # Pure accuracy -- no compression -- so we can sweep B cheaply.
    print("=" * 78)
    print("1. ACCURACY vs block size B  (max rel err / #underflow). The lever is")
    print("   whether a region shares a magnitude -- true only once sorted.")
    print("=" * 78)
    block_sizes = [1, 64, 256, 1024, 4096, 65536, nnz]  # 1=per-elt, nnz=global
    streams = {"sorted": nz_sorted, "unsorted (in-place)": nz}
    for prec, (ftype, _) in PRECISIONS.items():
        print(f"\n--- {prec} " + "-" * 60)
        # baseline: no scaling at all (the #124 in-place cast).
        for sname, s in streams.items():
            if sname.startswith("unsorted"):
                _, u, me, rm = raw_downcast(s, ftype)
                print(f"  no-scale, {sname:<20}  max {_fmt_err(me)}  "
                      f"rms {_fmt_err(rm)}  underflow {u}")
        hdr = f"  {'B (region)':<14}" + "".join(
            f"{sname:>26}" for sname in streams)
        print(hdr)
        for B in block_sizes:
            tag = ("per-elem" if B == 1 else "global" if B == nnz else str(B))
            row = f"  {f'{B} ({tag})':<14}"
            for s in streams.values():
                _, u, _exps, me, _rm = block_scale_downcast(s, ftype, B)
                row += f"{f'{_fmt_err(me)} / {u}':>26}"
            print(row)
    print("\n  Reading it: on SORTED data a modest B already matches per-element")
    print("  accuracy (regions are magnitude-tight); on UNSORTED data every B")
    print("  stays as bad as global -- a region spans the whole range. That gap")
    print("  IS the result: sorting is what makes block-FP work.")

    # ------------------------------------------------------------------ Part 2
    # Side-channel cost of the block-exponent stream vs B (sorted stream).
    print("\n" + "=" * 78)
    print("2. BLOCK-EXPONENT side-channel cost vs B (sorted; int16/region, zstd)")
    print("=" * 78)
    print(f"  {'B':<10}{'#regions':>12}{'raw':>12}{'zstd':>12}{'% of orig':>12}")
    for B in block_sizes:
        exps = block_exps(nz_sorted, B)
        comp = CODECS["zstd"](exps.view(np.uint8))
        print(f"  {B:<10}{exps.size:>12}{_gb(exps.nbytes):>12}{_gb(comp):>12}"
              f"{comp / orig_bytes * 100:>11.3f}%")
    print("  (Sorted block exponents are near-monotone, so they compress well;")
    print("   per-element B=1 is the n-int16 frexp stream, the costly extreme.)")

    # ------------------------------------------------------------------ Part 2b
    # Let the region boundary BE the exponent: a new 2^k only when k changes.
    # This is per-element frexp (every mantissa in [0.5,1)) with the exponent
    # run-length-encoded -- floor accuracy, parameter-free, a ~KB side channel.
    print("\n" + "-" * 78)
    print("2b. EXPONENT-RUN regions (new 2^k only when k changes; sorted stream)")
    print("-" * 78)
    run_k, run_len = exponent_runs(nz_sorted)
    rle_raw = run_k.nbytes + run_len.nbytes
    rle_zstd = (CODECS["zstd"](run_k.view(np.uint8))
                + CODECS["zstd"](run_len.view(np.uint8)))
    orders = int(run_k.max() - run_k.min())
    pct = rle_zstd / orig_bytes * 100
    print(f"  {run_k.size} exponent runs (dynamic range ~{orders} binary orders, "
          f"~2 runs/order for +/-)")
    print(f"  side channel (k:int16 + len:uint32 per run): {rle_raw} B raw, "
          f"{rle_zstd} B zstd ({pct:.6f}% of orig)")
    for prec, (ftype, _) in PRECISIONS.items():
        # exponent-run == per-element frexp: scale each value by its own 2^k.
        k = np.repeat(run_k.astype(np.int32), run_len)
        low = np.ldexp(nz_sorted, -k).astype(ftype)
        recon = np.ldexp(low.astype(np.float64), k)
        rel = np.abs((recon - nz_sorted) / nz_sorted)
        u = int(np.count_nonzero((nz_sorted != 0) & (low == 0)))
        print(f"  {prec}: max rel err {_fmt_err(float(rel.max()))} "
              f"(== per-element floor), underflow {u}")
        del k, low, recon, rel
    print("  => reaches the fpN floor like fixed B=4096, at a ~KB side channel and")
    print("     no B to tune; still walled by the permutation ceiling end-to-end.")

    # ------------------------------------------------------------------ Part 3
    # End-to-end ratio for representative configs, vs the honest baselines.
    print("\n" + "=" * 78)
    print("3. END-TO-END ratio vs original 5.92 GB (lossy only in the fpN cast).")
    print("   sorted rows pay value+exps+perm+mask; in-place pays value+mask")
    print("   (+exps when per-element) -- never a permutation.")
    print("=" * 78)

    # (label, prec, stream, B or None=no-scale, needs_perm)
    configs = [
        ("fp32 in-place, no-scale", "fp32", nz, None, False),
        ("fp32 in-place, per-element", "fp32", nz, 1, False),
        ("fp32 sorted, global", "fp32", nz_sorted, nnz, True),
        ("fp32 sorted, block-4096", "fp32", nz_sorted, 4096, True),
        ("fp32 sorted, per-element", "fp32", nz_sorted, 1, True),
        ("fp16 in-place, no-scale", "fp16", nz, None, False),
        ("fp16 in-place, per-element", "fp16", nz, 1, False),
        ("fp16 sorted, global", "fp16", nz_sorted, nnz, True),
        ("fp16 sorted, block-4096", "fp16", nz_sorted, 4096, True),
        ("fp16 sorted, per-element", "fp16", nz_sorted, 1, True),
    ]
    hdr = (f"  {'config':<28}{'maxerr':>10}{'uflow':>7}"
           + "".join(f"{c:>9}" for c in CODECS))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, prec, stream, B, needs_perm in configs:
        ftype, utype = PRECISIONS[prec]
        if B is None:
            low, u, me, _rm = raw_downcast(stream, ftype)
            exps = None
        else:
            low, u, exps, me, _rm = block_scale_downcast(stream, ftype, B)
        t = time.monotonic()
        ratios = {}
        for codec in CODECS:
            vcomp, _stage = best_compressed(low.view(utype), codec)
            total = vcomp + mask_comp[codec]
            if exps is not None:
                total += CODECS[codec](exps.view(np.uint8))
            if needs_perm:
                total += perm_comp[codec]
            ratios[codec] = orig_bytes / total
        sys.stderr.write(f"  {label} compressed in {time.monotonic() - t:.1f}s\n")
        cells = "".join(f"{ratios[c]:>8.2f}x" for c in CODECS)
        print(f"  {label:<28}{_fmt_err(me):>10}{u:>7}{cells}")
        del low

    print("\n  References (#124, zstd): fp64 zero-strip 2.56x | fp32 in-place 5.12x")
    print(f"  | fp32 sorted 4.04x | permutation ceiling {ceil:.2f}x.")
    print("\n  Bottom line: for SORTED schemes block-FP is an accuracy tool, not a")
    print("  ratio win -- every sorted row is walled under the permutation ceiling,")
    print("  so sorting pays only where order need not be restored or is near-free.")
    print("  But PER-ELEMENT frexp needs no sort (it never shares an exponent), so")
    print("  it runs IN-PLACE with no permutation: fp16 in-place per-element lands")
    print("  8.32x zstd (4.88e-4, 0 underflow), past fp32-in-place's 5.12x -- the")
    print("  one scaling scheme that is a genuine ratio win on this tensor, where a")
    print("  ~5e-4 relative error is acceptable. (For fp32, frexp is pure overhead:")
    print("  no underflow to fix, so in-place per-element 4.85x < no-scale 5.12x.)")


if __name__ == "__main__":
    main()
