#!/usr/bin/env python3
"""Sort the nonzero CDERI stream, fp32-downcast, compress -- end to end.

Chains three ideas from the earlier spikes on the gly10 RI-fit CDERI tensor:

  1. strip the 57% exact zeros (#123) -- keep a presence bitmask to invert;
  2. SORT the dense nonzero stream so neighbouring magnitudes cluster, which
     finally gives byte_transpose/xor_delta smooth data to exploit -- but keep
     the sort PERMUTATION to restore the original order;
  3. fp32-downcast the values (lossy, ~6e-8 rel err) for a free 2x width cut.

The question is whether sorting's compression gain on the values survives the
cost of the two side-channels it forces us to store:

  * the presence bitmask (1 bit/elt; near-free under zstd, ~925x in #123);
  * the sort permutation -- an arbitrary order over ~320M elements needs
    ~n*log2(n) bits (~1.1-1.3 GB as uint32) and is essentially INCOMPRESSIBLE.

So four numbers matter, all end-to-end vs the original fp64 bytes (lossy only
in the fp32 cast; the fp64 rows are lossless):

  A  fp32 in-place  : compress(fp32 nz)        + mask              [no perm]
  B  fp32 sorted    : compress(fp32 sort(nz))  + perm + mask
  C  fp64 in-place  : compress(fp64 nz)        + mask   (#123, re-measured)
  D  fp64 sorted    : compress(fp64 sort(nz))  + perm + mask

D vs C answers whether sorting pays with no downcast at all; D is also reported
against the uncompressed nonzero fp64 stream, where a ratio < 1 means sorting
plus the permutation costs more than storing the values raw.

    uv run --with numpy --with cramjam --with h5py \
        experimental/compression_demo/cderi_fp32_sort_demo.py [path/to/cderi.h5]
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
from lossy_downcast import byte_transpose as ld_byte_transpose  # noqa: E402
from lossy_downcast import xor_delta_encode  # noqa: E402

DEFAULT_H5 = pathlib.Path("/home/lostica/projects/calaman_G/main/.temp") / (
    "gly10_ccpvdz_rifit_cderi.h5"
)
CODECS = cd.CODECS

# #123 lossless (fp64) references, for the end-to-end comparison.
REF_LOSSLESS = {  # codec -> (in-place, zero-strip+mask)
    "LZ4": (2.23, 2.48),
    "Snappy": (2.10, 2.45),
    "zstd": (2.29, 2.56),
}


def byte_transpose_safe(u: np.ndarray) -> np.ndarray:
    """Width-generic plane-wise byte_transpose (no (nbytes, n) intermediate)."""
    nbytes = u.dtype.itemsize
    n = u.shape[0]
    out = np.empty(nbytes * n, dtype=np.uint8)
    dt = u.dtype.type
    for b in range(nbytes):
        shift = dt(8 * (nbytes - 1 - b))
        out[b * n : (b + 1) * n] = ((u >> shift) & dt(0xFF)).astype(np.uint8)
    return out


# Pin the memory-safe transpose to lossy_downcast's tested one, at both widths.
for _probe in (np.arange(37, dtype=np.uint32), cd._make_values(37)):
    assert np.array_equal(byte_transpose_safe(_probe), ld_byte_transpose(_probe))
del _probe


def best_compressed(values_u: np.ndarray, codec: str) -> tuple[int, str]:
    """Smallest compressed size of a uintN value stream over the 4 stages."""
    stages = {
        "raw": values_u.view(np.uint8),
        "xor": xor_delta_encode(values_u).view(np.uint8),
        "transpose": byte_transpose_safe(values_u),
        "xor+transpose": byte_transpose_safe(xor_delta_encode(values_u)),
    }
    sizes = {name: CODECS[codec](p) for name, p in stages.items()}
    best = min(sizes, key=sizes.get)
    return sizes[best], best


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


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
    nz = arr[mask]
    nnz = nz.size
    mask_packed = np.packbits(mask)
    del arr, mask  # free 5.92 GB + the bool mask

    # fp32 cast (lossy) -- the only precision loss; measure it on the nonzero.
    nz32 = nz.astype(np.float32)
    rel = np.abs((nz32.astype(np.float64) - nz) / nz)
    maxerr, rmserr = float(rel.max()), float(np.sqrt(np.mean(rel**2)))
    uflow = int(np.count_nonzero(nz32 == 0))  # nonzero values that cast to 0
    del rel

    # Sort, and build the permutation needed to restore original order.
    t = time.monotonic()
    order = np.argsort(nz, kind="stable")  # nz[order] is ascending
    sys.stderr.write(f"  argsort in {time.monotonic() - t:.1f}s\n")
    assert nnz <= np.iinfo(np.uint32).max  # indices fit uint32
    perm = order.astype(np.uint32)  # the side-channel we must store
    nz32_sorted = nz32[order]
    # Sanity: scatter the sorted values back by the permutation recovers nz32.
    _recon = np.empty_like(nz32_sorted)
    _recon[order] = nz32_sorted
    assert np.array_equal(_recon, nz32)
    del _recon

    # Lossless fp64: the same sort, no downcast. Compressed sizes only, so the
    # 2.56 GB fp64 streams can be freed before the fp32 tables run.
    nz64_bytes = nz.nbytes
    nz64_sorted = nz[order]
    val64 = {}  # codec -> (unsorted_bytes, unsorted_stage, sorted_bytes, stage)
    for codec in CODECS:
        t = time.monotonic()
        u_bytes, u_stage = best_compressed(nz.view(np.uint64), codec)
        s_bytes, s_stage = best_compressed(nz64_sorted.view(np.uint64), codec)
        val64[codec] = (u_bytes, u_stage, s_bytes, s_stage)
        sys.stderr.write(f"  {codec} fp64 values in {time.monotonic() - t:.1f}s\n")
    del order, nz, nz64_sorted

    print(f"\nDataset: {h5.name}  ({system})")
    print(f"  {n} values, {nnz} nonzero ({nnz / n * 100:.2f}%); "
          f"original {_gb(orig_bytes)} fp64")
    print(f"  fp32 cast: max rel err {maxerr:.2e}, rms {rmserr:.2e}, "
          f"underflow {uflow}")
    print(f"  nonzero fp32 stream = {_gb(nz32.nbytes)}  "
          f"(half of {_gb(nnz * 8)} fp64)\n")

    # --- 1. does sorting help the VALUES? intrinsic ratio vs stream bytes ----
    print("1. Intrinsic compressibility of the fp32 value stream (ratio vs its")
    print("   own bytes): does sorting give the transforms something to bite?\n")
    hdr = f"{'codec':<10}{'unsorted (best)':>22}{'sorted (best)':>22}"
    print(hdr)
    print("-" * len(hdr))
    val = {}  # codec -> (unsorted_bytes, sorted_bytes, sorted_best_stage)
    for codec in CODECS:
        t = time.monotonic()
        u_bytes, u_stage = best_compressed(nz32.view(np.uint32), codec)
        s_bytes, s_stage = best_compressed(nz32_sorted.view(np.uint32), codec)
        val[codec] = (u_bytes, s_bytes, s_stage)
        sys.stderr.write(f"  {codec} values in {time.monotonic() - t:.1f}s\n")
        print(f"{codec:<10}"
              f"{f'{nz32.nbytes / u_bytes:.2f}x ({u_stage})':>22}"
              f"{f'{nz32.nbytes / s_bytes:.2f}x ({s_stage})':>22}")

    # --- 2. the permutation tax ---------------------------------------------
    print("\n2. The permutation tax (uint32 order array, "
          f"{_gb(perm.nbytes)} raw):\n")
    perm_u8 = perm.view(np.uint8)
    perm_comp = {}
    print(f"{'codec':<10}{'perm compressed':>20}{'ratio':>10}")
    print("-" * 40)
    for codec in CODECS:
        t = time.monotonic()
        pc = CODECS[codec](perm_u8)
        perm_comp[codec] = pc
        sys.stderr.write(f"  {codec} perm in {time.monotonic() - t:.1f}s\n")
        print(f"{codec:<10}{_gb(pc):>20}{perm.nbytes / pc:>9.2f}x")

    # --- 3. end-to-end vs original fp64 -------------------------------------
    mask_comp = {c: CODECS[c](mask_packed) for c in CODECS}
    print("\n3. END-TO-END vs original fp64 (lossy only in the fp32 cast).")
    print("   A = fp32 in-place (values+mask); B = fp32 sorted (values+perm+mask).\n")
    hdr = (f"{'codec':<8}"
           f"{'A values':>10}{'A total':>9}{'A ratio':>9}"
           f"{'B values':>10}{'B perm':>9}{'B total':>9}{'B ratio':>9}"
           f"{'#123 strip':>12}")
    print(hdr)
    print("-" * len(hdr))
    for codec in CODECS:
        u_bytes, s_bytes, _ = val[codec]
        mc = mask_comp[codec]
        a_total = u_bytes + mc
        b_total = s_bytes + perm_comp[codec] + mc
        print(f"{codec:<8}"
              f"{_gb(u_bytes):>10}{_gb(a_total):>9}{orig_bytes / a_total:>8.2f}x"
              f"{_gb(s_bytes):>10}{_gb(perm_comp[codec]):>9}{_gb(b_total):>9}"
              f"{orig_bytes / b_total:>8.2f}x"
              f"{REF_LOSSLESS[codec][1]:>11.2f}x")
    print("\n('#123 strip' = fp64 lossless zero-strip+mask, the no-fp32/no-sort "
          "baseline.)")

    # --- 4. lossless fp64: does sorting pay with no downcast? ---------------
    print("\n4. LOSSLESS fp64 (no downcast). C = fp64 in-place (values+mask);")
    print("   D = fp64 sorted (values+perm+mask). 'D vs nz' is D against the")
    print(f"   uncompressed nonzero fp64 stream ({_gb(nz64_bytes)}, no mask);")
    print("   < 1.00x means sorting + permutation outweighs storing the values raw.\n")
    hdr = (f"{'codec':<8}"
           f"{'C values':>26}{'C ratio':>9}"
           f"{'D values':>26}{'D perm':>9}{'D total':>9}{'D ratio':>9}"
           f"{'D vs nz':>9}")
    print(hdr)
    print("-" * len(hdr))
    for codec in CODECS:
        u_bytes, u_stage, s_bytes, s_stage = val64[codec]
        mc = mask_comp[codec]
        c_total = u_bytes + mc
        d_values = s_bytes + perm_comp[codec]
        d_total = d_values + mc
        print(f"{codec:<8}"
              f"{f'{_gb(u_bytes)} ({u_stage})':>26}{orig_bytes / c_total:>8.2f}x"
              f"{f'{_gb(s_bytes)} ({s_stage})':>26}{_gb(perm_comp[codec]):>9}"
              f"{_gb(d_total):>9}{orig_bytes / d_total:>8.2f}x"
              f"{nz64_bytes / d_values:>8.2f}x")


if __name__ == "__main__":
    main()
