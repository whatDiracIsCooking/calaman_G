#!/usr/bin/env python3
"""Compression spike on a CDERI tensor with the exact zeros stripped out.

The gly10 RI-fit CDERI matrix is ~57% exact zeros (RI-V metric contraction
drives most AO-pair blocks to zero). `cderi_demo.py` showed that in-place no
byte transform beats the codec, because the compressible structure IS those
zero-runs. This asks the complementary question: strip the zeros first, then how
compressible is what's left -- the dense stream of actual integral magnitudes?

Two ratios are reported, and the distinction matters:

  * INTRINSIC -- compressed size vs the *nonzero* byte count. "How much is left
    to squeeze once the zeros are gone?" (Expected: little -- dense fp64.)
  * END-TO-END -- compressed size vs the *original* fp64 byte count, and
    LOSSLESS, so it must also store a presence bitmask to put the zeros back:
        total = compress(nonzero values) + compress(packed mask)
    This is the honest number to compare against the in-place result from #120
    (zstd 2.28x, see REF_INPLACE below). Dropping zeros without the mask is not
    reconstructable, so the bare orig/nonzero "shrink" is a ceiling, not a codec.

    uv run --with numpy --with cramjam --with h5py \
        experimental/compression_demo/cderi_nonzero_demo.py [path/to/cderi.h5]
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

# The dataset is git-ignored and lives outside the worktree; default to where
# the main checkout keeps it, override with argv[1].
DEFAULT_H5 = pathlib.Path("/home/lostica/projects/calaman_G/main/.temp") / (
    "gly10_ccpvdz_rifit_cderi.h5"
)

# In-place full-matrix ratios measured in PR #120 (same data, no zero strip),
# the baseline the end-to-end numbers below must beat to be worth it.
REF_INPLACE = {"LZ4": 2.23, "Snappy": 2.10, "zstd": 2.29}  # best stage per codec


def byte_transpose_safe(u: np.ndarray) -> np.ndarray:
    """Plane-wise byte_transpose (no (8,n) uint64 intermediate); see cderi_demo."""
    n = u.shape[0]
    out = np.empty(8 * n, dtype=np.uint8)
    for b in range(8):
        out[b * n : (b + 1) * n] = ((u >> cd._SHIFTS[b]) & np.uint64(0xFF)).astype(
            np.uint8
        )
    return out


_probe = cd._make_values(4097)
assert np.array_equal(byte_transpose_safe(_probe), cd.byte_transpose(_probe))
del _probe

STAGES = {
    "raw": lambda u: u.view(np.uint8),
    "xor": lambda u: cd.xor_delta_encode(u).view(np.uint8),
    "transpose": lambda u: byte_transpose_safe(u),
    "xor+transpose": lambda u: byte_transpose_safe(cd.xor_delta_encode(u)),
}
CODECS = cd.CODECS


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    cd._selfcheck()
    print("faithfulness check vs C++ layout contract: PASS")
    h5 = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_H5
    with h5py.File(h5, "r") as f:
        attrs = dict(f.attrs)
        t = time.monotonic()
        arr = f["j3c"][:].ravel()
        sys.stderr.write(f"  loaded h5 in {time.monotonic() - t:.1f}s\n")

    n = arr.size
    mask = arr != 0.0
    nz = arr[mask]
    nnz = nz.size
    orig_bytes = arr.nbytes
    nz_bytes = nz.nbytes
    mask_packed = np.packbits(mask)  # ceil(n/8) bytes, one bit per element

    print(f"\nDataset: {h5.name}  ({attrs.get('system')})")
    print(f"  shape      : {arr.shape[0]} values (naux x npair flattened) fp64")
    print(
        f"  zeros      : {n - nnz} exact zero ({(n - nnz) / n * 100:.1f}%), "
        f"{nnz} nonzero ({nnz / n * 100:.2f}%)"
    )
    print(f"  original   : {_gb(orig_bytes)} fp64")
    print(f"  nonzero    : {_gb(nz_bytes)}  (bare zero-strip 'shrink' = "
          f"{orig_bytes / nz_bytes:.2f}x, but needs the mask to invert)")
    print(f"  mask packed: {mask_packed.nbytes / 1e6:.1f} MB  "
          f"(1 bit/elt, {mask.mean() * 100:.1f}% set)\n")

    u = nz.view(np.uint64)
    stages = list(STAGES)

    # --- INTRINSIC: ratio vs the nonzero byte count --------------------------
    print("INTRINSIC compressibility of the NONZERO stream (ratio vs nonzero")
    print("bytes). How much is left to squeeze once the zeros are gone?\n")
    lw = 16
    header = f"{'codec':<{lw}}" + "".join(f"{s:>15}" for s in stages)
    print(header)
    print("-" * len(header))
    payloads = {}
    for sname, stage in STAGES.items():
        t = time.monotonic()
        payloads[sname] = stage(u)
        sys.stderr.write(f"  built '{sname}' in {time.monotonic() - t:.1f}s\n")
    comp = {}  # (codec, stage) -> compressed bytes, reused for end-to-end
    for cname, codec in CODECS.items():
        vals, best = {}, 0.0
        for sname in stages:
            t = time.monotonic()
            cbytes = codec(payloads[sname])
            comp[(cname, sname)] = cbytes
            vals[sname] = nz_bytes / cbytes
            best = max(best, vals[sname])
            sys.stderr.write(f"  {cname:<7}{sname:<14}{time.monotonic() - t:.1f}s\n")
        cells = "".join(
            f"{vals[s]:>14.2f}{'*' if vals[s] == best else ' '}" for s in stages
        )
        print(f"{cname:<{lw}}{cells}")
    print("\n* = best stage for that codec.\n")

    # --- END-TO-END: lossless, vs the ORIGINAL size (values + mask) ----------
    print("END-TO-END lossless ratio vs ORIGINAL fp64, storing the presence")
    print("bitmask so the zeros are reconstructable:")
    print("  ratio = original / [ compress(best nonzero stage) + compress(mask) ]\n")
    mask_comp = {c: CODECS[c](mask_packed) for c in CODECS}
    hdr = (f"{'codec':<10}{'best stage':>16}{'values':>12}{'mask':>10}"
           f"{'total':>12}{'end2end':>10}{'in-place #120':>15}")
    print(hdr)
    print("-" * len(hdr))
    for cname in CODECS:
        best_stage = min(stages, key=lambda s: comp[(cname, s)])
        vbytes = comp[(cname, best_stage)]
        mbytes = mask_comp[cname]
        total = vbytes + mbytes
        print(
            f"{cname:<10}{best_stage:>16}{_gb(vbytes):>12}"
            f"{mbytes / 1e6:>8.1f}MB{_gb(total):>12}"
            f"{orig_bytes / total:>10.2f}{REF_INPLACE[cname]:>15.2f}"
        )
    print()


if __name__ == "__main__":
    main()
