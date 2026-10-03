#!/usr/bin/env python3
"""Run the Phase-1 compression spike on a real PySCF RI-fit CDERI matrix.

This drives the SAME transforms + codecs as `compression_demo.py`, but over the
3-center metric-contracted Cholesky integrals stored in
`gly10_ccpvdz_rifit_cderi.h5` (dataset `j3c`, shape (naux, npair), fp64). The
matrix is ~5.9 GB, so the demo's vectorised `byte_transpose` -- which builds an
(8, n) uint64 intermediate (~47 GB here) -- is replaced by a plane-wise version
that is asserted bit-identical to the demo's faithful one on a small slice.

    uv run --with numpy --with cramjam --with h5py \
        experimental/compression_demo/cderi_demo.py
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

H5 = pathlib.Path(__file__).parent / "gly10_ccpvdz_rifit_cderi.h5"


# --- memory-safe byte_transpose (plane-wise; no (8,n) uint64 intermediate) ----


def byte_transpose_safe(u: np.ndarray) -> np.ndarray:
    """Same result as cd.byte_transpose, but O(n) peak instead of O(8n) u64.

    Writes each of the 8 significance planes directly into a preallocated uint8
    buffer, so the only large transient is one (u >> shift) at a time.
    """
    n = u.shape[0]
    out = np.empty(8 * n, dtype=np.uint8)
    for b in range(8):
        out[b * n : (b + 1) * n] = ((u >> cd._SHIFTS[b]) & np.uint64(0xFF)).astype(
            np.uint8
        )
    return out


# Pin faithfulness to the demo's tested transpose on a non-trivial slice.
_probe = cd._make_values(4097)
assert np.array_equal(byte_transpose_safe(_probe), cd.byte_transpose(_probe))
del _probe


# Stages return a bytes-like payload (numpy uint8 view/array -> buffer protocol,
# so cramjam reads it with no extra tobytes() copy).
STAGES = {
    "raw": lambda u: u.view(np.uint8),
    "xor": lambda u: cd.xor_delta_encode(u).view(np.uint8),
    "transpose": lambda u: byte_transpose_safe(u),
    "xor+transpose": lambda u: byte_transpose_safe(cd.xor_delta_encode(u)),
}
CODECS = cd.CODECS  # name -> lambda(bytes-like) -> compressed byte length


def _fmt_bytes(nbytes: int) -> str:
    return f"{nbytes / 1e9:.2f} GB"


# --- report sections ----------------------------------------------------------


def core_table(u: np.ndarray) -> None:
    orig = u.nbytes
    stages = list(STAGES)
    lw = 16
    header = f"{'codec':<{lw}}" + "".join(f"{s:>15}" for s in stages)
    print("Compression ratio (original / compressed), higher is better.")
    print("Pipeline: fp64 -> u64 -> [stage] -> codec.\n")
    print(header)
    print("-" * len(header))
    # Build each payload once; reuse across the 3 codecs.
    payloads = {}
    for sname, stage in STAGES.items():
        t = time.monotonic()
        payloads[sname] = stage(u)
        sys.stderr.write(f"  built stage '{sname}' in {time.monotonic() - t:.1f}s\n")
    for cname, codec in CODECS.items():
        cells = ""
        best = 0.0
        vals = {}
        for sname in stages:
            t = time.monotonic()
            vals[sname] = orig / codec(payloads[sname])
            sys.stderr.write(
                f"  {cname:<7} {sname:<14} {time.monotonic() - t:.1f}s\n"
            )
            best = max(best, vals[sname])
        for sname in stages:
            mark = "*" if vals[sname] == best else " "
            cells += f"{vals[sname]:>14.2f}{mark}"
        print(f"{cname:<{lw}}{cells}")
    print("\n* = best stage for that codec.\n")


def scaling_sweep(arr: np.ndarray, codec: str = "zstd") -> None:
    print(f"Scaling x stage sweep ({codec}).  'base T' = transpose, no scale.")
    print("'pow2' = lossless /2^k before the stage; 'maxabs' = lossy /|M|.\n")
    orig = arr.nbytes

    def r(scaled: np.ndarray, stage: str) -> float:
        u = scaled.view(np.uint64)
        return orig / CODECS[codec](STAGES[stage](u))

    mx = float(np.max(np.abs(arr)))
    pow2 = cd.pow2_scale(arr)
    maxabs = arr / mx if mx else arr.copy()
    # /2^k losslessness: multiply the same power back.
    _, e = np.frexp(mx) if mx else (0.0, 0)
    exact = bool(np.array_equal(pow2 * np.ldexp(1.0, e), arr)) if mx else True

    cols = {
        "base T": lambda: r(arr, "transpose"),
        "pow2+raw": lambda: r(pow2, "raw"),
        "pow2+xor": lambda: r(pow2, "xor"),
        "pow2+T": lambda: r(pow2, "transpose"),
        "pow2+xorT": lambda: r(pow2, "xor+transpose"),
        "maxabs+T": lambda: r(maxabs, "transpose"),
    }
    header = "".join(f"{c:>11}" for c in cols) + f"{'pow2 exact?':>13}"
    print(header)
    print("-" * len(header))
    cells = ""
    for name, fn in cols.items():
        t = time.monotonic()
        cells += f"{fn():>11.2f}"
        sys.stderr.write(f"  {name:<10} {time.monotonic() - t:.1f}s\n")
    print(f"{cells}{('yes' if exact else 'NO'):>13}\n")


def mantexp_split(arr: np.ndarray, codec: str = "zstd") -> None:
    print(f"Per-element [0.5,1) frexp split vs transpose ({codec}).\n")
    orig = arr.nbytes
    base = orig / CODECS[codec](byte_transpose_safe(arr.view(np.uint64)))
    mant, exp = np.frexp(arr)
    exact = bool(np.array_equal(np.ldexp(mant, exp), arr))
    mant_bytes = CODECS[codec](byte_transpose_safe(mant.view(np.uint64)))
    exp_bytes = CODECS[codec](exp.astype(np.int16))
    total = mant_bytes + exp_bytes
    header = f"{'base T':>10}{'mant/exp split':>16}{'lossless?':>11}{'mant %':>9}"
    print(header)
    print("-" * len(header))
    print(
        f"{base:>10.2f}{orig / total:>16.2f}"
        f"{('yes' if exact else 'NO'):>11}{mant_bytes / total * 100:>8.0f}%\n"
    )


def main() -> None:
    cd._selfcheck()
    print("faithfulness check vs C++ layout contract: PASS")
    with h5py.File(H5, "r") as f:
        d = f["j3c"]
        attrs = dict(f.attrs)
        print(f"\nDataset: {H5.name}")
        print(f"  system   : {attrs.get('system')}")
        print(f"  basis    : {attrs.get('basis')} / aux {attrs.get('auxbasis')}")
        print(f"  layout   : {attrs.get('layout')}")
        print(
            f"  shape    : {d.shape} (naux x npair) fp64  = {_fmt_bytes(d.size * 8)}"
        )
        t = time.monotonic()
        arr = d[:].ravel()  # C-order: each aux vector P contiguous
        sys.stderr.write(f"  loaded h5 in {time.monotonic() - t:.1f}s\n")
    print(
        f"  stats    : min {arr.min():.3e}  max {arr.max():.3e}  "
        f"mean {arr.mean():.3e}  nnz-frac {np.count_nonzero(arr) / arr.size:.4f}\n"
    )

    u = arr.view(np.uint64)
    core_table(u)
    scaling_sweep(arr, "zstd")
    mantexp_split(arr, "zstd")


if __name__ == "__main__":
    main()
