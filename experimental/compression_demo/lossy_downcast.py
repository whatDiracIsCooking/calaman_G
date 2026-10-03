#!/usr/bin/env python3
"""Lossy fp64 -> {fp32, fp16} downcast compression experiment.

Pipeline (lossy):

    fp64 -> scale -> cast to fpN (LOSSY) -> reinterpret uN
         -> {raw, xor_delta, byte_transpose, xor+transpose} -> {LZ4, Snappy, zstd}

Round-trip: decompress -> fpN -> upcast fp64 -> unscale, then compare to the
original fp64. Two scaling modes, both of which keep the cast itself the only
lossy step:

- "global"      : divide the whole array by 2^k (k = smallest power of two >=
                  max|arr|), so every value lands in (-1, 1]. One exponent shift,
                  exact. Fine for fp32 (8 exponent bits have headroom under 1),
                  but for fp16 (5 exponent bits, min normal ~6.1e-5) it drags the
                  smallest elements into subnormals/underflow -> errors up to 1.0.

- "per-element" : frexp each value into mantissa in [0.5,1) + its own integer
                  exponent (block-floating-point), downcast the mantissa, store
                  the exponents as a second (int16) stream. The mantissa is always
                  a normal fpN, so there is NO underflow and the error is a uniform
                  ~unit-roundoff regardless of magnitude -- the fix that makes fp16
                  safe on wide-dynamic-range data. Costs the exponent stream (cheap:
                  small, clustered ints) and fragments data sitting right on a
                  power-of-two boundary (two exponent classes).

Error floors: fp32 ~2^-24 (5.96e-8), fp16 ~2^-11 (4.88e-4). Ratio is measured
against the ORIGINAL fp64 byte count, so the width reduction (2x fp32, 4x fp16)
is part of the win reported.

Reuses the merged calaman.experimental compression spike for datasets, codecs and
the fp64 byte_transpose contract; adds width-generic transforms here.

    uv run --with numpy --with cramjam \
        experimental/compression_demo/lossy_downcast.py

Needs the geqp3 .f64 dumps (see README) for the geqp3 rows; runs on the synthetic
datasets regardless.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compression_demo import CODECS, datasets, ratio  # noqa: E402

# Target precisions: label -> (float dtype, unsigned-int dtype of same width).
PRECISIONS: dict[str, tuple[type, type]] = {
    "fp32": (np.float32, np.uint32),
    "fp16": (np.float16, np.uint16),
}

# --- width-generic transforms (byte_transpose / xor_delta at any width) --------


def xor_delta_encode(u: np.ndarray) -> np.ndarray:
    out = u.copy()
    out[1:] = u[1:] ^ u[:-1]
    return out


def byte_transpose(u: np.ndarray) -> np.ndarray:
    """uintN array -> (N_bytes * n) uint8 planes (block 0 = MSB ... last = LSB)."""
    nbytes = u.dtype.itemsize
    shifts = (8 * (nbytes - 1 - np.arange(nbytes, dtype=u.dtype))).astype(u.dtype)
    planes = ((u[None, :] >> shifts[:, None]) & u.dtype.type(0xFF)).astype(np.uint8)
    return planes.reshape(-1)


STAGES: dict[str, callable] = {
    "raw": lambda u: u.tobytes(),
    "xor": lambda u: xor_delta_encode(u).tobytes(),
    "transpose": lambda u: byte_transpose(u).tobytes(),
    "xor+transpose": lambda u: byte_transpose(xor_delta_encode(u)).tobytes(),
}


# --- scaling modes: each returns (low-precision array, reconstruction, #underflow,
#     optional int16 exponent side-stream to also compress) --------------------


def scale_factor(arr: np.ndarray) -> float:
    """2^k, smallest power of two >= max|arr|, so arr/scale lands in (-1, 1]."""
    m = float(np.max(np.abs(arr)))
    if m == 0.0:
        return 1.0
    _, e = np.frexp(m)
    return float(np.ldexp(1.0, e))


def encode_global(arr, ftype):
    s = scale_factor(arr)
    scaled = arr / s  # exact: exponent-only shift
    low = scaled.astype(ftype)  # lossy
    recon = low.astype(np.float64) * s
    underflow = int(np.count_nonzero((scaled != 0) & (low == 0)))
    return low, recon, underflow, None


def encode_perelem(arr, ftype):
    mant, exp = np.frexp(arr)  # arr == mant * 2**exp, |mant| in [0.5,1) or 0
    low = mant.astype(ftype)  # mantissa -> fpN, always normal (no underflow)
    recon = np.ldexp(low.astype(np.float64), exp)
    underflow = int(np.count_nonzero((mant != 0) & (low == 0)))
    return low, recon, underflow, exp.astype(np.int16)


ENCODERS = {"global": encode_global, "per-element": encode_perelem}


def measure(arr, ftype, utype, mode, stage, codec):
    """Return (ratio vs fp64 bytes, max rel err, rms rel err, #underflows)."""
    low, recon, underflow, exp = ENCODERS[mode](arr, ftype)
    comp = CODECS[codec](STAGES[stage](low.view(utype)))
    if exp is not None:
        comp += CODECS[codec](exp.tobytes())  # exponent side-stream
    nz = arr != 0
    rel = np.abs((recon[nz] - arr[nz]) / arr[nz]) if nz.any() else np.zeros(1)
    rms = float(np.sqrt(np.mean(rel**2)))
    return arr.nbytes / comp, float(rel.max()), rms, underflow


# --- report ------------------------------------------------------------------


def ratio_tables(data) -> None:
    """Ratio landscape under GLOBAL scaling: precision x codec x stage."""
    stages = list(STAGES)
    lw = 38
    print("Ratio = ORIGINAL fp64 bytes / compressed (width reduction included),")
    print("GLOBAL scaling.  'f64 T(ll)' = lossless fp64 transpose, reference.")
    for prec, (ftype, utype) in PRECISIONS.items():
        for codec in CODECS:
            cols = [f"{prec} {s}" for s in stages]
            header = (f"{'dataset':<{lw}}{'f64 T(ll)':>11}"
                      + "".join(f"{c:>16}" for c in cols))
            tag = f"{prec} / {codec}"
            print(f"\n=== {tag} " + "=" * (len(header) - len(tag) - 5))
            print(header)
            print("-" * len(header))
            for name, arr in data.items():
                cells = f"{ratio(arr, 'none', 'transpose', codec):>11.2f}"
                r = {s: measure(arr, ftype, utype, "global", s, codec)[0]
                     for s in stages}
                best = max(r.values())
                for s in stages:
                    cells += f"{r[s]:>15.2f}{'*' if r[s] == best else ' '}"
                print(f"{name:<{lw}}{cells}")


def mode_comparison(data, codec="zstd", stage="transpose") -> None:
    """GLOBAL vs PER-ELEMENT (block-floating-point): ratio, error and underflow."""
    lw = 38
    print(f"\n\nGlobal vs per-element (block-FP) scaling  ({codec}, {stage} stage).")
    print("Error is stage/codec-independent -- only the fpN cast loses precision.\n")
    for prec, (ftype, utype) in PRECISIONS.items():
        header = (f"{'dataset':<{lw}}"
                  f"{'glob ratio':>11}{'glob maxerr':>13}{'glob uflow':>11}"
                  f"{'pe ratio':>11}{'pe maxerr':>13}{'pe uflow':>10}")
        print(f"--- {prec} " + "-" * (len(header) - len(prec) - 5))
        print(header)
        for name, arr in data.items():
            gr, ge, _, gu = measure(arr, ftype, utype, "global", stage, codec)
            pr, pe, _, pu = measure(arr, ftype, utype, "per-element", stage, codec)
            print(f"{name:<{lw}}{gr:>11.2f}{ge:>13.2e}{gu:>11d}"
                  f"{pr:>11.2f}{pe:>13.2e}{pu:>10d}")
        print()
    print("Per-element removes fp16's magnitude-dependent error + underflow, giving")
    print("a uniform ~2^-11 (4.88e-4) at ~10% ratio cost; it fragments data sitting")
    print("on a power-of-two boundary (see near-constant).  fp32 needs neither mode.")


def main() -> None:
    data = datasets()
    ratio_tables(data)
    mode_comparison(data)


if __name__ == "__main__":
    main()
