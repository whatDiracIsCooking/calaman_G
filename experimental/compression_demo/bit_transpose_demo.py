#!/usr/bin/env python3
"""Bit-plane transpose vs byte-plane transpose, for fp64 compression.

byte_transpose groups the 8 bytes of each value by significance (8 planes).
bit_transpose goes one level finer: 64 planes, plane b holding bit b of every
value, bit-packed. Same total size when n is a multiple of 8, but an individual
low-entropy bit -- a sign bit that is constant across the array, or a high
exponent bit shared with varying neighbours in its byte -- becomes a contiguous
run of identical bits that the codec collapses, which byte_transpose cannot do
because that bit is mixed into a byte whose other bits vary.

This is a numpy mirror (validates the idea, not shipped C++); bit_untranspose is
its exact inverse, checked by _selfcheck(). Bit layout matches byte_transpose's
contract: plane 0 = MSB (bit 63), plane 63 = LSB (bit 0).

    uv run --with numpy --with cramjam \
        experimental/compression_demo/bit_transpose_demo.py

Needs the geqp3 .f64 dumps (see README) for the geqp3 rows; runs on the synthetic
datasets regardless.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compression_demo import (  # noqa: E402
    CODECS,
    as_u64,
    byte_transpose,
    datasets,
    xor_delta_encode,
)

_BIT_SHIFTS = (63 - np.arange(64, dtype=np.uint64)).astype(np.uint64)


def bit_transpose(u: np.ndarray) -> np.ndarray:
    """uint64 array (n,) -> 64 bit-planes, packed. Plane b = bit (63-b) of every
    value (plane 0 = MSB), each plane's n bits packed MSB-first into ceil(n/8)
    bytes. Total 64*ceil(n/8) bytes (= 8*n when n % 8 == 0)."""
    bits = ((u[None, :] >> _BIT_SHIFTS[:, None]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(bits, axis=1).reshape(-1)  # (64, ceil(n/8)) -> flat


def bit_untranspose(planes: np.ndarray, n: int) -> np.ndarray:
    """Inverse of bit_transpose."""
    per_plane = (n + 7) // 8
    bits = np.unpackbits(planes.reshape(64, per_plane), axis=1)[:, :n]  # (64, n)
    out = np.zeros(n, dtype=np.uint64)
    for b in range(64):
        out |= bits[b].astype(np.uint64) << _BIT_SHIFTS[b]
    return out


def _selfcheck() -> None:
    rng = np.random.default_rng(0xBE11)
    for n in (0, 1, 2, 7, 8, 9, 63, 64, 1000, 4097):
        u = rng.integers(0, 1 << 63, size=n, dtype=np.uint64)
        u |= rng.integers(0, 1 << 63, size=n, dtype=np.uint64) << np.uint64(1)
        assert np.array_equal(bit_untranspose(bit_transpose(u), n), u), f"n={n}"
    # boundary values
    bound = np.array(
        [0, ~np.uint64(0), 1, 0x8000000000000000, 0x0123456789ABCDEF],
        dtype=np.uint64,
    )
    assert np.array_equal(bit_untranspose(bit_transpose(bound), len(bound)), bound)


# Stages to compare (all operate on the u64 view of the fp64 data).
STAGES: dict[str, callable] = {
    "raw": lambda u: u.tobytes(),
    "byte T": lambda u: byte_transpose(u).tobytes(),
    "bit T": lambda u: bit_transpose(u).tobytes(),
    "xor+byte T": lambda u: byte_transpose(xor_delta_encode(u)).tobytes(),
    "xor+bit T": lambda u: bit_transpose(xor_delta_encode(u)).tobytes(),
}


def main() -> None:
    _selfcheck()
    print("bit_transpose roundtrip self-check: PASS\n")
    print("Compression ratio (orig fp64 / compressed), higher is better.")
    print("byte T = 8 byte-planes; bit T = 64 bit-planes.\n")

    data = datasets()
    stages = list(STAGES)
    lw = 38
    for codec in CODECS:
        header = f"{'dataset':<{lw}}" + "".join(f"{s:>14}" for s in stages)
        print(f"=== {codec} " + "=" * (len(header) - len(codec) - 5))
        print(header)
        print("-" * len(header))
        for name, arr in data.items():
            u = as_u64(arr)
            orig = u.nbytes
            r = {s: orig / CODECS[codec](STAGES[s](u)) for s in stages}
            best = max(r.values())
            cells = "".join(f"{r[s]:>13.2f}{'*' if r[s] == best else ' '}"
                            for s in stages)
            print(f"{name:<{lw}}{cells}")
        print()
    print("* = best stage for that (dataset, codec).")


if __name__ == "__main__":
    main()
