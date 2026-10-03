#!/usr/bin/env python3
"""Phase-1 compression spike for the calaman experimental transforms.

Measures whether the pipeline

    fp64 -> reinterpret as u64 -> xor_delta_encode -> byte_transpose -> {LZ4,Snappy}

actually improves the compression ratio of fp64 data, and whether each transform
earns its place (vs raw, vs either transform alone).

This is a NUMPY MIRROR of the two C++ modules, not the modules themselves -- it
validates the *idea*, not the shipped code. Faithfulness is pinned by
`_selfcheck()`, which reproduces the exact layout contract the C++ tests assert
(experimental/{xor_delta,byte_transpose} + test/experimental/...): block 0 holds
every value's most-significant byte, block 7 the least, computed by value (shift
+ mask) so the result is endianness-independent, exactly like byte_transpose.cppm.

Run (no changes to uv.lock -- deps are ephemeral):

    uv run --with numpy --with cramjam \
        experimental/compression_demo/compression_demo.py
"""

from __future__ import annotations

import pathlib

import cramjam
import numpy as np

_DATA = pathlib.Path(__file__).parent / "data"

# --- the two transforms, as numpy mirrors of the C++ modules -----------------


def xor_delta_encode(u: np.ndarray) -> np.ndarray:
    """Mirror of calaman.experimental.xor_delta_encode on a u64 array.

    out[0] = u[0]; out[i] = u[i] ^ u[i-1] (XOR against the ORIGINAL previous).
    """
    out = u.copy()
    out[1:] = u[1:] ^ u[:-1]
    return out


def xor_delta_decode(u: np.ndarray) -> np.ndarray:
    """Inverse: the decoded stream is the running XOR prefix of the deltas."""
    return np.bitwise_xor.accumulate(u)


# block b holds byte (7 - b) of every value: b=0 -> MSB, b=7 -> LSB.
_SHIFTS = (8 * (7 - np.arange(8, dtype=np.uint64))).astype(np.uint64)


def byte_transpose(u: np.ndarray) -> np.ndarray:
    """Mirror of calaman.experimental.byte_transpose -> (8*n) uint8 planes.

    planes[b*n + i] = byte (7 - b) of value i. Computed by shift+mask (by value,
    not by reinterpreting storage), so it matches the C++ on any host.
    """
    # (8, n): row b = byte (7-b) of each value; flatten C-order -> block0..block7.
    planes = ((u[None, :] >> _SHIFTS[:, None]) & np.uint64(0xFF)).astype(np.uint8)
    return planes.reshape(-1)


def byte_untranspose(planes: np.ndarray, n: int) -> np.ndarray:
    """Inverse of byte_transpose: (8*n) uint8 planes -> n u64 values."""
    rows = planes.reshape(8, n).astype(np.uint64)
    out = np.zeros(n, dtype=np.uint64)
    for b in range(8):
        out |= rows[b] << _SHIFTS[b]
    return out


# --- faithfulness check against the C++ layout contract ----------------------


def _make_values(n: int) -> np.ndarray:
    """The exact LCG stream from test/experimental/byte_transpose_tests.cpp."""
    v = np.empty(n, dtype=np.uint64)
    state = np.uint64(0x9E3779B97F4A7C15)
    mul = np.uint64(6364136223846793005)
    inc = np.uint64(1442695040888963407)
    step = np.uint64(0x0101010101010101)
    with np.errstate(over="ignore"):  # wrap-around is the point
        for i in range(n):
            state = state * mul + inc
            v[i] = state ^ (np.uint64(i) * step)
    return v


def _selfcheck() -> None:
    # Layout contract: block b == byte (7-b) of each value, on make_values(37).
    v = _make_values(37)
    planes = byte_transpose(v)
    for b in range(8):
        shift = np.uint64(8 * (7 - b))
        expected = ((v >> shift) & np.uint64(0xFF)).astype(np.uint8)
        assert np.array_equal(planes[b * 37 : (b + 1) * 37], expected), f"block {b}"

    # Boundary values + roundtrips, same cases the C++ suite pins.
    bound = np.array(
        [0, ~np.uint64(0), 1, 0x8000000000000000, 0x00000000000000FF,
         0xFF00000000000000, 0x0123456789ABCDEF, 0xFEDCBA9876543210],
        dtype=np.uint64,
    )
    for n in (0, 1, 2, 7, 64, 1000, 4097):
        w = _make_values(n)
        assert np.array_equal(byte_untranspose(byte_transpose(w), n), w), f"bt n={n}"
        assert np.array_equal(xor_delta_decode(xor_delta_encode(w)), w), f"xd n={n}"
    assert np.array_equal(byte_untranspose(byte_transpose(bound), len(bound)), bound)
    assert np.array_equal(xor_delta_decode(xor_delta_encode(bound)), bound)


# --- pipeline stages + measurement -------------------------------------------

# Each stage maps a u64 view to the bytes handed to the compressor.
STAGES: dict[str, callable] = {
    "raw": lambda u: u.tobytes(),
    "xor": lambda u: xor_delta_encode(u).tobytes(),
    "transpose": lambda u: byte_transpose(u).tobytes(),
    "xor+transpose": lambda u: byte_transpose(xor_delta_encode(u)).tobytes(),
}

CODECS: dict[str, callable] = {
    "LZ4": lambda b: len(bytes(cramjam.lz4.compress(b))),
    "Snappy": lambda b: len(bytes(cramjam.snappy.compress(b))),
    # zstd at default level -- an upper-bound reference (stronger ratio than the
    # two we'd actually ship), to see how much headroom the pipeline leaves.
    "zstd": lambda b: len(bytes(cramjam.zstd.compress(b))),
}


def as_u64(arr: np.ndarray) -> np.ndarray:
    """reinterpret_cast<uint64_t*> of a contiguous fp64 array (stage 0)."""
    return np.ascontiguousarray(arr, dtype=np.float64).view(np.uint64)


def pow2_scale(arr: np.ndarray) -> np.ndarray:
    """Divide by 2^k, the smallest power of two >= max|arr|, so all values land
    in (-1, 1]. LOSSLESS: division by a power of two only decrements the exponent
    field, leaving sign + all 52 mantissa bits untouched (barring underflow to
    subnormals). frexp gives that exponent directly: max|arr| in [2^(e-1), 2^e).
    """
    m = float(np.max(np.abs(arr)))
    if m == 0.0:
        return arr.copy()
    _, e = np.frexp(m)  # m == mantissa * 2^e, mantissa in [0.5, 1)
    return arr / np.ldexp(1.0, e)  # /2^e -> (-1, 1], exact


def ratios(arr: np.ndarray) -> dict[tuple[str, str], float]:
    """compression ratio (orig / compressed), per (stage, codec)."""
    u = as_u64(arr)
    orig = u.nbytes
    out: dict[tuple[str, str], float] = {}
    for sname, stage in STAGES.items():
        payload = stage(u)
        for cname, codec in CODECS.items():
            out[(sname, cname)] = orig / codec(payload)
    return out


# --- datasets ----------------------------------------------------------------


def datasets() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0xC0FFEE)
    n = 1 << 16  # 65536 values = 512 KiB of fp64
    d: dict[str, np.ndarray] = {}

    # Synthetic patterns -- isolate which data shapes the pipeline helps.
    d["near-constant"] = 1.0 + 1e-9 * rng.standard_normal(n)
    d["monotonic/sorted"] = np.sort(rng.standard_normal(n))
    d["linspace"] = np.linspace(-3.0, 3.0, n)
    d["noisy (gaussian)"] = rng.standard_normal(n)
    d["small ints as fp64"] = rng.integers(0, 256, n).astype(np.float64)

    # Real geqp3-path fp64: column-major matrices dumped by geqp3_dump.cpp, which
    # reuses the geqp3 test suite's exact generator (mt19937 + uniform(-3,3) +
    # make_rank_deficient) compiled with the project's clang/libc++ -- so these
    # are the matrices the suite feeds the device, only larger. See README to
    # (re)generate. Each file is already raw column-major f64; load it flat.
    for f in sorted(_DATA.glob("geqp3_*.f64")):
        label = f.stem.removeprefix("geqp3_")
        d[f"geqp3: {label}"] = np.fromfile(f, dtype=np.float64)
    return d


# --- report ------------------------------------------------------------------


# Pre-scalings applied to the fp64 array BEFORE reinterpret+stage. "none" is the
# identity; "pow2" is lossless (/2^k, exponent-only); "maxabs" is lossy (/|M|).
SCALINGS: dict[str, callable] = {
    "none": lambda a: a,
    "pow2": pow2_scale,
    "maxabs": lambda a: a / np.max(np.abs(a)) if np.max(np.abs(a)) else a.copy(),
}


def ratio(arr: np.ndarray, scaling: str, stage: str, codec: str) -> float:
    u = as_u64(SCALINGS[scaling](arr))
    return u.nbytes / CODECS[codec](STAGES[stage](u))


def mantexp_split_ratio(arr: np.ndarray, codec: str) -> tuple[float, bool, float]:
    """Per-element scale into [0.5,1): frexp each double into mantissa ∈ [0.5,1)
    and an int exponent, compress the two streams separately (mantissa via
    byte_transpose, exponents as int16), and total the two.

    This is the only way to put EVERY element in [0.5,1) -- it makes the mantissa
    stream's exponent field constant (top plane collapses) at the cost of a second
    stream. Lossless: ldexp(mant, exp) reconstructs arr bit-for-bit. Returns
    (combined ratio, lossless?, mantissa's share of the compressed bytes).
    """
    mant, exp = np.frexp(arr)  # arr == mant * 2**exp, |mant| ∈ [0.5,1) or 0
    exact = bool(np.array_equal(np.ldexp(mant, exp), arr))
    mant_bytes = CODECS[codec](byte_transpose(mant.view(np.uint64)).tobytes())
    # Double exponents fit int16 (range ~[-1021,1024]); store the stream raw.
    exp_bytes = CODECS[codec](exp.astype(np.int16).tobytes())
    total = mant_bytes + exp_bytes
    return arr.nbytes / total, exact, mant_bytes / total


def scaling_experiment(codec: str = "zstd") -> None:
    """Full scaling x stage sweep: does pre-scaling into [-1,1] help ANY stage --
    byte_transpose, xor_delta, or the combined pipeline -- and is /2^k lossless?

    For each dataset we show the four stages under the lossless /2^k scaling
    against the no-scaling baseline, plus the one lossy /|M|+transpose cell (to
    confirm the lossy variant stays a wild card) and the /2^k round-trip check.
    """
    print(f"\nScaling x stage sweep ({codec}).  ratio higher=better.")
    print("'pow2' = lossless /2^k before the stage; 'base T' = transpose, no scale.\n")
    lw = 30
    cols = ["base T", "pow2+raw", "pow2+xor", "pow2+T", "pow2+xorT", "maxabs+T"]
    header = f"{'dataset':<{lw}}" + "".join(f"{c:>11}" for c in cols)
    header += f"{'pow2 exact?':>13}"
    print(header)
    print("-" * len(header))
    for name, arr in datasets().items():
        vals = {
            "base T": ratio(arr, "none", "transpose", codec),
            "pow2+raw": ratio(arr, "pow2", "raw", codec),
            "pow2+xor": ratio(arr, "pow2", "xor", codec),
            "pow2+T": ratio(arr, "pow2", "transpose", codec),
            "pow2+xorT": ratio(arr, "pow2", "xor+transpose", codec),
            "maxabs+T": ratio(arr, "maxabs", "transpose", codec),
        }
        # /2^k losslessness: reconstruct by multiplying the same power back.
        mx = float(np.max(np.abs(arr)))
        _, e = np.frexp(mx) if mx else (0.0, 0)
        exact = True
        if mx:
            exact = bool(np.array_equal(pow2_scale(arr) * np.ldexp(1.0, e), arr))
        cells = "".join(f"{vals[c]:>11.2f}" for c in cols)
        print(f"{name:<{lw}}{cells}{('yes' if exact else 'NO'):>13}")


def split_experiment(codec: str = "zstd") -> None:
    """Per-element scale into [0.5,1) via frexp (mantissa + exponent streams),
    compared against plain byte_transpose. The only scaling that makes EVERY
    element's exponent field constant -- but it merely relocates the exponent bits
    that byte_transpose already groups, so it is lossless yet a slight net loss.
    """
    print(f"\nPer-element [0.5,1) split vs transpose ({codec}).  frexp -> compress")
    print("mantissa (transpose) + exponent (int16) streams separately.\n")
    lw = 38
    header = (f"{'dataset':<{lw}}{'base T':>10}{'mant/exp split':>16}"
              f"{'lossless?':>11}{'mant %':>9}")
    print(header)
    print("-" * len(header))
    for name, arr in datasets().items():
        base = ratio(arr, "none", "transpose", codec)
        r, exact, mant_share = mantexp_split_ratio(arr, codec)
        print(f"{name:<{lw}}{base:>10.2f}{r:>16.2f}"
              f"{('yes' if exact else 'NO'):>11}{mant_share * 100:>8.0f}%")


def main() -> None:
    _selfcheck()
    print("faithfulness check vs C++ layout contract: PASS\n")
    print("Compression ratio (original / compressed), higher is better.")
    print("Pipeline: fp64 -> u64 -> [stage] -> codec.\n")
    if not any(_DATA.glob("geqp3_*.f64")):
        print(f"(no geqp3 dumps in {_DATA}/ -- see README to generate them)\n")

    stages = list(STAGES)
    lw = 38  # label column width
    data = datasets()  # build once; reuse across codecs
    for codec in CODECS:
        header = f"{'dataset':<{lw}}" + "".join(f"{s:>15}" for s in stages)
        print(f"=== {codec} " + "=" * (len(header) - len(codec) - 5))
        print(header)
        print("-" * len(header))
        for name, arr in data.items():
            r = ratios(arr)
            best = max(r[(s, codec)] for s in stages)
            cells = ""
            for s in stages:
                val = r[(s, codec)]
                mark = "*" if val == best else " "
                cells += f"{val:>14.2f}{mark}"
            print(f"{name:<{lw}}{cells}")
        print()
    print("* = best stage for that (dataset, codec).  'xor+transpose' is the")
    print("  proposed pipeline; compare it against 'raw' and the two singles.")

    scaling_experiment("zstd")
    split_experiment("zstd")


if __name__ == "__main__":
    main()
