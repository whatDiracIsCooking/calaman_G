# `calaman.lag2`

The GPU counterpart of LAPACK's general-matrix precision conversions — `dlag2s`
(d→s), `slag2d` (s→d), `zlag2c` (z→c) and `clag2z` (c→z) — as one template over
the `(From, To)` pair, constrained by the exported concept `calaman::lag2_pair`:

```cpp
import calaman.lag2;   // also re-exports calaman::Status

// d_a: m x n double (lda); d_sa: m x n float (ldsa); d_info: one device int
calaman::lag2<double, float>(stream, m, n, d_a, lda, d_sa, ldsa, d_info);
```

## INFO

Narrowing reproduces the reference's `INFO`: 1 if any entry — any real or
imaginary part, for complex — lies outside ±`SLAMCH('O')` (`FLT_MAX` from
`<cfloat>` on the device), else 0, by the reference's own comparisons (so a NaN
converts and does not count). As in the reference, `SA` is unspecified when
`INFO = 1`. Widening cannot fail and always reports 0.

`INFO` is a **device int the caller owns and reads back**, the convention
`calaman.sterf` and `calaman.lahqr` use for a single device-side outcome. The
named status blocks of #211 (`WorkspaceLayout::fixed_struct`, `CgStatus`) are
for several device ints carved from a workspace; one int needs none. `Status`
carries only argument errors (`d_info` null, `lda`/`ldsa < max(1, m)`) and
runtime errors. The wrapper zeroes `d_info` on the stream before the kernel, so
every overflowing thread stores the same 1 and the race is benign.

## Shape

- `lag2.cuh` — `lag2_convert`, the per-element conversion and overflow check,
  linked as the INTERFACE target `calaman::lag2::header`. `calaman.lat2` (the
  triangular `?lat2?`) builds on it.
- `lag2.cu` — one thread per element over `lacpy.cu`'s 2-D grid.
- `lag2_bridge.h`, `interface.cppm`, `instantiations.cpp` — the usual module /
  `.cu` split.

## Tested

`test/lag2/`: all four pairs bitwise against `LAPACKE_?lag2?` over the whole
sentinel-filled `ldsa`×`n` buffer, with `lda ≠ ldsa`. Narrowing `INFO` for an
entry just above `FLT_MAX`, just below `-FLT_MAX` and `+inf`, and for complex
the real and imaginary parts separately; exactly ±`FLT_MAX` and an overflow
in `A`'s padding rows both give 0.
