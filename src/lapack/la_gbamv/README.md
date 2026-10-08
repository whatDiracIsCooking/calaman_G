# `calaman.la_gbamv`

The GPU counterpart of LAPACK's `?la_gbamv` (`sla_gbamv`/`dla_gbamv`/
`cla_gbamv`/`zla_gbamv`): the absolute-value matrix-vector product of a
general **band** matrix,

```
y := alpha * |op(A)| * |x| + beta * |y|
```

followed by the symbolic-zero nudge. Its only reference caller is
`?la_gbrfsx_extended` (hence `?gbsvxx`), not in this tree yet. A stream in, a
`calaman::Status` out, one kernel, no workspace.

```cpp
import calaman.la_gbamv;   // also re-exports calaman::Trans and calaman::Status
// d_ab: (kl+ku+1)-by-n band of an m-by-n A, leading dim ldab >= kl+ku+1
calaman::la_gbamv<double>(stream, calaman::Trans::T, m, n, kl, ku,
                          1.0, d_ab, ldab, d_x, 1, 0.0, d_y, 1);
```

## Same conventions as `calaman.la_geamv`

Everything in [`la_geamv`'s README](../la_geamv/README.md) holds unchanged:
`Trans` for the INTEGER `TRANS`, CABS1, real `alpha`/`beta`/`y`, the
`SYMB_ZERO` bookkeeping and the `safe1 = (n + 1) * ?LAMCH('S')` nudge, signed
`INCX`/`INCY` walking from the far end when negative, validation before the
quick return, and bitwise agreement with the loop through no-FMA arithmetic.
The per-`y(i)` body is literally shared: `la_amv_row` in
`la_geamv/la_amv.cuh` (target `calaman::la_amv`). This module adds only the band
storage, `A(i, j) = AB(ku + i - j, j)` (0-based), and the `j` range of the band:

- `Trans::N`: `y(i)` sums columns `max(0, i - kl) .. min(n - 1, i + ku)`.
- `Trans::T`/`C`: `y(i)` sums rows `max(0, i - ku) .. min(m - 1, i + kl)`.

Both run in ascending `j`. Only the band is read. The corner triangles of `AB`
and any padding rows below `kl + ku + 1` are never touched, so they may hold
anything, NaN included.

## Validation

`wwrErrorInvalidValue` for `kl >= max(1, m)`, `ku >= max(1, n)`,
`ldab < kl + ku + 1`, `incx == 0` or `incy == 0` (the reference's `INFO` 4, 5,
6, 8, 11), checked **before** the quick return. A null pointer on a call that
would launch is also rejected.

One deviation: the reference tests `KL > M-1` and `KU > N-1`, which rejects
**every** `m == 0` or `n == 0` call and leaves its own quick return for those
unreachable. Here an empty dimension accepts `kl`/`ku == 0` and returns quickly.

## Where the reference loop is wrong

`DLA_GBAMV`/`CLA_GBAMV` (checked against v3.12.0 and `master`) mis-index two of
their four branches. This module computes the product the routine documents,
not those loops:

- **`TRANS != N`** walks `J = MAX(I-KL,1) .. MIN(I+KU,LENX)` and reads
  `AB(KE-I+J, I)` with `KE = KL+1`. That is `A(J + KL - KU, I)`, not `A(J, I)`.
  It is correct only when `kl == ku`. When `ku > kl` it can read the corner
  triangle.
- **`INCX != 1`** starts `JX = KX` for every row, while `J` starts at
  `MAX(I-KL,1)`. So for every row past the first `kl`, `x` is misaligned with
  the band, which `?gbmv` corrects with its `KX` bump.

The unit-stride `TRANS = N` branch is correct, and the kernel reproduces it
bit for bit. `?la_gbrfsx_extended` passes unit strides, so of the two bugs only
the first reaches it, on a transposed solve with `kl != ku`.

## Tested

`test/la_gbamv/` has the same oracle shape as `la_geamv`. It is a
same-precision host transcription, compared **bit for bit** on all four types
and all three `Trans`. No real oracle is available: Debian's `liblapack` does
not export `?la_gbamv_`, which is XBLAS-only. The oracle reads the **dense**
`A` that the band was packed from, and skips off-band entries by the band's
definition. So the kernel's band index arithmetic is checked against an
independent spelling.

The suite covers:

- `kl = 0`, `ku = 0`, `kl = ku = 0`, and `kl != ku` both ways.
- A band wider than `A` is tall, and the transposed shape.
- Padded `ldab`.
- An entirely zero band row (top, bottom, main diagonal).
- `x` with exact zeros, and symbolic-zero rows and columns.
- The all-underflow `safe1` case, both stride signs, the quick returns, and
  every rejection.

Every off-band entry of `AB` is NaN. Two mutations were caught: reproducing the
reference's transposed indexing, and reading one entry past the band.
