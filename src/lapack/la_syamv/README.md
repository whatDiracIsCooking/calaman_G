# `calaman.la_syamv`

The GPU counterpart of LAPACK's `?la_syamv` (`sla_syamv`/`dla_syamv`/
`cla_syamv`/`zla_syamv`): the absolute-value matrix-vector product of a
symmetric matrix held in one triangle,

```
y := alpha * |A| * |x| + beta * |y|
```

followed by the symbolic-zero nudge. Its reference callers are
`?la_syrfsx_extended` and `?la_porfsx_extended` (hence `?sysvxx`/`?posvxx`), not
in this tree yet. A stream in, a `calaman::Status` out, one kernel, no
workspace. Its kernel also serves [`calaman.la_heamv`](../la_heamv/README.md).

```cpp
import calaman.la_syamv;   // also re-exports calaman::Uplo and calaman::Status
// d_a: n-by-n, the Uplo::L triangle stored, leading dim lda >= max(1, n)
calaman::la_syamv<double>(stream, calaman::Uplo::L, n, 1.0, d_a, lda,
                          d_x, 1, 0.0, d_y, 1);
```

## Same conventions as `calaman.la_geamv`

Everything in [`la_geamv`'s README](../la_geamv/README.md) holds unchanged:
CABS1, real `alpha`/`beta`/`y`, the `SYMB_ZERO` bookkeeping and the
`safe1 = (n + 1) * ?LAMCH('S')` nudge, signed `INCX`/`INCY` walking from the far
end when negative, validation before the quick return, and bitwise agreement
with the loop through no-FMA arithmetic. The per-`y(i)` body is literally
shared: `la_amv_row` in `la_geamv/la_amv.cuh` (target `calaman::la_amv`). This
module adds only the triangle. `y(i)` sums `j = 0 .. n-1` ascending, as the
reference's two `J` loops do, reading `A(i, j)` from the stored entry at
`(min(i, j), max(i, j))` for `Uplo::U` and `(max, min)` for `Uplo::L`. The
other triangle is never read, so it may hold anything, NaN included.

`UPLO` is an INTEGER in the reference, `ILAUPLO`'s `BLAS_UPPER = 121`,
`BLAS_LOWER = 122`. Here it is `calaman::Uplo`, so there is no `ilauplo` port,
and the reference's `INFO = 1` (a bad `UPLO`) and `INFO = 2` (`N < 0`) cannot
arise. `lda < max(1, n)`, `incx == 0` and `incy == 0` return
`wwrErrorInvalidValue` (`INFO` 5, 7, 10), before the quick return
(`n == 0`, or `alpha == 0 && beta == 1`).

## Where the reference loop is wrong

`DLA_SYAMV`/`CLA_SYAMV`/`CLA_HEAMV` (v3.12.0) test the wrong `x` element for a
symbolic zero when `INCX != 1`. Both strided branches multiply by `X(JX)` but
AND `X(J) .EQ. ZERO` into `SYMB_ZERO`, which is the buffer position of logical
element `J` only for a unit stride. `?la_geamv` gets this right (`X(JX)`). This
module tests the `x` it multiplies. So for a strided `x` a row whose non-zeros
meet only zeros of `x` is a symbolic zero here, as the routine documents.

The unit-stride branches are correct, and the kernel reproduces them bit for
bit. `?la_syrfsx_extended` and `?la_porfsx_extended` pass unit strides, so the
bug never reaches them.

## Tested

`test/la_syamv/` has the same oracle shape as `la_geamv`. It is a
same-precision host transcription, compared **bit for bit** on all four types
and both `Uplo`. No real oracle is available: Debian's `liblapack` does not
export `?la_syamv_` (checked with `nm -D`), which is XBLAS-only. The oracle
reads the **dense** full matrix that the triangle was packed from, so the
kernel's triangle indexing is checked against an independent spelling. The
complex case is complex **symmetric** (`A(j, i) == A(i, j)`, no conjugate), with
a complex diagonal.

The suite covers:

- `n = 1`, `2`, small and multi-block sizes, and padded `lda`.
- An entirely zero row, `x` with exact zeros, and a row whose non-zeros meet
  only the zeros of `x`.
- The all-underflow `safe1` case, both stride signs, the quick returns, and
  every rejection.

The unstored triangle, the `lda` padding and `x`'s stride gaps are NaN.
Reading the wrong triangle was caught by both this suite and `la_heamv`'s.
