# `calaman.la_heamv`

The GPU counterpart of LAPACK's `?la_heamv` (`cla_heamv`/`zla_heamv`; there is
no real form): the absolute-value matrix-vector product of a Hermitian matrix
held in one triangle,

```
y := alpha * |A| * |x| + beta * |y|
```

followed by the symbolic-zero nudge. Its reference callers are
`?la_herfsx_extended` and `?la_porfsx_extended` (hence `?hesvxx`/`?posvxx`),
not in this tree yet.

```cpp
import calaman.la_heamv;   // also re-exports calaman::Uplo and calaman::Status
calaman::la_heamv<wwr::wwrDoubleComplex>(stream, calaman::Uplo::U, n, 1.0,
                                         d_a, lda, d_x, 1, 0.0, d_y, 1);
```

## One kernel, no Hermitian flag

This module owns no device code. It runs
[`calaman.la_syamv`](../la_syamv/README.md)'s kernel and links
`calaman.la_syamv.device`, which is the `lanhe`/`lansy` arrangement. Unlike
`lanhe`, it passes no Hermitian flag, because there is nothing for one to
switch. `|.|` is CABS1, and `CABS1(conj(a)) == CABS1(a)` exactly, so the
Hermitian and symmetric readings of a stored triangle give the same `|A|`. The
reference agrees: `CLA_HEAMV` and `CLA_SYAMV` have the same executable body up
to whitespace and the `XERBLA` name. `CLA_HEAMV` calls `XERBLA('CHEMV ', ...)`,
which is wrong; a `Status` replaces it here.

So every convention, and the reference's strided-`x` bug that this module does
**not** copy, is `la_syamv`'s.

### The diagonal's imaginary part is read

A Hermitian matrix has a real diagonal, and `?hemv`/`?lanhe` read only its
real part. `CLA_HEAMV` reads every entry through CABS1, the diagonal included,
so a stored diagonal with a non-zero imaginary part contributes
`|Re| + |Im|`. This module does the same, deliberately: it is what the
reference computes, and it is what keeps `la_heamv` and `la_syamv` one kernel.

## Tested

`test/la_heamv/` uses `la_syamv`'s oracle on a dense **Hermitian** matrix
(`A(j, i) == conj(A(i, j))`) whose diagonal has non-zero imaginary parts,
compared **bit for bit**, for c/z and both `Uplo`. It covers the same shapes and
edge cases as `la_syamv`'s suite. Two further checks:

- On every case, `la_syamv` runs on the same stored triangle and must agree
  with `la_heamv` bit for bit, so the shared kernel is asserted to be shared.
- `A = i * I` must give `y = alpha * |x|`, not a symbolic zero, which pins the
  diagonal's imaginary part as read.
