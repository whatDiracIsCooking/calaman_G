# `calaman.larscl2`

The GPU counterpart of LAPACK's `?larscl2` (`slarscl2`/`dlarscl2`/`clarscl2`/
`zlarscl2`): divide a column-major matrix `X` in place by a diagonal matrix `D`,
held as a length-`m` vector — `X <- D^-1 * X`, i.e. `X(i,j) <- X(i,j) / d(i)`.
It is the inverse twin of [`calaman.lascl2`](../lascl2/README.md) and shares its
shape: a stream in, a `calaman::Status` out, one hand-launched per-element
kernel, no workspace, no `INFO`.

```cpp
import calaman.larscl2;   // also re-exports calaman::Status
// d_d: length-m device vector of REAL divisors
// d_x: device matrix, column-major, leading dimension ldx
calaman::larscl2<wwr::wwrDoubleComplex>(stream, m, n, d_d, d_x, ldx);
```

## Mapping from ?LARSCL2

`D` is `calaman::ComplexToRealType<T>` — real even when `X` is complex, as in the
reference (`SRC/clarscl2.f`: `REAL D(*)`, `COMPLEX X(LDX,*)`). The four
precisions become one template over `T`; the extents become `std::size_t`; `ldx`
is kept. Rows `m..ldx-1` are never touched. Unlike the reference (which has no
argument checks), `ldx < m` or a null pointer on a non-empty call returns
`wwrErrorInvalidValue`.

## The divide is literal

The reference is `X(I,J) / D(I)`, and this routine's would-be callers (the
extra-precise `?la_*` refinement) want a tight error bound, so each element is an
IEEE divide — never a reciprocal-then-multiply, which changes the last bit. A
complex element is divided per component by the real `d(i)`, which is what a
Fortran `COMPLEX / REAL` reduces to.

Nothing in Reference-LAPACK calls `?larscl2`; it lands as a leaf for API
completeness.

## Tested

`test/larscl2/` compares against a same-precision host transcription of the
Fortran loop, **bit for bit**. Debian's `liblapack` does not export `?larscl2`
(the extra-precise routines are built only with XBLAS) and LAPACKE has no
wrapper, so the host loop is the oracle. Division is correctly rounded on host and
device alike, so the comparison stays exact with non-trivial quotients and a `D`
spanning many binades — which is what lets the suite catch a reciprocal-multiply.
