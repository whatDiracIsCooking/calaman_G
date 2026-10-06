# `calaman.lat2`

The GPU counterpart of LAPACK's triangular precision conversions `dlat2s` and
`zlat2c` (LAPACK has no `slat2d`/`clat2z`), as one template over the
`(From, To)` pair, constrained by the exported concept `calaman::lat2_pair`:

```cpp
import calaman.lat2;   // also re-exports calaman::Uplo and Status

// d_a: n x n double (lda); d_sa: n x n float (ldsa); d_info: one device int
calaman::lat2<double, float>(stream, calaman::Uplo::L, n, d_a, lda, d_sa, ldsa, d_info);
```

Only the `uplo` triangle of `SA` is written; the other triangle and the
`ldsa - n` padding rows are left untouched.

## INFO

The same as `calaman.lag2`, which this module shares its per-element
conversion and overflow check with (`lag2_convert` in `lapack/lag2/lag2.cuh`,
linked as `calaman::lag2::header`): `INFO = 1` if any entry **of the triangle**
— any real or imaginary part — lies outside ±`SLAMCH('O')` (`FLT_MAX`), else 0.
An overflow only in the opposite triangle gives 0, as in the reference. `SA` is
unspecified on `INFO = 1`. One difference: the reference assigns `INFO` only on
overflow (its callers `dsposv`/`zcposv` pre-zero it), while `calaman::lat2`
always writes it.

`INFO` is a device int the caller owns and reads back (`calaman.lag2`'s README
has the reasoning); `Status` carries only argument and runtime errors.

## Shape

`lat2.cu` is `lag2.cu`'s kernel with an in-triangle test (the triangle is a
template argument, as `lacpy.cu`'s region is). The full `n`-by-`n` grid is
launched and the out-of-triangle threads return; `lat2_bridge.h`,
`interface.cppm` and `instantiations.cpp` are the usual module / `.cu` split.

## Tested

`test/lat2/`: bitwise against the Fortran `dlat2s_`/`zlat2c_` (LAPACKE has no
binding) for both `Uplo` values with `lda ≠ ldsa`, over the whole
sentinel-filled `ldsa`×`n` buffer. INFO is checked for overflow in the triangle,
on the diagonal and only in the opposite triangle, and for complex in the real
and imaginary parts separately.
