# `calaman.la_geamv`

The GPU counterpart of LAPACK's `?la_geamv` (`sla_geamv`/`dla_geamv`/
`cla_geamv`/`zla_geamv`): the absolute-value matrix-vector product of a general
matrix,

```
y := alpha * |op(A)| * |x| + beta * |y|
```

followed by the **symbolic-zero nudge** (below). It computes the componentwise
error bound used by `?la_gerfsx_extended`, hence `?gesvxx`. Neither is in this
tree yet, so this is a leaf of a chain that has not been started. Same shape as
`calaman.lagtm`: a stream in, a `calaman::Status` out, one hand-launched
kernel, no workspace, no allocation.

```cpp
import calaman.la_geamv;   // also re-exports calaman::Trans and calaman::Status
// d_a: m-by-n column-major, leading dim lda; d_x complex; d_y REAL
calaman::la_geamv<wwr::wwrDoubleComplex>(stream, calaman::Trans::N, m, n,
                                         1.0, d_a, lda, d_x, 1, 0.0, d_y, 1);
```

## The conventions the other `?la_*amv` modules copy

`?la_gbamv` (#291) and `?la_heamv`/`?la_syamv` (#292) run the same loop over a
different storage scheme. Copy these three decisions. The loop body itself is
shared: `la_amv_row` in `la_amv.cuh` (target `calaman::la_amv`) takes a `j`
range and an accessor for `op(A)(i, j)`, so a sibling's kernel supplies only
its storage.

### 1. `TRANS` is `calaman::Trans`

The reference `TRANS` is an INTEGER, `ILATRANS`'s BLAS codes:
`BLAS_NO_TRANS = 111`, `BLAS_TRANS = 112`, `BLAS_CONJ_TRANS = 113`. Here it is
`calaman::Trans` (`N`, `T`, `C`), so there is no `ilatrans` port. The kernel
asks only `trans == Trans::N`. `T` and `C` do the same thing for every type,
because `|conj(a)| == |a|`, and the reference also treats every non-`N` code
alike. A test that calls a Fortran `?la_*amv_` symbol directly must pass the
integer, not a character. No such test exists yet, because the image's
`liblapack` does not export these symbols (see *Tested*).

### 2. Symbolic zeros: carry the flag, nudge with `copysign`

One thread per `y(i)` replays the reference's bookkeeping exactly:

- `symb_zero` starts **true** when `beta == 0` (and `y(i)` is set to `+0`
  without being read) or when `y(i) == 0` (left as it is, so `-0` stays `-0`).
  Otherwise it starts **false** and `y(i) := beta * |y(i)|`.
- Each term ANDs in `(|x(j)| == 0 || |a(i,j)| == 0)` before adding
  `(alpha * |x(j)|) * |a(i,j)|`. The comparison is on the CABS1 value, which is
  zero exactly when both components are.
- At the end, `if (!symb_zero) y(i) += copysign(safe1, y(i))`, with
  `safe1 = (n + 1) * ?LAMCH('S')`. `n` is **A's column count whatever `trans`
  is**, as in the reference. `?LAMCH('S')` is `FLT_MIN`/`DBL_MIN`. The host
  computes `safe1` once (`std::numeric_limits<R>::min()` is fine there) and
  passes it to the kernel, so device code never needs a machine constant.

`|.|` on a complex element is CABS1, `|re| + |im|`, never the modulus. `alpha`,
`beta` and `y` are real (`ComplexToRealType<T>`) for all four types. The quick
return (`m == 0`, `n == 0`, or `alpha == 0 && beta == 1`) leaves `y` exactly as
given, without even taking its absolute value.

The result is **bitwise** the reference loop's, not just within a tolerance.
An exact `0` and an exact `safe1` differ by far less than any tolerance, so a
tolerance check cannot see whether the nudge is there. Bitwise agreement needs
each multiply and add rounded on its own, in the reference's order:

- CUDA uses `__fmul_rn`/`__dadd_rn` and friends.
- HIP spells the plain operators under `#pragma clang fp contract(off)`. HIP's
  own `_rn` functions lower to plain operators that its default
  `fp-contract=fast` then fuses. This was seen on gfx1200: about 1 ULP off
  until the change.

### 3. `INCX`/`INCY` are kept, signed

The strides are kept as `int`, as in `calaman.lassq` and `calaman.rscl`. A
negative stride walks from the far end, as the reference's `KX`/`KY` do. The
host wrapper moves the pointer to the logical first element, so the kernel
only ever sees `x + j*incx`. `incx == 0` or `incy == 0` returns
`wwrErrorInvalidValue` (the reference's `INFO = 8`/`11`), and so does
`lda < max(1, m)` (`INFO = 6`). These checks run **before** the quick return,
in the reference's order. A null pointer on a call that would launch is also
`wwrErrorInvalidValue`. The callers (`?la_*rfsx_extended`) pass unit strides,
but keeping the strides costs one multiply and keeps the signature 1:1 with the
reference.

## Kernel

There is one thread per `y(i)`, in blocks of `4 * WWR_WARP_SIZE`. With
`Trans::N`, thread `i` walks row `i`, which is coalesced across the block. With
`T`/`C` it walks column `i`, which is strided across threads. Coalescing that
path needs a block-cooperative transpose that keeps each thread's sequential
sum order, which the bitwise contract depends on. That is left until a caller
needs the speed.

## Tested

`test/la_geamv/` compares against a same-precision host transcription of the
Fortran loop, **bit for bit**, on all four types and all three `Trans`. No
real oracle is available: LAPACKE has no wrapper, `lapack.h` has no
`LAPACK_?la_geamv`, and Debian's `liblapack` does not export `?la_geamv_`,
because the extra-precise routines are built only with XBLAS. The host port
writes each operation as its own statement so that it cannot contract either.

The suite covers:

- A with an all-zero row and column, and a row (and column) whose non-zeros
  meet only the exact zeros of `x`. These are symbolic zeros that are not zero
  rows.
- `y` with `+0`/`-0` entries.
- A case where every product underflows, so the result must be exactly
  `safe1`, not `0`.
- `alpha`/`beta` of 0, 1 and general values; both stride signs; padded `lda`.
- The quick returns and the invalid-argument rejections.

NaN fills `x`'s stride gaps and `A`'s padding rows, so a read outside the
operands shows up in the result.

Mutating the kernel was caught every time: dropping the nudge, ignoring `x`
in the `symb_zero` test, and using the modulus instead of CABS1 each failed the
suite.
