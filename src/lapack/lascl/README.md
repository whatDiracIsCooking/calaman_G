# `calaman.lascl`

The GPU counterpart of LAPACK's `?lascl`: multiply a column-major matrix `A` in
place by the ratio `cto/cfrom`, computed WITHOUT over/underflow —
`A(i,j) <- (cto/cfrom) * A(i,j)`. One entry point, templated over `float`,
`double`, `wwrFloatComplex` and `wwrDoubleComplex`; `cfrom`/`cto` are always
real (`ComplexToRealType<T>`):

```cpp
import calaman.lascl;    // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_a: device matrix, column-major, leading dimension lda
calaman::lascl(stream, cfrom, cto, m, n, d_a, lda);
```

The ratio `cto/cfrom` is never formed directly — it could over- or underflow
when the two are far apart. The scaling is enqueued on `stream` and returns
without synchronizing, like a BLAS call; the caller synchronizes when it needs
`A`. `A` is a device pointer the caller owns; nothing is allocated here. A
stream and not a device handle, because this routine allocates nothing — the
same choice as `calaman.lascl2` and `calaman.laset`. See `test/shared/README.md`.

## The guarded multiplier loop

The over/underflow safety is the whole point of `?lascl`, and it lives on the
**host**: `cto/cfrom` is decomposed into a short chain of factors, each
guaranteed in range, and the per-element multiply kernel is launched once per
factor. The loop is transcribed verbatim from netlib `?lascl`, driven by
`smlnum = DLAMCH('S')` (the smallest normal, `std::numeric_limits<T>::min()` on
the host) and `bignum = 1/smlnum`. Because the decomposition is host scalar
arithmetic, the kernel is a plain multiply and needs no device machine constants
— side-stepping the `nvcc`-rejects-`std::numeric_limits`-in-`__device__` trap.

## Mapping from `?lascl`

Kept: the name, `cfrom`, `cto`, and the leading dimension `lda` — that last a
fact of column-major storage, not a Fortran accommodation. Changed, per
`docs/architecture.md` §4: the `s/d/c/z` variants become one template over `T`,
and the `INTEGER` extents become `std::size_t`. As in `?lascl`, `cfrom`/`cto`
stay real for a complex `T`, and the kernel scales the real and imaginary parts
by the same factor (`elem_ops<T>::scale`, `common/elem_ops.cuh`).

**Scope: `TYPE='G'` (full matrix) only.** `?lascl` also accepts the lower/upper
triangular, upper-Hessenberg and three banded `TYPE` codes (with `KL`/`KU`);
those are not implemented, so the `KL`/`KU` arguments do not appear.

`INFO`: `?lascl`'s argument checks that survive the `TYPE='G'` scope become one
`InvalidValue` runtime `Status` — a zero or `NaN` `cfrom`, or a `NaN` `cto`
(`?lascl`'s `INFO=-4/-5`). There is no separate `INFO` out-parameter; the error
rides the `Status` return.

A new element type goes into the three lists that must stay in step —
`interface.cppm`'s `extern template`, `instantiations.cpp`, and `lascl.cu`.

## Shape

`interface.cppm` is the host wrapper (and the home of the guarded loop);
`lascl.cu` is the device half — the per-factor per-element multiply;
`lascl_bridge.h` carries the launcher declaration across the host/device
boundary (a global module fragment cannot `import`). `instantiations.cpp`
explicitly instantiates the wrapper for each type. The host module links
`calaman.common` for `ComplexToRealType`; the device library links it for
`idivup` (`common/align_up.h`) and `elem_ops` (`common/elem_ops.cuh`).

The kernel is one hand-launched 2-D grid of 1-D blocks, each block
`4*WWR_WARP_SIZE` threads along the rows (x), with `blockIdx.y` naming the
column. Column-major storage makes the `A` write a coalesced per-row access, and
`idivup` sizes the x-direction block count. Lifting the `gridDim.y` ≤ 65535
column bound is a possible later optimisation, as in `calaman.lascl2`.

## Tested

`test/lascl/` runs the kernel on the device and compares against the reference
`LAPACKE_?lascl` (`TYPE='G'`) over the identical inputs, for `s`, `d`, `c` and `z` to
`test/shared/tolerance.cppm`. `?lascl` applies the same guarded multiplier chain,
so the device and the reference agree to rounding — the comparison is a relative
tolerance, not bit-for-bit. Cases cover square/tall/wide shapes, a leading
dimension `> m` (padding rows that must stay untouched), a `1x1` matrix, ordinary
and over/underflow-provoking `cfrom`/`cto` pairs, and the `m==0`/`n==0`
early-return. The suite is `REQUIRES_GPU` (labeled `gpu`, excluded by
`ctest -LE gpu`) and guarded on `calaman::lapack_reference` at configure time.
