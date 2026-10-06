# `calaman.trttp`

The GPU counterpart of LAPACK's `?trttp`: pack the `uplo` triangle of an n×n
full column-major matrix `A` into packed storage `AP`. One entry point,
templated over the four `usual_fp` element types:

```cpp
import calaman.trttp;   // also re-exports calaman::Uplo and Status
import wwr.runtime_api;

// d_a: n-by-n device matrix, leading dim lda; d_ap: n(n+1)/2 packed elements
calaman::trttp(stream, calaman::Uplo::L, n, d_a, lda, d_ap);
```

Only the `uplo` triangle of `A` (diagonal included) is read; the opposite
triangle and the `lda > n` padding rows are never touched. The copy is enqueued
on `stream` and returns without synchronizing. A stream, not a device handle,
because nothing is allocated (see `test/shared/README.md`).

## Not cuBLAS

`cublas{S,D,C,Z}trttp` exists, but only in `wwr.cuda.cublas_v2`: WarpWraps
leaves it out of the portable `wwrblas*` surface because hipBLAS has no
counterpart. This module is its own `parallel_for` kernel so it builds for both
backends.

## Mapping from ?TRTTP

`CHARACTER UPLO` becomes `calaman::Uplo`; the `s/d/c/z` variants become one
template over `T`; `N`/`LDA` become `std::size_t`; `INFO` becomes a returned
`calaman::Status` — `wwrErrorInvalidValue` when `lda < max(1, n)` (the only
argument error left once `UPLO` is typed and `N` unsigned).

## Shape

`interface.cppm` is the host wrapper; `trttp.cu` the device half — one thread
per element of the n×n square, skipping those outside the triangle;
`trttp_bridge.h` carries the launcher declaration across the host/device
boundary. The packed index comes from the shared header-only
`calaman::tri_index` (`src/lapack/tri_index/tri_index.cuh`), the same map
`calaman.tpttr` inverts.

## Tested

`test/trttp/` checks the device `AP` bit-for-bit against `LAPACKE_?trttp` for
s/d/c/z, both `uplo`, `lda > n`, and n ∈ {0, 1, small, non-power-of-two}. The
opposite triangle and padding of `A` hold distinct sentinels, so reading one of
them shows up in `AP`. It is `REQUIRES_GPU` and builds only when
`calaman::lapack_reference` is present.
