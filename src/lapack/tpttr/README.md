# `calaman.tpttr`

The GPU counterpart of LAPACK's `?tpttr`: unpack the `uplo` triangle of an
n×n matrix from packed storage `AP` into full column-major `A`. One entry point,
templated over the four `usual_fp` element types:

```cpp
import calaman.tpttr;   // also re-exports calaman::Uplo and Status
import wwr.runtime_api;

// d_ap: n(n+1)/2 packed elements; d_a: n-by-n device matrix, leading dim lda
calaman::tpttr(stream, calaman::Uplo::L, n, d_ap, d_a, lda);
```

Only the `uplo` triangle of `A` (diagonal included) is written; the opposite
triangle is left untouched, as in the reference. The copy is enqueued on
`stream` and returns without synchronizing. A stream, not a device handle,
because nothing is allocated (see `test/shared/README.md`).

## Not cuBLAS

`cublas{S,D,C,Z}tpttr` exists, but only in `wwr.cuda.cublas_v2`: WarpWraps
leaves it out of the portable `wwrblas*` surface because hipBLAS has no
counterpart. This module is its own `parallel_for` kernel so it builds for both
backends.

## Mapping from ?TPTTR

`CHARACTER UPLO` becomes `calaman::Uplo`; the `s/d/c/z` variants become one
template over `T`; `N`/`LDA` become `std::size_t`; `INFO` becomes a returned
`calaman::Status` — `wwrErrorInvalidValue` when `lda < max(1, n)` (the only
argument error left once `UPLO` is typed and `N` unsigned).

## Shape

`interface.cppm` is the host wrapper; `tpttr.cu` the device half — one thread
per element of the n×n square, skipping those outside the triangle;
`tpttr_bridge.h` carries the launcher declaration across the host/device
boundary. The packed index comes from the shared header-only
`calaman::tri_index` (`src/lapack/tri_index/tri_index.cuh`).

## Tested

`test/tpttr/` checks the device result bit-for-bit against `LAPACKE_?tpttr`
for s/d/c/z, both `uplo`, `lda > n`, and n ∈ {0, 1, small, non-power-of-two},
comparing the whole `lda`-by-n buffer so the untouched triangle and the padding
rows are checked too. It is `REQUIRES_GPU` and builds only when
`calaman::lapack_reference` is present.
