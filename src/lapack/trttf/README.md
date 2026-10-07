# `calaman.trttf`

The GPU counterpart of LAPACK's `?trttf`: convert the `uplo` triangle of an n×n
full column-major matrix `A` into Rectangular Full Packed (RFP) storage `ARF`.
One entry point, templated over the four `usual_fp` element types:

```cpp
import calaman.trttf;   // also re-exports calaman::Trans, Uplo and Status
import wwr.runtime_api;

// d_a: n-by-n device matrix, leading dim lda; d_arf: n(n+1)/2 RFP elements
calaman::trttf(stream, calaman::Trans::N, calaman::Uplo::L, n, d_a, lda, d_arf);
```

Only the `uplo` triangle of `A` (diagonal included) is read. The conversion is
enqueued on `stream` and returns without synchronizing. A stream, not a device
handle, because nothing is allocated (see `test/shared/README.md`).

## RFP layout

With `n1 = n/2` and `n2 = n - n1`, the `TRANSR = 'N'` layout is a column-major
`(n + 1 - n%2)`-by-`n2` array: one trapezoid of the triangle sits in place and
the remaining `n1`- or `n2`-sized triangle is stored as its transpose in the
otherwise-unused corner — **conjugate** transpose for a complex `T`, the
Hermitian convention. `TRANSR = 'T'`/`'C'` stores the (conjugate) transpose of
that whole array. Which corner depends on `UPLO` and the parity of `n`, giving
LAPACK's 8 cases; the map lives once, in `calaman::device::rfp_index`
(`src/lapack/tri_index/tri_index.cuh`), shared with tfttr, tpttf and tfttp.
The same layout per *block* -- where A11, A22 and the off-diagonal block sit,
for routines that run BLAS on RFP sub-blocks (?hfrk/?sfrk, ?tfsm) -- is the
host constexpr `calaman::rfp_blocks` (`src/lapack/rfp_blocks/rfp_blocks.cppm`).

## Mapping from ?TRTTF

`CHARACTER TRANSR` becomes `calaman::Trans` and `CHARACTER UPLO` becomes
`calaman::Uplo`; the `s/d/c/z` variants become one template over `T`;
`N`/`LDA` become `std::size_t`; `INFO` becomes a returned `calaman::Status`.
`TRANSR` keeps LAPACK's per-type rule — `Trans::N`, or `Trans::T` for a real
`T` and `Trans::C` for a complex one — and anything else, like `lda < max(1, n)`,
returns `wwrErrorInvalidValue`.

## Shape

`interface.cppm` is the host wrapper; `trttf.cu` the device half — one thread
per element of the n×n square, skipping those outside the triangle;
`trttf_bridge.h` carries the launcher declaration across the host/device
boundary.

## Tested

`test/trttf/` checks the device `ARF` bit-for-bit against `LAPACKE_?trttf` for
s/d/c/z, all 8 `TRANSR` × `UPLO` × n-parity cases, `lda > n`, and
n ∈ {0, 1, small, non-power-of-two}. The opposite triangle and padding of `A`
hold distinct values, so reading one of them shows up in `ARF`. It is
`REQUIRES_GPU` and builds only when `calaman::lapack_reference` is present.
