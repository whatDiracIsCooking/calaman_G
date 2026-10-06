# `calaman.tfttr`

The GPU counterpart of LAPACK's `?tfttr`: convert an n×n triangle from
Rectangular Full Packed (RFP) storage `ARF` into the `uplo` triangle of full
column-major `A` — the inverse of `calaman.trttf`. One entry point, templated
over the four `usual_fp` element types:

```cpp
import calaman.tfttr;   // also re-exports calaman::Trans, Uplo and Status
import wwr.runtime_api;

// d_arf: n(n+1)/2 RFP elements; d_a: n-by-n device matrix, leading dim lda
calaman::tfttr(stream, calaman::Trans::N, calaman::Uplo::L, n, d_arf, d_a, lda);
```

Only the `uplo` triangle of `A` (diagonal included) is written; the opposite
triangle and the `lda > n` padding rows are left untouched, as in the
reference. The conversion is enqueued on `stream` and returns without
synchronizing. A stream, not a device handle, because nothing is allocated (see
`test/shared/README.md`).

The RFP layout and its 8 `TRANSR` × `UPLO` × n-parity cases are described in
`src/lapack/trttf/README.md`; the map itself is `calaman::device::rfp_index`
in the shared `src/lapack/tri_index/tri_index.cuh`, so the two directions
cannot disagree.

## Mapping from ?TFTTR

`CHARACTER TRANSR` becomes `calaman::Trans` and `CHARACTER UPLO` becomes
`calaman::Uplo`; the `s/d/c/z` variants become one template over `T`;
`N`/`LDA` become `std::size_t`; `INFO` becomes a returned `calaman::Status`.
`TRANSR` keeps LAPACK's per-type rule — `Trans::N`, or `Trans::T` for a real
`T` and `Trans::C` for a complex one — and anything else, like `lda < max(1, n)`,
returns `wwrErrorInvalidValue`.

## Shape

`interface.cppm` is the host wrapper; `tfttr.cu` the device half — one thread
per element of the n×n square, skipping those outside the triangle;
`tfttr_bridge.h` carries the launcher declaration across the host/device
boundary.

## Tested

`test/tfttr/` checks the device `A` bit-for-bit against `LAPACKE_?tfttr` for
s/d/c/z, all 8 `TRANSR` × `UPLO` × n-parity cases, `lda > n`, and
n ∈ {0, 1, small, non-power-of-two}, comparing the whole `lda`-by-n buffer so
the untouched triangle and the padding rows are checked too. It is
`REQUIRES_GPU` and builds only when `calaman::lapack_reference` is present.
