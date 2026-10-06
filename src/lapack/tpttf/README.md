# `calaman.tpttf`

The GPU counterpart of LAPACK's `?tpttf`: convert the `uplo` triangle of an n×n
matrix from packed storage `AP` into Rectangular Full Packed (RFP) storage
`ARF`. One entry point, templated over the four `usual_fp` element types:

```cpp
import calaman.tpttf;   // also re-exports calaman::Trans, Uplo and Status
import wwr.runtime_api;

// d_ap: n(n+1)/2 packed elements; d_arf: n(n+1)/2 RFP elements
calaman::tpttf(stream, calaman::Trans::N, calaman::Uplo::L, n, d_ap, d_arf);
```

The conversion is enqueued on `stream` and returns without synchronizing. A
stream, not a device handle, because nothing is allocated (see
`test/shared/README.md`). It goes straight from one format to the other — no
full matrix is formed — through the packed and RFP maps of the shared
`src/lapack/tri_index/tri_index.cuh` (`packed_index`, `rfp_index`). The RFP
layout and its 8 `TRANSR` × `UPLO` × n-parity cases are described in
`src/lapack/trttf/README.md`.

## Mapping from ?TPTTF

`CHARACTER TRANSR` becomes `calaman::Trans` and `CHARACTER UPLO` becomes
`calaman::Uplo`; the `s/d/c/z` variants become one template over `T`; `N`
becomes `std::size_t`; `INFO` becomes a returned `calaman::Status`. `TRANSR`
keeps LAPACK's per-type rule — `Trans::N`, or `Trans::T` for a real `T` and
`Trans::C` for a complex one — and anything else returns `wwrErrorInvalidValue`.

## Shape

`interface.cppm` is the host wrapper; `tpttf.cu` the device half — one thread
per element of the n×n square, skipping those outside the triangle;
`tpttf_bridge.h` carries the launcher declaration across the host/device
boundary.

## Tested

`test/tpttf/` checks the device `ARF` bit-for-bit against `LAPACKE_?tpttf` for
s/d/c/z, all 8 `TRANSR` × `UPLO` × n-parity cases, and
n ∈ {0, 1, small, non-power-of-two}; `ARF` starts as sentinels, so a slot left
unwritten shows up, and `AP` is checked unchanged. It is `REQUIRES_GPU` and
builds only when `calaman::lapack_reference` is present.
