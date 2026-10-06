# `calaman.tfttp`

The GPU counterpart of LAPACK's `?tfttp`: convert an n×n triangle from
Rectangular Full Packed (RFP) storage `ARF` into the packed storage `AP` of its
`uplo` triangle — the inverse of `calaman.tpttf`. One entry point, templated
over the four `usual_fp` element types:

```cpp
import calaman.tfttp;   // also re-exports calaman::Trans, Uplo and Status
import wwr.runtime_api;

// d_arf: n(n+1)/2 RFP elements; d_ap: n(n+1)/2 packed elements
calaman::tfttp(stream, calaman::Trans::N, calaman::Uplo::L, n, d_arf, d_ap);
```

The conversion is enqueued on `stream` and returns without synchronizing. A
stream, not a device handle, because nothing is allocated (see
`test/shared/README.md`). It goes straight from one format to the other — no
full matrix is formed — through the packed and RFP maps of the shared
`src/lapack/tri_index/tri_index.cuh` (`packed_index`, `rfp_index`). The RFP
layout and its 8 `TRANSR` × `UPLO` × n-parity cases are described in
`src/lapack/trttf/README.md`.

## Mapping from ?TFTTP

`CHARACTER TRANSR` becomes `calaman::Trans` and `CHARACTER UPLO` becomes
`calaman::Uplo`; the `s/d/c/z` variants become one template over `T`; `N`
becomes `std::size_t`; `INFO` becomes a returned `calaman::Status`. `TRANSR`
keeps LAPACK's per-type rule — `Trans::N`, or `Trans::T` for a real `T` and
`Trans::C` for a complex one — and anything else returns `wwrErrorInvalidValue`.

## Shape

`interface.cppm` is the host wrapper; `tfttp.cu` the device half — one thread
per element of the n×n square, skipping those outside the triangle;
`tfttp_bridge.h` carries the launcher declaration across the host/device
boundary.

## Tested

`test/tfttp/` checks the device `AP` bit-for-bit against `LAPACKE_?tfttp` for
s/d/c/z, all 8 `TRANSR` × `UPLO` × n-parity cases, and
n ∈ {0, 1, small, non-power-of-two}; `AP` starts as sentinels, so a slot left
unwritten shows up, and `ARF` is checked unchanged. It is `REQUIRES_GPU` and
builds only when `calaman::lapack_reference` is present.
