# `calaman.lacpy`

The GPU counterpart of LAPACK's `?lacpy`: copy all or part of a column-major
matrix `A` to `B` on the device. One entry point, templated over `float` and
`double`:

```cpp
import calaman.lacpy;
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_a, d_b: device matrices, column-major, leading dimensions lda / ldb
calaman::lacpy(stream, calaman::copy_region::upper, m, n, d_a, lda, d_b, ldb);
```

`copy_region` selects the part copied — `full`, `upper` (diagonal and above), or
`lower` (diagonal and below). Elements of `B` outside the copied region are left
untouched. The copy is enqueued on `stream` and returns without synchronizing,
like a BLAS call; the caller synchronizes when it needs `B`.

A stream and not a device handle, because this routine allocates nothing: it
needs neither a device index nor a memory pool. That also keeps a concrete handle
type — and the error policy such a type hard-codes — out of calaman's shipped
surface, matching `calaman.diff_norm`'s bare `wwrblasHandle_t`. See
`test/shared/README.md` for the full reasoning and for what a future routine
*with* a workspace should do instead.

## Mapping from DLACPY

Kept: the name, and the leading dimensions `lda`/`ldb` — those are a fact of
column-major storage, not a Fortran accommodation. Changed, per
`docs/architecture.md` §4: the `CHARACTER*1 UPLO` becomes the typed
`copy_region` enum, the `s/d/c/z` variants become one template over `T`, and the
`INTEGER` extents become `std::size_t`. There is no `INFO` — `?lacpy` is an
auxiliary routine that reports none.

Complex (`c`/`z`) is a straightforward extension: add the type to the three
lists that must stay in step — `interface.cppm`'s `extern template`,
`instantiations.cpp`, and `lacpy.cu`.

## Shape

`interface.cppm` is the host wrapper and the `copy_region` enum; `lacpy.cu` is
the device half; `lacpy_bridge.h` carries the launcher declaration across the
host/device boundary (a global module fragment cannot `import`).
`instantiations.cpp` explicitly instantiates the wrapper for each type.

`test/utils/elementwise_compare/` has the same four-file shape, but takes a
`calaman::test::DeviceHandle` rather than a stream — it allocates scratch, so it
needs the device index and pool that a stream does not carry. The difference is
the rule, not an inconsistency: each takes the narrowest thing it uses.

The copy is one `wwr.extension.parallel_for` map over the `m*n` elements. A
triangular region launches the full `m*n` grid and skips out-of-region elements
rather than iterating a packed triangular index range — correct and simplest;
skipping the wasted threads is a possible later optimisation.

## Tested

Against the reference LAPACK oracle (`LAPACKE_?lacpy`) at
`test/test/lacpy/` — the copy is exact, so the GPU result must match the oracle
bit for bit, including the untouched region. That suite is skipped when
`calaman::lapack_reference` is absent (`docs/architecture.md` §3).
