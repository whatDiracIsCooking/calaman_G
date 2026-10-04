# `calaman.laset`

The GPU counterpart of LAPACK's `?laset`: set all or part of a column-major
matrix `A` to constants on the device — `beta` on the diagonal, `alpha` off it.
One entry point, templated over `float` and `double`:

```cpp
import calaman.laset;   // also re-exports calaman::Region
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_a: device matrix, column-major, leading dimension lda
// identity: alpha = 0 off the diagonal, beta = 1 on it
calaman::laset(stream, calaman::Region::A, m, n, 0.0, 1.0, d_a, lda);
```

`Region` selects the part set — `A` (all of the matrix), `U` (the upper
triangle, diagonal and above), or `L` (the lower triangle, diagonal and below).
Within the region, the diagonal `A(i,i)` (for `0 <= i < min(m,n)`) is set to
`beta` and every other element to `alpha`; elements of `A` outside the region
are left untouched. The fill is enqueued on `stream` and returns without
synchronizing, like a BLAS call; the caller synchronizes when it needs `A`.

`Region` lives in `calaman.common` (its `:enums` partition, shared with the
device `.cu` through `common/enums.h`) and is re-exported by `calaman.laset`, so
`import calaman.laset;` alone names it. It is kept distinct from `Uplo` — which
is strictly `U`/`L` for the symmetric routines — because `?lacpy`/`?laset`
overload `uplo` with the third "all of the matrix" case.

A stream and not a device handle, because this routine allocates nothing: it
needs neither a device index nor a memory pool. That also keeps a concrete handle
type — and the error policy such a type hard-codes — out of calaman's shipped
surface, matching `calaman.lacpy`'s bare `wwrStream_t`. See
`test/shared/README.md` for the full reasoning and for what a future routine
*with* a workspace should do instead.

## Mapping from DLASET

Kept: the name, the fill constants `alpha` (off-diagonal) and `beta` (diagonal),
and the leading dimension `lda` — that last is a fact of column-major storage,
not a Fortran accommodation. Changed, per `docs/architecture.md` §4: the
`CHARACTER*1 UPLO` becomes the typed `calaman::Region` enum (`common/enums.h`),
the `s/d/c/z` variants become one template over `T`, and the `INTEGER` extents
become `std::size_t`. There is no `INFO` — `?laset` is an auxiliary routine that
reports none.

The diagonal always receives `beta`: `U` admits `i == j` through `i <= j`, `L`
through `i >= j`, and `A` sets everything — matching DLASET, which writes the
strictly-upper/lower (or whole) off-diagonal to `ALPHA` and then overwrites the
`min(m,n)` diagonal with `BETA` in a separate pass.

Complex (`c`/`z`) is a straightforward extension: add the type to the three
lists that must stay in step — `interface.cppm`'s `extern template`,
`instantiations.cpp`, and `laset.cu`.

## Shape

`interface.cppm` is the host wrapper (it re-exports `Region` from
`calaman.common`); `laset.cu` is the device half; `laset_bridge.h` carries the
launcher declaration across the host/device boundary (a global module fragment
cannot `import`) and passes `Region` across it by `#include`. `instantiations.cpp`
explicitly instantiates the wrapper for each type.

The fill is one hand-launched kernel: a 2-D grid of 1-D blocks, each block
`4*WWR_WARP_SIZE` threads along the rows (x), with `blockIdx.y` naming the
column. Column-major storage makes that a coalesced access, and `idivup`
(`calaman.common`, `common/align_up.h`) sizes the x-direction block count. A
triangular region launches the full grid and skips out-of-region elements rather
than iterating a packed triangular index range — correct and simplest; skipping
the wasted threads, and lifting the `gridDim.y` ≤ 65535 column bound that the
one-block-per-column mapping carries, are possible later optimisations.

## Tested

Not currently — like `calaman.lacpy`, `laset` has no automated suite yet.
Reinstating one needs an exact device-side comparison against the reference
LAPACK (e.g. via `calaman.diff_norm` or a replacement for the removed
`calaman.test.elementwise_compare` utility).
