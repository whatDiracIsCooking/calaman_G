# `calaman.lascl2`

The GPU counterpart of LAPACK's `?lascl2`: scale a column-major matrix `X` in
place by a diagonal matrix `D`, held as a length-`m` vector — `X <- D * X`, i.e.
`X(i,j) <- d(i) * X(i,j)`. One entry point, templated over `float` and `double`:

```cpp
import calaman.lascl2;   // also re-exports calaman::Status
import wwr.runtime_api;

wwr::wwrStream_t stream{};
wwr::wwrStreamCreate(&stream);
// d_d: length-m device vector of row scale factors
// d_x: device matrix, column-major, leading dimension ldx
calaman::lascl2(stream, m, n, d_d, d_x, ldx);
```

Each element `X(i,j)` is multiplied by `d(i)`, the scale factor for its row. The
scaling is enqueued on `stream` and returns without synchronizing, like a BLAS
call; the caller synchronizes when it needs `X`. `D` and `X` are device pointers
the caller owns; nothing is allocated here.

A stream and not a device handle, because this routine allocates nothing: it
needs neither a device index nor a memory pool. That also keeps a concrete
handle type — and the error policy such a type hard-codes — out of calaman's
shipped surface, matching `calaman.laset`'s bare `wwrStream_t`. See
`test/shared/README.md` for the full reasoning and for what a future routine
*with* a workspace should do instead.

## Mapping from DLASCL2

Kept: the name, the length-`m` diagonal vector `D`, and the leading dimension
`ldx` — that last is a fact of column-major storage, not a Fortran
accommodation. Changed, per `docs/architecture.md` §4: the `s/d` variants become
one template over `T`, and the `INTEGER` extents become `std::size_t`. There is
no `INFO` — `?lascl2` is an auxiliary routine that reports none.

`DLASCL2` (unlike the better-known `?lascl`) does no overflow-safe rescaling and
takes no `type`/`cfrom`/`cto`: it is the plain diagonal product, destined in the
reference to be replaced by `BLAS_dge_diag_scale`.

Complex (`c`/`z`) is a straightforward extension: add the type to the three
lists that must stay in step — `interface.cppm`'s `extern template`,
`instantiations.cpp`, and `lascl2.cu`.

## Shape

`interface.cppm` is the host wrapper; `lascl2.cu` is the device half;
`lascl2_bridge.h` carries the launcher declaration across the host/device
boundary (a global module fragment cannot `import`). `instantiations.cpp`
explicitly instantiates the wrapper for each type. Unlike `calaman.laset`, the
host module links no `calaman.common`: `lascl2` names no `Region` enum, so its
GMF includes no `common/` header; only the device library needs `calaman.common`,
for `idivup` (`common/align_up.h`).

The scaling is one hand-launched kernel: a 2-D grid of 1-D blocks, each block
`4*WWR_WARP_SIZE` threads along the rows (x), with `blockIdx.y` naming the
column. Column-major storage makes both the `X` write and the `d(i)` read a
coalesced per-row access, and `idivup` sizes the x-direction block count.
Lifting the `gridDim.y` ≤ 65535 column bound that the one-block-per-column
mapping carries is a possible later optimisation.

## Tested

`test/lascl2/` runs the kernel on the device and compares against a same-precision
host reference — the literal transcription of DLASCL2's `x(i,j) *= d(i)` double
loop. There is no LAPACKE (or BLAS) binding for `?lascl2` to use as an oracle, so
a host transcription of its (definitional, arithmetic-free-of-reordering) loop is
the reference; inputs are small exact integers, so every product is exact in `T`
and the comparison is bit-for-bit `==`, as in `calaman.laset`. The suite is
`REQUIRES_GPU` (labeled `gpu`, excluded by `ctest -LE gpu`).
