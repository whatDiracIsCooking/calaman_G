# calaman.reduce_columns

Header-only, backend-neutral infrastructure for reducing each column of a
column-major matrix to one value, with a per-element pre-transform and an
associative binary fold. One directory, one header (`reduce_columns.cuh`), no
module and no `.cu` of its own: consumers `#include` it into their own device
translation unit, like WarpWraps' `wwr.extension.parallel_for`.

## API

Both entry points live in `calaman::device` and take a `wwr::wwrStream_t`; both
are asynchronous (enqueue one kernel, no synchronize) and no-op on empty input.

- `reduce_columns_transform<T, ValT>(stream, d_A, d_result, n, ncols, lda, pre, op)`
  — applies `pre` (a `ValT(T)` functor) to each element, folds the results of
  each column under `op` (an associative `ValT(ValT, ValT)` functor), and writes
  `d_result[j]` for column `j`. The input type `T` and reduction type `ValT` may
  differ (e.g. complex → real for a norm).
- `reduce_columns<T>(stream, d_A, d_result, n, ncols, lda, op)` — the common
  `ValT == T`, no-pre-transform case, a thin wrapper over the above with
  `identity_functor<T>`.

`d_A` is column-major: column `j` starts at `d_A + j*lda`, with `lda >= n` so a
submatrix view (columns non-contiguous) works unchanged.

## Why a hand-written kernel, not Thrust

WarpWraps removed its Thrust-based `parallel_for` because Thrust's algorithms
break under relocatable device code (see its `parallel_for.cuh` and the project
memory it cites). This follows that precedent: the segmented reduction is a
hand-written `__global__` kernel rather than `thrust::reduce_by_key`. A bonus is
that it needs no caller-supplied key/scratch buffer — one block owns one column
and writes its single output directly.

## Kernel shape

One block per column; `4 * WWR_WARP_SIZE` threads (128, or 256 on a CDNA build).
Each thread folds its strided share of the column into a private partial,
**seeded by its first element** so the fold is identity-free; the partials then
tree-reduce in shared memory, with a `tid + s < nactive` bound that skips the
ragged tail. So `op` need only be associative — no identity, no commutativity.

## Consumers

`src/columnwise_ell1` (per-column L1 norm: `pre = |·|`, `op = +`). New column
reductions should reach for this rather than re-rolling a reduction.
