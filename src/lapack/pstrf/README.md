# calaman.pstrf — blocked (level-3) pivoted Cholesky

`calaman.pstrf` is LAPACK's `?pstrf`: the **blocked** complete-pivoting Cholesky
factorization of a symmetric positive **semi**definite matrix (Higham/Lucas),

```
P^T A P = U^H U   (Uplo::U)   or   P^T A P = L L^H   (Uplo::L)
```

It is the level-3 driver over the `calaman.pstf2` level-2 panel. The module
(`pstrf.cppm`) is a host composition of the wrapped BLAS over two device stages
in `pstrf.cu` (declared in `pstrf_bridge.h`) — structurally the `pstf2` kernels,
kept separate on the module boundary.

## Two paths, and why this is *not* `laqps → laqp2`

`?pstrf` does **not** delegate to `?pstf2` per panel (the issue #65 note). It
splits on an internal block size `nb`:

- **`nb <= 1` or `nb >= n`** — the whole matrix goes to `calaman.pstf2`, the
  unblocked panel. Small `n` or a degenerate block size.
- **otherwise** — the **inlined** blocked path. Each block factors up to `nb`
  columns with the per-column `gemv` + `scal` + `swap` look-ahead, then one
  `syrk` pushes the block's effect onto the trailing submatrix.

The reason it can't delegate per panel: complete pivoting keeps a running vector
of Schur-complement diagonals, and the column-`j` pivot search is **interleaved**
with that running update — the panel loop and the pivot bookkeeping are fused, so
they are inlined here, not handed to a self-contained kernel.

## The blocked recipe

The driver is **left-looking within a block, right-looking across blocks**. It
maintains the running Schur-complement diagonals in `work`, reset at each block
start, while the already-finished blocks' contribution lives in `A`'s diagonal
via the trailing `syrk`. Per block starting at column `k`, `jb = min(nb, n−k)`:

1. **Reset** the within-block dot products `dots[k:n] = 0`.
2. **Panel**, for each `j` in `[k, k+jb)`:
   - *Diagonals + pivot.* Fold the previous in-block factor row/column into
     `dots`, form the Schur diagonal `A(i,i) − dots[i]` over `i ≥ j`, pick the
     **largest** (complete pivoting — no norm downdate). (`pstrf_pivot` fuses
     this.) The first ever pivot comes from an initial global-max pass that also
     sets the default `TOL`.
   - *Rank test.* If that diagonal is `≤ TOL` (or NaN), stop: `rank = j`, zero
     the trailing factor, return via `info = rank + 1`. The stop can land
     **mid-block**.
   - *Swap.* Bring row/column `pvt` to position `j` (the symmetric triangle
     dance, recorded 1-based in `piv`).
   - *Factor.* `sqrt` the pivot, left-looking `gemv` of the new row/column
     against the **current block's** factor (inner dimension `j − k`, not `j` —
     prior blocks were applied by `syrk`), then `scal` by `1/√pivot`.
3. **Trailing update.** One `syrk` subtracts the finished block's outer product
   from the stored triangle of `A(k+jb:n, k+jb:n)`.

## The block size, and matching the reference

`nb` is a small internal constant (`kPstrfBlockSize = 32`), not LAPACK's
`ILAENV` choice. It need not match: a different `nb` changes only the
floating-point accumulation order of the trailing updates, and pivoted Cholesky
is a **unique** factorization, so `piv`, `rank` and the factor still agree with
the oracle on well-separated diagonals (the same robustness `pstf2`'s test
relies on). A small `nb` is deliberate so modest test shapes span several blocks,
including a non-multiple of `nb`.

## The stopping value and `info`

As `pstf2`: a negative `tol` means the default `N · ε · maxₖ A(k,k)` with `ε =
DLAMCH('E') = numeric_limits::epsilon()/2`, so the stop matches the reference
bit-for-bit. `info` is `0` on a full-rank factorization (`rank == n`) and
`rank + 1` (a positive *success* code) on a rank-revealing or NaN stop — the
calaman `?pst*` convention, which departs from LAPACK's flat `info = 1`; the
returned `Status` carries only device/BLAS errors.

## The oracle

`LAPACKE_?pstrf` run on the whole matrix: `piv`, `rank`, the triangular factor,
and the backward error of the reconstructed `P^T A P` all agree.

## Not yet

- **Complex `c`/`z`** is a deliberate later extension, for the reasons
  `larfg`/`laqp2`/`pstf2` document: a Hermitian pivot and conjugation differ
  materially from a trivial instantiation (and `herk`, not `syrk`, for the
  trailing update).
