# calaman.pstf2 — level-2 pivoted Cholesky

`calaman.pstf2` is LAPACK's `?pstf2`: the unblocked, complete-pivoting Cholesky
factorization of a symmetric positive **semi**definite matrix (Higham/Lucas),

```
P^T A P = U^H U   (Uplo::U)   or   P^T A P = L L^H   (Uplo::L)
```

It is the level-2 panel the blocked `?pstrf` driver would run in a loop; here it
factors the whole matrix. The module (`pstf2.cppm`) is a host composition of the
wrapped BLAS over two device stages in `pstf2.cu` (declared in
`pstf2_bridge.h`) — the same module/`.cu` split `laqp2`/`laqps`/`nnls` use.

## Why `pst*`, not `potf2`

`?potf2` factors a positive-**definite** matrix and treats a non-positive pivot
as breakdown (a hard error). `?pst*` factors a positive-**semi**definite one and
treats a vanishing pivot as the **rank boundary**: it stops cleanly, reports how
far it got, and the stop is a *success*. Transcribing `potf2`'s breakdown here
would defeat the entire point of the routine. See the issue (#64) for the
acceptance items this encodes.

## The per-step recipe

`?pstf2` is **left-looking**: it never updates the trailing submatrix. Instead it
maintains the running Schur-complement diagonals in a workspace vector and
reconstructs each factor row/column from the already-computed factor. Per step
`j` (0-based):

1. **Diagonals.** For each remaining `i ≥ j`, fold the previous step's factor
   entry into the running dot product `dots[i]`, then form the Schur diagonal
   `A(i,i) − dots[i]`. (`pstf2_pivot` fuses this with step 2.)
2. **Pivot.** Pick the index of the **largest** Schur diagonal — complete
   pivoting hits rows and columns together, so the pivot comes straight from the
   diagonals with no norm downdate (the departure from `laqp2`'s column-only
   pivoting).
3. **Rank test.** If that largest diagonal is `≤ tol` (or NaN), stop:
   `rank = j`, zero the trailing factor, return via `info = rank + 1`.
4. **Swap.** Bring row/column `pvt` to position `j` with the symmetric
   permutation. On a triangle-only store this is a diagonal-corner copy plus
   three strided swaps that reflect the off-diagonals across the diagonal (the
   reference `?pstf2` dance), recorded 1-based in `piv`.
5. **Factor.** `sqrt` the pivot into the diagonal, left-looking `gemv` of the new
   row/column against the computed factor, then `scal` by `1/√pivot`.

## The stopping value and `info`

When the caller passes a negative `tol`, the default is `N · ε · maxₖ A(k,k)`,
with `ε` taken as `DLAMCH('E')` = `numeric_limits::epsilon()/2` so the stop
matches the reference LAPACK bit-for-bit. `info` is `0` on a full-rank
factorization (`rank == n`) and `rank + 1` (a positive *success* code) on a
rank-revealing or NaN stop; the returned `Status` carries only device/BLAS
errors.

## The oracle

LAPACKE ships no `?pstf2` binding, so the test oracle is `LAPACKE_?pstrf` run on
the whole matrix — `?pstrf` and `?pstf2` produce the same unique pivoted
factorization, so `piv`, `rank`, the triangular factor, and the backward error of
the reconstructed `P^T A P` all agree.

## Not yet

- **Complex `c`/`z`** is a deliberate later extension, for the reasons
  `larfg`/`laqp2` document: a Hermitian pivot and conjugation differ materially
  from a trivial instantiation.
