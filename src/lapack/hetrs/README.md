# calaman.hetrs — Hermitian Bunch-Kaufman solve

`calaman.hetrs` is LAPACK's `?hetrs`: given a Hermitian matrix already factored by
the Bunch-Kaufman method,

```
A = U D U^H   (Uplo::U)   or   A = L D L^H   (Uplo::L)
```

it solves `A X = B`, overwriting `B` with `X`. `D` is block diagonal with real
1×1 and Hermitian 2×2 blocks; the factor and pivots come from `?hetrf`.

**Complex only** (`c`/`z`). A real Hermitian matrix is symmetric — that is
`calaman.sytrs`.

## Same shape as `sytrs`, plus the Hermitian twists

The host loop walks the pivot sequence exactly as the symmetric `calaman.sytrs`
does; it diverges only where Hermitian-ness requires it (the reference `?hetrs`):

- **Rank-1 updates** use `geru` (unconjugated), as in the complex-symmetric case.
- **1×1 `D` apply** divides the `B` row by the **real** diagonal `D(k,k)`
  (`hetrs_scale_row_real`).
- **2×2 `D` apply** is Hermitian: the off-diagonal is conjugated on one row
  (`hetrs_solve_2x2`, with a `top_conj` flag selecting which row by `uplo`).
- **Back-substitution** is conjugate-transpose: each `gemv` uses `OP_C`, bracketed
  by conjugating the target `B` row (`hetrs_conj_row` — the reference's
  `ZLACGV`/`ZGEMV('C')`/`ZLACGV`), so a plain `gemv('C')` realizes the `U^H`/`L^H`
  update without needing scratch for the conjugate of a factor column.

Those three device stages live in `hetrs.cu` (declared in `hetrs_bridge.h`), each
a functor through `wwr.extension.parallel_for`. The pivot array is read
device→host once (the only synchronization); everything else is enqueued on the
handle's stream.

## The oracle

`LAPACKE_?hetrf` factors a Hermitian indefinite matrix; the device `hetrs` and
`LAPACKE_?hetrs` then solve from the same factor and pivots, and the two `X` agree
to tolerance (and `A X` reproduces `B`).
