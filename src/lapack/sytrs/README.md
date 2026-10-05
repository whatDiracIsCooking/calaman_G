# calaman.sytrs — symmetric Bunch-Kaufman solve

`calaman.sytrs` is LAPACK's `?sytrs`: given a symmetric matrix already factored
by the Bunch-Kaufman method,

```
A = U D U^T   (Uplo::U)   or   A = L D L^T   (Uplo::L)
```

it solves `A X = B`, overwriting the right-hand sides `B` with the solution `X`.
`D` is block diagonal with 1×1 and 2×2 blocks; the factor and the pivot sequence
`ipiv` come from `?sytrf` / `wwr::sytrf` for the same `uplo`.

**Complex is symmetric, not Hermitian** — no conjugation anywhere (`geru`, not
`gerc`; `OP_T`, not `OP_C`; `1/A(k,k)` with no real-part projection). The
Hermitian cousin is `calaman.hetrs`.

## Structure — the reference level-2 solve, verbatim

`sytrs.cppm` is a host loop that walks the pivot sequence exactly as reference
`?sytrs` does, issuing one wrapped BLAS call per step. The two solve phases per
`uplo`:

- **`L D X = B` / `U D X = B`** — forward (lower) or backward (upper) substitution
  through the unit-triangular factor: a `swap` for the row interchange, a `ger`
  (real) / `geru` (complex) rank-1 update against the factor column, then the
  `D^-1` block apply.
- **`L^T X = B` / `U^T X = B`** — the transpose back-substitution: one `gemv`
  (`OP_T`) per column against the factor, then the row interchange.

The pivot array drives host control flow, so it is copied device→host **once**
(the routine's only synchronization); every BLAS call and kernel after that is
enqueued on the handle's stream.

## The one device stage — applying `D^-1` (`sytrs.cu`)

Everything above is wrapped BLAS; the block-diagonal `D^-1` is the piece no BLAS
expresses, so it lives in `sytrs.cu` (declared in `sytrs_bridge.h`), two functors
through `wwr.extension.parallel_for`, one thread per right-hand-side column:

- **1×1** — `sytrs_scale_row`: divide one `B` row by the stored `D(k,k)`.
- **2×2** — `sytrs_solve_2x2`: the symmetric 2×2 inverse across two `B` rows, the
  `AKM1K`/`AKM1`/`AK`/`DENOM` arithmetic the reference spells per column, read
  straight off the stored factor via `calaman::device::elem_ops` (so one body
  covers real and complex). No host round-trip per pivot.

## Pivots

`ipiv` is the device array `wwr::sytrf` wrote — 1-based, with the Bunch-Kaufman
sign convention (positive = a 1×1 pivot and its interchange; a negative pair
marks a 2×2 block). It is read, never written.

## The oracle

`LAPACKE_?sytrf` factors a symmetric indefinite matrix; the device `sytrs` and
`LAPACKE_?sytrs` then solve from the same factor and pivots, and the two `X`
agree to tolerance (and `A X` reproduces the original `B`).
