# calaman.hetf2 — unblocked Hermitian Bunch-Kaufman factorization

`calaman.hetf2` is LAPACK's `?hetf2`: the **unblocked** Bunch-Kaufman
factorization of a Hermitian indefinite matrix,

```
A = U D U^H   (Uplo::U)   or   A = L D L^H   (Uplo::L)
```

with `D` block diagonal (real 1×1 and Hermitian 2×2 blocks) and the symmetric
interchanges recorded in `ipiv` (1-based, a negative pair marks a 2×2 block).

**Complex only** (`c`/`z`). A real Hermitian matrix is symmetric — that is the
vendor's `?sytrf` (`calaman.sysv`).

## Why this exists

cuSOLVER and hipSOLVER ship `sytrf` (symmetric) but **no `hetrf`** (Hermitian).
So where `calaman.sysv` factors via a single vendor call, the Hermitian solve has
to own its factorization. `calaman.hetf2` is that factorization (unblocked);
`calaman.hesv` pairs it with `calaman.hetrs` for the end-to-end solve. A blocked
`hetrf` (via `lahef`) would be a later performance refinement.

## Structure — the reference `?hetf2`, host-driven

A host loop walks the columns (`k = n…1` upper, `1…n` lower, in steps of 1 or 2):

1. **Pivot search + decision** — host-side, the Bunch-Kaufman criteria (`alpha =
   (1+√17)/8`): `wwr::iamax` finds the column max (`COLMAX`/`IMAX`); if an
   interchange is indicated, a second `iamax` finds the row max (`ROWMAX`); a
   handful of per-step scalar reads (`cabs1`, real diagonals) feed the branch that
   picks `kp` and a 1×1 or 2×2 pivot.
2. **Interchange** — the Hermitian swap of the pivot pair, a conjugating
   row/column dance over the stored triangle (`hetf2_interchange`, single thread).
3. **Update** — 1×1: `wwr::her` rank-1 then `wwr::scal` to store the factor column.
   2×2: `hetf2_rank2` forms the two factor columns `W`, subtracts `W inv(D) W^H`
   from the trailing triangle, stores `W`, and keeps the diagonal real.

The two non-BLAS stages live in `hetf2.cu`. Pivots and `info` are accumulated
host-side and uploaded once at the end; each pivot step synchronizes to read the
scalars its decision needs (the unblocked routine is correctness-first, not the
performance path).

## Workspace

`2n` device elements — the two factor-column scratch vectors a 2×2 update needs.
`hetf2_bufferSize<T>(n)` returns `2n`.

## `info`

`0` on success, else the 1-based first column with a zero or NaN pivot — the
reference convention. The returned `Status` carries only device/BLAS errors.

## The oracle

Exercised through `calaman.hesv` against `LAPACKE_?hesv`: the device factor differs
from the reference's (unblocked vs blocked), but the solution of `A X = B` is
unique, so a small backward-error residual confirms the factorization.
