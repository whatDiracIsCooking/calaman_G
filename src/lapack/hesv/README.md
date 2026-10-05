# calaman.hesv — Hermitian indefinite solve

`calaman.hesv` is LAPACK's `?hesv`: solve `A X = B` for a **Hermitian indefinite**
matrix `A`. Two steps, like the reference driver:

1. **Factor** — `A = U D U^H` (`Uplo::U`) or `L D L^H` (`Uplo::L`) by the Hermitian
   Bunch-Kaufman method.
2. **Solve** — with the factor and the pivots.

**Complex only** (`c`/`z`). A real Hermitian matrix is symmetric — that is
`calaman.sysv`.

## All of it is ours

Unlike `calaman.sysv`, whose factorization is a single vendor call (`wwr::sytrf`),
the Hermitian factorization has **no vendor counterpart** — cuSOLVER/hipSOLVER
ship `sytrf` but not `hetrf`. So `calaman.hesv` factors with `calaman.hetf2` (the
from-scratch unblocked Hermitian Bunch-Kaufman) and solves with `calaman.hetrs2`
(level-3) when the workspace holds at least `n` elements — the reference `?hesv`'s
`lwork` test — else `calaman.hetrs` (level-2). Both halves run on the one BLAS
handle; there is no solver handle.

`hesv_bufferSize` forwards to `hetf2_bufferSize` (= `2n` elements), so a size query
and the routine reserve the same workspace.

## `info`, and not solving a singular factor

`hesv` reads the factorization's `info` once (one synchronization). As the
reference `?hesv` does, a non-zero `info` (a zero/NaN pivot column) leaves `B`
untouched and returns; the numerical outcome rides `d_info` (device). The returned
`Status` carries only device/BLAS errors.

## The oracle

`LAPACKE_?hesv` on the same Hermitian indefinite system: the device factor differs
from the reference's (unblocked vs blocked `hetrf`), but the solution is unique, so
the device `X` and the reference `X` agree to tolerance and `A X` reproduces `B`.
