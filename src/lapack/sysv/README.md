# calaman.sysv — symmetric indefinite solve

`calaman.sysv` is LAPACK's `?sysv`: solve `A X = B` for a **symmetric indefinite**
matrix `A`. Like the reference driver, it is two steps:

1. **Factor** — `A = U D U^T` (`Uplo::U`) or `L D L^T` (`Uplo::L`) by the
   Bunch-Kaufman diagonal-pivoting method.
2. **Solve** — with the factor and the pivots.

**Complex is symmetric, not Hermitian** — no conjugation. The Hermitian cousin is
`calaman.hesv`.

## What is ours and what is the vendor's

The factorization half is `wwr::sytrf` — the cuSOLVER / hipSOLVER routine, which
WarpWraps exposes **backend-neutral for both vendors**, so `calaman.sysv` owns no
factorization code. The solve half is `calaman.sytrs`, built on wrapped BLAS (the
vendor ships `sytrf` but no `sytrs`). So this module is just the driver: it sizes
and runs `wwr::sytrf`, then hands the factor to `calaman.sytrs`.

`sysv_bufferSize` forwards to `wwr::sytrf_bufferSize`, so a size query and the
routine reserve the **same** vendor workspace and cannot drift. The count is in
elements of `T` (the `lwork` both take).

## `info`, and not solving a singular factor

`sysv` synchronizes `stream` once, to read the factorization's `info`. The
reference `?sysv` does **not** solve against a singular `D`, so a non-zero `info`
(a singular factor, `info > 0`, or a vendor bad-argument report, `info < 0`)
leaves `B` untouched and returns; the numerical outcome rides `d_info` (device),
as `rank`/`info` ride in `pstrf`. The returned `Status` carries only
device/BLAS/solver errors.

## Handles and the stream

Takes a BLAS handle (for `sytrs`) and a solver handle (for `sytrf`), both bound to
the one `stream` all work is enqueued on — the `expm` shape. The BLAS handle must
be in default (host) pointer mode, since `sytrs`'s BLAS scalars are host
constants.

## The oracle

`LAPACKE_?sysv` on the same symmetric indefinite system: the device `X` and the
reference `X` agree to tolerance, and `A X` reproduces the original `B`.
