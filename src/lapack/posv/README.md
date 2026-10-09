# calaman.posv — positive definite solve

`calaman.posv` is LAPACK's `?posv`: solve `A X = B` for a **Hermitian (complex)
or symmetric (real) positive definite** matrix `A`. Like the reference driver,
it is two steps:

1. **Factor** — `A = U^H U` (`Uplo::U`) or `L L^H` (`Uplo::L`) by Cholesky.
2. **Solve** — two triangular solves against the factor.

## What is ours and what is the vendor's

Both halves are the vendor's: `wwr::potrf` and `wwr::potrs`, the cuSOLVER /
hipSOLVER routines WarpWraps exposes backend-neutral for both vendors. So the
reference callees `?potrf`, `?potrf2` and `?potrs` get no calaman module of
their own; this module is the driver that sequences the two and keeps the
reference's "do not solve against a failed factor" rule.

`posv_bufferSize` forwards to `wwr::potrf_bufferSize`, so a size query and the
routine reserve the **same** vendor workspace and cannot drift. The count is in
elements of `T`.

## `info`, and not solving a failed factor

`posv` synchronizes `stream` once, to read the factorization's `info`. The
reference `?posv` solves only when `?potrf` succeeds, so a non-zero `info`
(`info = i > 0`: the leading minor of order `i` is not positive definite) leaves
`B` untouched and returns; the numerical outcome rides `d_info` (device). The
returned `Status` carries only device/solver errors.

## Handles and the stream

Takes a solver handle bound to `stream` — no BLAS handle, since both halves are
solver calls.

## The oracle

`LAPACKE_?posv` on the same Hermitian/symmetric positive definite system: the
device `X` has a backward error at `eps` and agrees with the reference `X` to
tolerance; an indefinite `A` reports the same `info` as the reference and leaves
`B` untouched.
