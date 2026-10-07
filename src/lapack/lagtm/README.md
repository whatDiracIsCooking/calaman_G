# calaman.lagtm

The GPU counterpart of LAPACK's `?lagtm`: `B := alpha * op(A) * X + beta * B`,
where `A` is the `n`-by-`n` general tridiagonal with sub-diagonal `dl` (length
`n-1`), diagonal `d` (length `n`) and super-diagonal `du` (length `n-1`), `op`
is a `Trans` (`N`, `T`, `C`), and `X`, `B` are `n`-by-`nrhs` column-major with
leading dimensions `ldx`, `ldb`. Templated over `float`, `double`,
`wwrFloatComplex` and `wwrDoubleComplex`; `alpha` and `beta` are real
(`ComplexToRealType<T>`), as in the reference.

## Entry point

**`calaman::lagtm<T>(stream, trans, n, nrhs, alpha, d_dl, d_d, d_du, d_x, ldx,
beta, d_b, ldb)`** — the module (`import calaman.lagtm;`, link
`calaman::lagtm`), re-exporting `Trans` and `Status`. One launch, one thread
per element of `B`, no workspace. A leaf: `?lagtm` calls no other routine.

## Scope: alpha and beta are general

The reference honours only `alpha ∈ {-1, 0, 1}` (anything else is taken as 0)
and `beta ∈ {-1, 0, 1}` (anything else is taken as 1). This module computes the
formula for any real `alpha`, `beta` instead — the per-element kernel makes that
free. On the admissible values the two agree, including the reference's
special cases: `beta == 0` overwrites `B` without reading it, and `alpha == 0`
never reads `A` or `X`. The terms accumulate in DLAGTM's order (`beta * b`
first, then the three products left to right).
