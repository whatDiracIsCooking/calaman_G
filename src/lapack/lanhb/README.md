# calaman.lanhb

The GPU counterpart of LAPACK's `?lanhb` (`c`/`z`): the norm of the `n`-by-`n`
complex Hermitian band matrix `A` with `k` super-diagonals, one triangle of
which is held in LAPACK band storage (`calaman.lansb`'s layout: `A(i,j)` at
`AB(k+i-j, j)` for `Uplo::U`, at `AB(i-j, j)` for `Uplo::L`). Of each diagonal
entry only the real part is read (the imaginary part of a Hermitian diagonal is
zero, and ZLANHB ignores whatever is stored there). Selected at runtime by a
`MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|Re A(j,j)\|, \|A(i,j)\|)` over the band |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum |
| `inf` | `'I'` | the same as `one` — the matrix is Hermitian |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum Re A(j,j)^2 + sum_{i != j} \|A(i,j)\|^2)` |

The result is a real device scalar (`ComplexToRealType<T>`); `n == 0` writes
`0`.

## Entry point

**`calaman::lanhb<T>(stream, which, uplo, n, k, d_AB, ldab, d_result)`** — the
module (`import calaman.lanhb;`, link `calaman::lanhb`), re-exporting
`MatrixNorm`, `Uplo` and `Status`.

It owns no device code: it calls `calaman.lansb`'s launcher
(`lapack/lansb/lansb_bridge.h`) with the `hermitian` flag set and links that
module's device archive — the `calaman.lanhp`/`calaman.lansp` split.

## Scope

As for `calaman.lansb`: the Frobenius norm is a plain sum of squares rather than
ZLANHB's scaled `?lassq`, and the max folds propagate a `NaN`.
