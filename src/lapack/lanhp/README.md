# calaman.lanhp

The GPU counterpart of LAPACK's `?lanhp` (`c`/`z`): the norm of the `n`-by-`n`
complex Hermitian matrix `A` held in packed storage (the `n(n+1)/2`-element
`AP` of `calaman.trttp`). Of each diagonal entry only the real part is read
(the imaginary part of a Hermitian diagonal is zero, and ZLANHP ignores
whatever is stored there). Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|Re A(j,j)\|, \|A(i,j)\|)` |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum |
| `inf` | `'I'` | the same as `one` — the matrix is Hermitian |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum Re A(j,j)^2 + sum_{i != j} \|A(i,j)\|^2)` |

The result is a real device scalar (`ComplexToRealType<T>`); `n == 0` writes
`0`.

## Entry point

**`calaman::lanhp<T>(stream, which, uplo, n, d_ap, d_result)`** — the module
(`import calaman.lanhp;`, link `calaman::lanhp`), re-exporting `MatrixNorm`,
`Uplo` and `Status`.

It owns no device code: it calls `calaman.lansp`'s launcher
(`lapack/lansp/lansp_bridge.h`) with the `hermitian` flag set and links that
module's device archive — the `calaman.lanhf`/`calaman.lansf` split.

## Scope

As for `calaman.lansp`: the Frobenius norm is a plain sum of squares rather than
ZLANHP's scaled `?lassq`, and the max folds propagate a `NaN`.
