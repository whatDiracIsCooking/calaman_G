# calaman.lanhf

The GPU counterpart of LAPACK's `?lanhf` (`c`/`z`): the norm of the `n`-by-`n`
complex Hermitian matrix `A` held in Rectangular Full Packed storage (the
`n(n+1)/2`-element `ARF` of `calaman.trttf`). Of each diagonal entry only the
real part is read (the imaginary part of a Hermitian diagonal is zero, and
ZLANHF ignores whatever is stored there). Selected at runtime by a
`MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|Re A(j,j)\|, \|A(i,j)\|)` |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum |
| `inf` | `'I'` | the same as `one` — the matrix is Hermitian |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum Re A(j,j)^2 + sum_{i != j} \|A(i,j)\|^2)` |

`TRANSR` is a `calaman::Trans`: `N`, or `C` for the conjugate-transposed layout
(any other value returns `InvalidValue`). The result is a real device scalar
(`ComplexToRealType<T>`); `n == 0` writes `0`.

## Entry point

**`calaman::lanhf<T>(stream, which, transr, uplo, n, d_arf, d_result)`** — the
module (`import calaman.lanhf;`, link `calaman::lanhf`), re-exporting
`MatrixNorm`, `Trans`, `Uplo` and `Status`.

It owns no device code: it calls `calaman.lansf`'s launcher
(`lapack/lansf/lansf_bridge.h`) with the `hermitian` flag set and links that
module's device archive — the `calaman.lanhe`/`calaman.lansy` split.

## Scope

As for `calaman.lansf`: the Frobenius norm is a plain sum of squares rather than
ZLANHF's scaled `?lassq`, and the max folds propagate a `NaN`.
