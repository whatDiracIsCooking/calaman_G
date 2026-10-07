# calaman.lantp

The GPU counterpart of LAPACK's `?lantp`: the norm of the `n`-by-`n` upper or
lower triangular matrix `A` held in packed storage — the `n(n+1)/2`-element
array `AP` of `calaman.trttp` (the triangle, column by column). With `Diag::U`
each diagonal entry counts as `1` and its `AP` slot is not read. Selected at
runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over the triangle |
| `one` | `'1'`/`'O'` | the largest column absolute sum |
| `inf` | `'I'` | the largest row absolute sum |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i,j} \|A(i,j)\|^2)` over the triangle |

The result is a real device scalar (`ComplexToRealType<T>`); `n == 0` writes
`0`. Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`.

## Entry point

**`calaman::lantp<T>(stream, which, uplo, diag, n, d_AP, d_result)`** — the
module (`import calaman.lantp;`, link `calaman::lantp`), re-exporting
`MatrixNorm`, `Uplo`, `Diag` and `Status`.

It owns no device code: it calls `calaman.lantr`'s launcher
(`lapack/lantr/lantr_bridge.h`) with `TriStorage::packed` and links that
module's device archive — the `calaman.lanhp`/`calaman.lansp` split. The slot
of each entry comes from `calaman::tri_index`'s `packed_index`.

## Scope

As for `calaman.lantr`: the Frobenius norm is a plain sum of squares rather than
DLANTP's scaled `?lassq`, and the max folds propagate a `NaN`.
