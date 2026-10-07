# calaman.lanhe

The GPU counterpart of LAPACK's `?lanhe` (`c`/`z`): the norm of the `n`-by-`n`
complex Hermitian matrix `A`, read from **only** the triangle `uplo` names — the
other triangle is never addressed. Of each diagonal entry only the real part is
read (the imaginary part of a Hermitian diagonal is zero, and ZLANHE ignores
whatever is stored there). Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|Re A(j,j)\|, \|A(i,j)\|)` over the stored triangle |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum of the full matrix |
| `inf` | `'I'` | the same as `one` — the matrix is Hermitian |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum Re A(j,j)^2 + 2 sum_{i != j, stored} \|A(i,j)\|^2)` |

The result is a real device scalar (`ComplexToRealType<T>`); `n == 0` writes
`0`. Templated over `wwrFloatComplex` and `wwrDoubleComplex`.

## Entry point

**`calaman::lanhe<T>(stream, which, uplo, n, d_A, lda, d_result)`** — the module
(`import calaman.lanhe;`, link `calaman::lanhe`), re-exporting `MatrixNorm`,
`Uplo` and `Status`.

It owns no device code: it calls `calaman.lansy`'s launcher
(`lapack/lansy/lansy_bridge.h`) with the `hermitian` flag set and links that
module's device archive. For these norms the diagonal is the only difference
between `?lansy` and `?lanhe`, so one kernel serves both.

## Scope

As for `calaman.lansy`: the Frobenius norm is a plain sum of squares rather than
ZLANHE's scaled `?lassq`, and the max folds propagate a `NaN` from the stored
triangle.
