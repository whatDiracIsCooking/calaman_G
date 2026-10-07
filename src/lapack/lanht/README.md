# calaman.lanht

The GPU counterpart of LAPACK's `?lanht` (`c`/`z`): the norm of the `n`-by-`n`
complex Hermitian tridiagonal matrix with **real** diagonal `d` (length `n`) and
complex off-diagonal `e` (length `n-1`), selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|d_i\|, \|e_i\|)` |
| `one` | `'1'`/`'O'` | `max_i \|e_{i-1}\| + \|d_i\| + \|e_i\|` |
| `inf` | `'I'` | the same as `one` — the matrix is Hermitian |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum d_i^2 + 2 sum \|e_i\|^2)` |

`|e_i|` is the complex modulus. The result is a real device scalar
(`ComplexToRealType<T>`); `n == 0` writes `0`, as ZLANHT does. Templated over
`wwrFloatComplex` and `wwrDoubleComplex`.

## Entry point

**`calaman::lanht<T>(stream, which, n, d_d, d_e, d_result)`** — the module
(`import calaman.lanht;`, link `calaman::lanht`), re-exporting `MatrixNorm` and
`Status`.

It owns no device code: it calls `calaman.lanst`'s launcher
(`lapack/lanst/lanst_bridge.h`), which is templated on the off-diagonal type,
and links that module's device archive. One single-block launch, no workspace.

## Scope

As for `calaman.lanst`: the Frobenius norm is a plain sum of squares rather
than ZLANHT's scaled `?lassq` (real and complex), and the max folds propagate a
`NaN`.
