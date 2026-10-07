# calaman.lansy

The GPU counterpart of LAPACK's `?lansy`: the norm of the `n`-by-`n` symmetric
matrix `A`, read from **only** the triangle `uplo` names — the other triangle is
never addressed, so it may hold anything. Selected at runtime by a
`MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over the stored triangle |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum of the full matrix |
| `inf` | `'I'` | the same as `one` — the matrix is symmetric |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum \|A(j,j)\|^2 + 2 sum_{i != j, stored} \|A(i,j)\|^2)` |

The result is written to a device scalar; `n == 0` writes `0`, as DLANSY does.
Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`; the
norm is always real (`ComplexToRealType<T>`). Complex `A` is *symmetric*
(`A = A^T`), as in `CLANSY`/`ZLANSY` — the Hermitian norm is `calaman.lanhe`.

## Entry point

**`calaman::lansy<T>(stream, which, uplo, n, d_A, lda, d_result)`** — the module
(`import calaman.lansy;`, link `calaman::lansy`), re-exporting `MatrixNorm`,
`Uplo` and `Status`. Two launches: one block per column folds that column's
partial (for `one`/`inf`, the full row sum — the stored part of column `j` plus
the stored part of row `j`) into an `n`-element scratch, then one block folds
the scratch. Like `calaman.lange`, the scratch is allocated stream-ordered on
`stream`, so the return carries an allocation failure too.

The device library (`lansy.cu`) is shared with `calaman.lanhe`: the launcher
takes a `hermitian` flag, the one difference between the two routines (a
Hermitian diagonal entry contributes only its real part).

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANSY's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN` from the stored triangle, as DLANSY's `DISNAN` guard does.
