# calaman.lanhs

The GPU counterpart of LAPACK's `?lanhs`: the norm of the `n`-by-`n` upper
Hessenberg matrix `A`, read from **only** rows `i <= j+1` of each column `j` —
entries below the subdiagonal are never addressed, so they may hold anything.
Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over `i <= j+1` |
| `one` | `'1'`/`'O'` | `max_j sum_{i <= j+1} \|A(i,j)\|` (column sums) |
| `inf` | `'I'` | `max_i sum_{j >= i-1} \|A(i,j)\|` (row sums) |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i <= j+1} \|A(i,j)\|^2)` |

The result is a real device scalar (`ComplexToRealType<T>`); `n == 0` writes
`0`, as DLANHS does. Templated over `float`, `double`, `wwrFloatComplex` and
`wwrDoubleComplex`.

## Entry point

**`calaman::lanhs<T>(stream, which, n, d_A, lda, d_result)`** — the module
(`import calaman.lanhs;`, link `calaman::lanhs`), re-exporting `MatrixNorm` and
`Status`. Two launches, `calaman.lansy`'s shape: one block per column (per row,
for `inf`) folds its partial into an `n`-element scratch through
`common/block_reduce.cuh`, then one block folds the scratch. The scratch is
allocated stream-ordered on `stream`, so the return carries an allocation
failure too.

`calaman.reduce_columns` (what `calaman.lange` builds on) is not used: its
pre-transform sees an element's value but not its row, so it cannot mask the
unreferenced part, and the infinity norm folds along rows.

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANHS's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN` from the referenced part, as DLANHS's `DISNAN` guard does.
