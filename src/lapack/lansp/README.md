# calaman.lansp

The GPU counterpart of LAPACK's `?lansp`: the norm of the `n`-by-`n` symmetric
matrix `A` held in packed storage — the `n(n+1)/2`-element array `AP` of
`calaman.trttp` (the stored triangle, column by column). Selected at runtime by
a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum |
| `inf` | `'I'` | the same as `one` — the matrix is symmetric |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i,j} \|A(i,j)\|^2)` |

The result is written to a device scalar; `n == 0` writes `0`, as DLANSP does.
Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`; the
norm is always real (`ComplexToRealType<T>`). Complex `A` is *symmetric*
(`A = A^T`), as in `CLANSP`/`ZLANSP` — the Hermitian norm is `calaman.lanhp`.

## Entry point

**`calaman::lansp<T>(stream, which, uplo, n, d_ap, d_result)`** — the module
(`import calaman.lansp;`, link `calaman::lansp`), re-exporting `MatrixNorm`,
`Uplo` and `Status`. Two launches, `calaman.lansf`'s shape: one block per
column `j` of the full matrix folds that column's partial into an `n`-element
scratch (an entry of the unstored triangle is read as its mirror), then one
block folds the scratch. Every `AP` slot comes from
`calaman::device::packed_index` (`calaman::tri_index`). The scratch is
allocated stream-ordered on `stream`, so the return carries an allocation
failure too.

The device library (`lansp.cu`) is shared with `calaman.lanhp`: the launcher
takes a `hermitian` flag (a Hermitian diagonal entry contributes only its real
part).

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANSP's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN`, as DLANSP's `DISNAN` guard does.
