# calaman.lansb

The GPU counterpart of LAPACK's `?lansb`: the norm of the `n`-by-`n` symmetric
band matrix `A` with `k` super-diagonals (equivalently, `k` sub-diagonals), one
triangle of which is held in LAPACK band storage — `A(i,j)` at `AB(k+i-j, j)`
for `Uplo::U`, at `AB(i-j, j)` for `Uplo::L`, `ldab >= k+1`. Only the in-matrix
band entries of the stored triangle are read; the unused corner of `AB` and any
padding rows may hold anything. Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over the band |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum |
| `inf` | `'I'` | the same as `one` — the matrix is symmetric |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i,j} \|A(i,j)\|^2)` over the band |

The result is written to a device scalar; `n == 0` writes `0`, as DLANSB does.
Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`; the
norm is always real (`ComplexToRealType<T>`). Complex `A` is *symmetric*
(`A = A^T`), as in `CLANSB`/`ZLANSB` — the Hermitian norm is `calaman.lanhb`.

## Entry point

**`calaman::lansb<T>(stream, which, uplo, n, k, d_AB, ldab, d_result)`** — the
module (`import calaman.lansb;`, link `calaman::lansb`), re-exporting
`MatrixNorm`, `Uplo` and `Status`. Two launches, `calaman.lansp`'s shape: one
block per column `j` of the full matrix folds the band of that column (rows
`max(0, j-k)` to `min(n-1, j+k)`) into an `n`-element scratch — an entry of the
unstored triangle is read as its mirror — then one block folds the scratch. The
scratch is allocated stream-ordered on `stream`, so the return carries an
allocation failure too.

The band index is computed inline in `lansb.cu` (`calaman.langb`'s precedent;
there is no shared band header). The device library is shared with
`calaman.lanhb`: the launcher takes a `hermitian` flag (a Hermitian diagonal
entry contributes only its real part).

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANSB's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN` from the band, as DLANSB's `DISNAN` guard does.
