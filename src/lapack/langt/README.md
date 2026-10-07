# calaman.langt

The GPU counterpart of LAPACK's `?langt`: the norm of the `n`-by-`n` general
tridiagonal matrix with sub-diagonal `dl` (length `n-1`), diagonal `d`
(length `n`) and super-diagonal `du` (length `n-1`) — `A(i+1,i) = dl_i`,
`A(i,i) = d_i`, `A(i,i+1) = du_i` — selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|dl_i\|, \|d_i\|, \|du_i\|)` |
| `one` | `'1'`/`'O'` | `max_i \|du_{i-1}\| + \|d_i\| + \|dl_i\|` (column sums) |
| `inf` | `'I'` | `max_i \|dl_{i-1}\| + \|d_i\| + \|du_i\|` (row sums) |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum \|dl_i\|^2 + \|d_i\|^2 + \|du_i\|^2)` |

Unlike `calaman.lanst`, the 1- and infinity-norms differ. The result is a real
device scalar (`ComplexToRealType<T>`); `n == 0` writes `0`, as DLANGT does.
Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`.

## Entry point

**`calaman::langt<T>(stream, which, n, d_dl, d_d, d_du, d_result)`** — the
module (`import calaman.langt;`, link `calaman::langt`), re-exporting
`MatrixNorm` and `Status`. One single-block launch, no workspace: each thread
folds a contiguous chunk of indices, then `common/block_reduce.cuh` folds the
partials — `calaman.lanst`'s shape. Its kernel is not reused, because lanst's
diagonal is real and its row and column sums coincide.

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANGT's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN`, as DLANGT's `DISNAN` guard does.
