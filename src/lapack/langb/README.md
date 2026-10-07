# calaman.langb

The GPU counterpart of LAPACK's `?langb`: the norm of the `n`-by-`n` general
band matrix `A` with `kl` sub-diagonals and `ku` super-diagonals, held in
LAPACK band storage — `A(i,j)` at `AB(ku+i-j, j)`, `ldab >= kl+ku+1`. Only the
in-matrix band entries are read; the unused corners of `AB` and any padding
rows may hold anything. Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over the band |
| `one` | `'1'`/`'O'` | the largest column absolute sum |
| `inf` | `'I'` | the largest row absolute sum |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum \|A(i,j)\|^2)` over the band |

The result is written to a device scalar; `n == 0` writes `0`, as DLANGB does.
Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`; the
norm is always real (`ComplexToRealType<T>`).

## Entry point

**`calaman::langb<T>(stream, which, n, kl, ku, d_AB, ldab, d_result)`** — the
module (`import calaman.langb;`, link `calaman::langb`), re-exporting
`MatrixNorm` and `Status`. Two launches, the `calaman.lansy` shape: one block
per index `j` folds column `j`'s band (row `j`'s, for `inf`) into an
`n`-element scratch, then one block folds the scratch. The scratch is allocated
stream-ordered on `stream`, so the return carries an allocation failure too.

The band index is computed inline in `langb.cu`; there is no shared band
header. Neither `calaman.lange`'s kernels (which read every row of a column)
nor `calaman.lansy`'s (a triangle) fit the band shape, so the kernel is new.

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANGB's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN` from the band, as DLANGB's `DISNAN` guard does.
