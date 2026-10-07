# calaman.lantb

The GPU counterpart of LAPACK's `?lantb`: the norm of the `n`-by-`n` upper or
lower triangular band matrix `A` with `k` super- (`Uplo::U`) or sub-diagonals
(`Uplo::L`), held in LAPACK band storage — `A(i,j)` at `AB(k+i-j, j)` for
`Uplo::U`, at `AB(i-j, j)` for `Uplo::L`, `ldab >= k+1`. Only in-matrix band
entries are read; the unused corner of `AB` and any padding rows may hold
anything. With `Diag::U` each diagonal entry counts as `1` and is not read.
Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over the band |
| `one` | `'1'`/`'O'` | the largest column absolute sum |
| `inf` | `'I'` | the largest row absolute sum |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i,j} \|A(i,j)\|^2)` over the band |

The result is a real device scalar (`ComplexToRealType<T>`); `n == 0` writes
`0`. Templated over `float`, `double`, `wwrFloatComplex` and `wwrDoubleComplex`.

## Entry point

**`calaman::lantb<T>(stream, which, uplo, diag, n, k, d_AB, ldab, d_result)`** —
the module (`import calaman.lantb;`, link `calaman::lantb`), re-exporting
`MatrixNorm`, `Uplo`, `Diag` and `Status`.

It owns no device code: it calls `calaman.lantr`'s launcher
(`lapack/lantr/lantr_bridge.h`) with `TriStorage::band` and links that module's
device archive. The band index is inline in `lantr.cu` (`calaman.langb`'s
precedent).

## Scope

As for `calaman.lantr`: the Frobenius norm is a plain sum of squares rather than
DLANTB's scaled `?lassq`, and the max folds propagate a `NaN`.
