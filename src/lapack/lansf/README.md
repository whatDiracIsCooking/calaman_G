# calaman.lansf

The GPU counterpart of LAPACK's `?lansf` (`s`/`d`): the norm of the `n`-by-`n`
real symmetric matrix `A` held in Rectangular Full Packed storage — the
`n(n+1)/2`-element array `ARF` of `calaman.trttf` (layout:
`src/lapack/trttf/README.md`). Selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` |
| `one` | `'1'`/`'O'` | the largest column (= row) absolute sum |
| `inf` | `'I'` | the same as `one` — the matrix is symmetric |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i,j} A(i,j)^2)` |

`TRANSR` is a `calaman::Trans`: `N`, or `T` for the transposed layout (any other
value returns `InvalidValue`). The result is written to a device scalar;
`n == 0` writes `0`, as DLANSF does.

## Entry point

**`calaman::lansf<T>(stream, which, transr, uplo, n, d_arf, d_result)`** — the
module (`import calaman.lansf;`, link `calaman::lansf`), re-exporting
`MatrixNorm`, `Trans`, `Uplo` and `Status`. Two launches, `calaman.lansy`'s
shape: one block per column `j` of the full matrix folds that column's partial
into an `n`-element scratch (an entry of the unstored triangle is read as its
mirror), then one block folds the scratch. Every `ARF` slot comes from
`calaman::device::rfp_index` (`calaman::tri_index`), which covers all eight
`TRANSR` x `UPLO` x n-parity cases. The scratch is allocated stream-ordered on
`stream`, so the return carries an allocation failure too.

The device library (`lansf.cu`) is shared with `calaman.lanhf`: the launcher
takes a `hermitian` flag (a Hermitian diagonal entry contributes only its real
part).

## Scope

The Frobenius norm is a plain sum of squares rather than DLANSF's scaled
`?lassq`, matching `calaman.lange` — correct for well-scaled inputs, a
deliberate simplification. The max folds propagate a `NaN`, as DLANSF's
`DISNAN` guard does.
