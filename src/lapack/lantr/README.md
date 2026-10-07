# calaman.lantr

The GPU counterpart of LAPACK's `?lantr`: the norm of the `m`-by-`n` upper
(`i <= j`) or lower (`i >= j`) trapezoidal matrix `A`. Only that trapezoid is
read; the opposite triangle and the `lda` padding may hold anything. With
`Diag::U` each diagonal entry counts as `1` and is not read. Selected at runtime
by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max \|A(i,j)\|` over the trapezoid |
| `one` | `'1'`/`'O'` | the largest column absolute sum |
| `inf` | `'I'` | the largest row absolute sum |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_{i,j} \|A(i,j)\|^2)` over the trapezoid |

The result is written to a device scalar; `min(m, n) == 0` writes `0`, as
DLANTR does. Templated over `float`, `double`, `wwrFloatComplex` and
`wwrDoubleComplex`; the norm is always real (`ComplexToRealType<T>`).

## Entry point

**`calaman::lantr<T>(stream, which, uplo, diag, m, n, d_A, lda, d_result)`** —
the module (`import calaman.lantr;`, link `calaman::lantr`), re-exporting
`MatrixNorm`, `Uplo`, `Diag` and `Status`. Two launches, `calaman.lanhs`'s
shape: one block per column (per row, for `inf`) folds that line's share of the
trapezoid into a scratch, then one block folds the scratch. The scratch (`m` or
`n` elements) is allocated stream-ordered on `stream`, so the return carries an
allocation failure too.

The device library is shared with `calaman.lantp` and `calaman.lantb`: the
kernel is templated on the storage layout (`device::TriStorage` in
`lantr_bridge.h` — full, packed via `calaman::tri_index`'s `packed_index`, or
band with an inline index), and the launcher takes it as a runtime switch.

## Scope

The Frobenius norm is a plain sum of squares (`re^2 + im^2` for complex) rather
than DLANTR's scaled `?lassq`, matching `calaman.lange` — correct for
well-scaled inputs, a deliberate simplification. The max folds propagate a
`NaN` from the trapezoid, as DLANTR's `DISNAN` guard does.
