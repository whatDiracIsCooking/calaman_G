# calaman.lanst

The GPU counterpart of LAPACK's `?lanst`: the norm of the `n`-by-`n` symmetric
tridiagonal matrix with diagonal `d` (length `n`) and off-diagonal `e` (length
`n-1`), selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max(\|d_i\|, \|e_i\|)` |
| `one` | `'1'`/`'O'` | `max_i \|e_{i-1}\| + \|d_i\| + \|e_i\|` |
| `inf` | `'I'` | the same as `one` — the matrix is symmetric |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum d_i^2 + 2 sum e_i^2)` |

The result is written to a device scalar; `n == 0` writes `0`, as DLANST does.
Templated over `float` and `double`.

## Two entry points

- **`calaman::lanst<T>(stream, which, n, d_d, d_e, d_result)`** — the module
  (`import calaman.lanst;`, link `calaman::lanst`). One single-block launch:
  each thread folds a contiguous chunk of rows, then `common/block_reduce.cuh`
  folds the partials. One block suffices because the input is only `2n-1`
  numbers.
- **`calaman::lanst_max_abs<T>(n, d, e)`** — `lanst.h`, header-only and
  `CLM_HOST_DEVICE`, for a kernel that needs `?lanst('M')` of a sub-block from
  one thread, as `?sterf` and `?steqr` do on each unreduced block. A `.cu`
  `#include`s `"lapack/lanst/lanst.h"` and its device library links
  `calaman::lanst::header` (the `src/` include root, nothing else). The module's
  own `max_abs` kernel calls it per chunk, so the oracle suite exercises it on a
  card; the suite also checks it on the host over sub-ranges.

## Scope

The Frobenius norm is a plain sum of squares rather than DLANST's scaled
`?lassq`, matching `calaman.lange` — correct for well-scaled inputs, a
deliberate simplification. The max folds propagate a `NaN` as DLANST's
`DISNAN` guard does.
