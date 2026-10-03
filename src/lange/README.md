# calaman.lange

The GPU counterpart of LAPACK's `?lange`: the norm of a general `m`-by-`n`
column-major matrix, selected at runtime by a `MatrixNorm`:

| `MatrixNorm` | LAPACK char | value |
|---|---|---|
| `max_abs` | `'M'` | `max_ij \|A(i,j)\|` — largest absolute element (not a consistent norm) |
| `one` | `'1'`/`'O'` | the 1-norm, `max_j sum_i \|A(i,j)\|` (max column sum) |
| `inf` | `'I'` | the infinity-norm, `max_i sum_j \|A(i,j)\|` (max row sum) |
| `frobenius` | `'F'`/`'E'` | `sqrt(sum_ij A(i,j)^2)` |

The result is written to a device scalar; an empty matrix (`m` or `n` zero)
writes `0`, as DLANGE returns `0`. Templated over `float` and `double`.

## Shape

Structurally a `calaman.gebal`-flavoured host driver (`interface.cppm`, returns
`calaman::Status`) over a device translation unit (`lange.cu`), sharing a launcher
bridge (`lange_bridge.h`) across the host/device line, with an
explicit-instantiation unit (`instantiations.cpp`). It takes a bare
`wwr::wwrStream_t` — but, unlike the stream-only `columnwise_*` modules, it needs
one scratch buffer for the reduction's intermediate, so it allocates that on the
stream (`wwrMallocAsync`/`wwrFreeAsync`, stream-ordered) and reports any failure
through `Status`, exactly as `gebal` does for its own scratch.

## How the norms are built

Each norm is a **two-stage reduction**, so the module owns almost no kernel of its
own. Stage one reduces `A` to a per-column (or, for `inf`, per-row) intermediate;
stage two folds that intermediate — treated as a one-column matrix — to a single
scalar. Both stages defer to [`calaman.reduce_columns`](../reduce_columns/README.md):

- `max_abs` — per-column `max|·|`, then `max` over columns
- `one` — per-column `sum|·|`, then `max` over columns
- `inf` — per-**row** `sum|·|`, then `max` over rows
- `frobenius` — per-column `sum(·²)`, then `sum` over columns, then `sqrt`
  (one element, through `wwr.extension.parallel_for`, as `columnwise_ell2`)

Only the per-**row** abs-sum is a hand-written kernel: `reduce_columns` folds down
a column (the contiguous direction), and reducing along rows of column-major
storage is the one access pattern no existing module covers. The max folds
propagate a `NaN` from either operand, matching DLANGE's
`VALUE.LT.temp .OR. disnan(temp)`.

## Scope

The Frobenius norm uses a plain sum of squares rather than DLANGE's scaled
`DLASSQ`, matching `calaman.columnwise_ell2` — correct for well-scaled inputs, a
deliberate simplification. The norm of a **complex** matrix is real-valued (a
`T → real` reduction), a deliberate later extension, as `calaman.diff_norm` notes
for the same reason; `reduce_columns` is already generic over that differing value
type, so only this wrapper is type-fixed.
