# calaman.davidson

Block **Davidson** solver for the lowest `n_roots` eigenpairs of a symmetric
operator known only through a matrix-vector-product callback.

Not a LAPACK routine (LAPACK ships no Davidson), so it is its own module rather
than a partition of a LAPACK-named one, like `calaman.feast` and `calaman.expm`.

> **Status: skeleton.** `:buffer_size` (the device workspace layout and its
> sizing) is complete and tested. `:solve` pins the public interface — the
> callback shapes, the `DavidsonOptions`/`DavidsonResult` types, and the
> `davidson_solve` signature — but the iteration is **not yet implemented**:
> `davidson_solve` returns `WWRBLAS_STATUS_NOT_SUPPORTED`. The solve body lands
> in a follow-up PR.

## Module

`calaman.davidson` — one module, two partitions:

| Partition | Contents |
|---|---|
| `:buffer_size` | `DavidsonSlices`, the single-buffer workspace layout, `make_davidson_slices` / `davidson_bufferSize` |
| `:solve` | `DavidsonOptions`, `DavidsonResult`, the three callbacks, `davidson_solve` |

## The idea

The operator enters as a caller callback (`DavidsonSigmaFn`: `sigma(B) -> A B`,
block-in block-out on device buffers), so the module knows nothing about what the
operator *is*. A second callback (`DavidsonPreconditionFn`) turns a residual
block into a correction block (the diagonal Davidson correction
`delta = r / (theta - diag)` is the usual choice, but unknown here). An optional
third callback (`DavidsonMetricFn`) supplies an SPD metric `M`, selecting a
generalized subspace problem `(V^T M Sigma_V) c = e (V^T M V) c` via `sygvd` for
an operator that is self-adjoint only in the `M`-inner-product.

Per `solve` (once implemented):

1. **Subspace expansion.** The orthonormal guess seeds `V`; each iteration calls
   `sigma` once, on only the newly appended columns (`V`, `Sigma_V` grow in
   lockstep), so the expensive operator is never reapplied to a direction already
   in the subspace.
2. **Rayleigh–Ritz.** `H = V^T Sigma_V`, diagonalized by `syevd` (Euclidean) or
   the generalized `sygvd` (metric). Ritz vectors `X = V S_k` and their image
   `A X = Sigma_V S_k` come from the same rotation — no second `sigma`.
3. **Residuals and locking.** `R = A X - X diag(theta)`; a converged root is
   locked (its correction computed but discarded).
4. **Collapse.** Before expanding past `max_subspace`, `V`/`Sigma_V` reset to the
   current Ritz pairs; `make_davidson_slices` requires `max_subspace >= 2*n_roots`
   so a post-collapse subspace always has room for a full new block.
5. **Re-orthogonalization.** Twice-modified Gram–Schmidt of the corrections
   against the retained `V`, then against each other; a candidate with no
   component outside the subspace is dropped.

## Workspace

One caller-provided device buffer, carved by `make_davidson_slices` through
`calaman::WorkspaceLayout` (the feast idiom: one function both sizes the layout
and hands out the pointers, so they cannot drift). The basis `V` and its image
`Sigma_V` dominate (each `n x max_subspace`); the rest is `O(n·n_roots)` or
`O(max_subspace²)`. The `syevd` and `sygvd` workspaces never overlap in time, so
they share one scratch region. The metric regions (`M V`, `V^T M V`, the metric
image of the residual block) are carved only `with_metric`.

```cpp
import calaman.davidson;
import wwr.blas;        // wwrblasHandle_t
import wwr.solver;      // wwrsolverDnHandle_t
import wwr.runtime_api; // wwrStream_t
using namespace calaman;

std::size_t lwork = 0;
davidson_bufferSize<double>(cusolver, n, n_roots, max_subspace, /*with_metric=*/false, &lwork);
// ... allocate d_work (lwork bytes); fill d_guess (n x guess_count, ORTHONORMAL) ...

DavidsonSlices<double> s;
make_davidson_slices<double>(cusolver, n, n_roots, max_subspace, false, d_work, &s, &lwork);

DavidsonResult<double> result;
davidson_solve<double>(cublas, cusolver, stream, n, n_roots, max_subspace,
                       d_guess, guess_count, s, sigma, precondition,
                       d_eigenvectors, &result);   // returns NOT_SUPPORTED today
```

Both handles must already be set to `stream`.

## Files

| File | Role |
|------|------|
| `buffer_size.cppm` | `:buffer_size` — slices, workspace layout, sizing |
| `solve.cppm` | `:solve` — options, result, callbacks, `davidson_solve` |
| `interface.cppm` | primary interface; re-exports the partitions |
| `CMakeLists.txt` | build configuration |
