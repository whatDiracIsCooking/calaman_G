# calaman.ritz

The Ritz-pair bookkeeping every projection eigensolver in `src/` repeats:
pick the wanted pairs, classify them as converged or not, and rotate the Ritz
vectors out of the basis. Header-only templates over `real_fp`.

```cpp
enum class RitzWhich { smallest, largest, both_ends, largest_magnitude };

template <real_fp T>
struct RitzSelection {
  std::vector<int> index;       // source positions, if the caller records them
  std::vector<T> values, residuals;
  std::vector<bool> converged;
  int converged_count = 0;
  bool all_converged() const;   // false when empty
};

template <real_fp T>
std::vector<int> ritz_select(RitzWhich which, std::span<const T> values,
                             int count);

template <real_fp T>
RitzSelection<T> classify_ritz(std::span<const T> values,
                               std::span<const T> residuals,
                               T tolerance, T scale);

template <real_fp T>
Status ritz_rotate(wwrblasHandle_t blas, int n, int k, int count,
                   const T *b, int ldb, const T *s, int lds, T *c, int ldc);
```

| Routine | Where | Does |
|---|---|---|
| `ritz_select` | host | the `count` positions, ascending, of the ascending `values` (`both_ends`: `ceil(count/2)` top, `floor(count/2)` bottom; `largest_magnitude`: the `count` largest `\|value\|`, merged inward from both ends, ties to the top; every other end sees only `values.size()`); empty unless `1 <= count <= values.size()` |
| `classify_ritz` | host | `converged[i] = residuals[i] <= tolerance * max(\|values[i]\|, scale)`; `index` left empty. A length mismatch returns an empty selection, and an empty selection is **not** `all_converged()`, so a sizing bug cannot end a solve as converged |
| `ritz_rotate` | device, no sync | `C = B(:, 0:k) * S(0:k, 0:count)`, one `gemm` in host pointer mode |

## The rules

- **Pointers and leading dimensions, never a solver's slices struct.** Each
  solver owns its workspace carve (the `workspace` skill); a shared routine
  taking `LanczosSlices`/`DavidsonSlices`/`FeastSlices` would force their
  layouts to agree. `lds` is explicit so a projected matrix stored with a
  larger leading dimension (davidson's `max_subspace`) passes as is.
- **`ritz_rotate` scopes host pointer mode itself** and restores the caller's
  on return. Inside a caller that already holds a `ScopedPointerMode` in host
  mode, the nested guard costs a get/set pair on the handle and no sync.
- **Selections nest.** For `count2 >= count1` at one `RitzWhich`, the
  `count2` selection contains the `count1` one, so a restart can keep extra
  pairs and still test convergence on the wanted ones. `largest_magnitude`
  keeps this because its inward merge only extends a smaller selection, and a
  `|value|` tie always breaks the same way.
- **One convergence predicate.** `scale` is the 2-norm of the solver's
  projected matrix, `max |theta|` over its subspace spectrum: lanczos passes
  `||T||_2`, davidson `||H||_2` from the `syevd`/`sygvd` it already runs. So
  one `tolerance` means the same relative accuracy in both. feast's backward
  error is the other meaning, and stays its own
  ([`docs/architecture.md` §8](../../docs/architecture.md)).

## Adopters

| Module | Uses |
|---|---|
| `calaman.lanczos` | `LanczosWhich` aliases `RitzWhich`; `lanczos_ritz_select` = `ritz_select` + `classify_ritz`, as is the true-residual pass; `lanczos_ritz_vectors` = `ritz_rotate` |
| `calaman.davidson` | `classify_ritz` at scale `\|\|H\|\|_2` (its `RitzSelection` gives `Converged`, `max_residual_norm` and the locked roots); `X = V S_k` and `A X = Sigma_V S_k` = two `ritz_rotate`s with `lds = max_subspace`. Selection stays the leading `n_roots` columns, no `ritz_select` |
| `calaman.feast` | `X = basis Y` and `A X = W Y` = two `ritz_rotate`s in its Rayleigh-Ritz partition (`rayleigh_ritz.cppm`). Selection is its own interval test (`feast_select`) and convergence its backward error, so no `ritz_select` or `classify_ritz` |
