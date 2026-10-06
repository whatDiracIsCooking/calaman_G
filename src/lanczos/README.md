# calaman.lanczos

Thick-restart **Lanczos** for the `nev` extreme eigenpairs of a real symmetric
operator known only through a single-vector matrix-vector-product callback.

Not a LAPACK routine, so it is its own module rather than a partition of a
LAPACK-named one, like `calaman.davidson` and `calaman.feast`.

> **Status: in progress.** `lanczos_solve` runs a **single cycle** of `ncv`
> steps and reports `converged` only when that one cycle sufficed
> (`restarts == 0`); `options.max_restarts` is not yet honoured. Thick restart
> and breakdown recovery land in a follow-up issue (milestone "calaman.lanczos").

Real `float`/`double` only (`calaman::real_fp`). Hermitian/complex, block Lanczos
and a tridiagonal eigensolver are out of scope.

## Module

`calaman.lanczos` — one module, four partitions:

| Partition | Contents |
|---|---|
| `:types` | `LanczosWhich`, `LanczosOptions`, `LanczosResult`, `LanczosMatvecFn` |
| `:buffer_size` | `LanczosSlices`, `lanczos_shape_ok`, `make_lanczos_slices` / `lanczos_bufferSize` |
| `:ritz` | `LanczosRitz`, `LanczosRitzSelection`, `lanczos_select`, `lanczos_ritz_extract` / `_select` / `_compact` / `_vectors` |
| `:solve` | `lanczos_solve` |

## The idea

The operator enters as a callback (`LanczosMatvecFn`: `y = A x`, one device
vector). Each cycle runs `ncv` Lanczos steps: a matvec, `alpha_j = v_j^T w`,
full reorthogonalization of `w` against the basis `V` by classical Gram–Schmidt
applied twice (CGS2), `beta_j = ||w||`, and `v_{j+1} = w / beta_j`. The scalars
stay on the device (pointer mode DEVICE); one status read per cycle is the only
host sync. A `beta_j` under `eps * ||T_j||_F` (the leading `(j+1) x (j+1)` block
of `T`, so the guard also sees a restart's arrowhead) is a breakdown, recorded in
`device::LanczosStatus`: the first such step is kept, its coupling in `T` is
written as zero, and no later step of the cycle divides.

The step and arrowhead launchers between them write every entry of `T` a cycle
uses — a fresh cycle's steps cover all of it, and after a restart the arrowhead
covers columns/rows `0..k-1` and steps `k..ncv-1` the rest — so `T` is never
memset.

The projected matrix `T` (`ncv x ncv`) is diagonalised densely with `syevd`:
WarpWraps has no tridiagonal eigensolver, and after a thick restart `T` is an
arrowhead plus a tridiagonal tail anyway. The wanted Ritz pairs are selected by
`LanczosWhich`; their residual estimates `|beta_m s_{m,i}|` cost no matvec.

`:ritz` splits that into four stages so the restart can reuse them:

| Stage | Where | Does |
|---|---|---|
| `lanczos_ritz_extract` | device + one sync | `t` copied to `s`, `syevd` in place on `s` (`theta` ascending); reads `theta`, `S`'s last row, `beta_m = beta[ncv-1]` and the status block back into a host `LanczosRitz` |
| `lanczos_ritz_select` | host | `count` positions per `LanczosWhich`, their estimates, and the flags `estimate <= tol * max(\|theta_i\|, \|\|T\|\|_2)` |
| `lanczos_ritz_compact` | device, no sync | moves the chosen `theta` / `S` columns to the leading slots — the layout `lanczos_arrowhead` reads |
| `lanczos_ritz_vectors` | device, no sync | `X = V S_k`, one `gemm` in host pointer mode |

Selections nest: for `k >= nev`, the `k`-pair selection at one `LanczosWhich`
contains the `nev`-pair one, so a restart can keep extra pairs and still test
convergence on the wanted ones.

### One cycle

`:solve` runs a cycle as three module-internal stages (`calaman::detail`, not
exported) so the restart loop can reuse them:

| Stage | Does | Host syncs |
|---|---|---|
| `lanczos_start` | seeds `s.rng` from `options.seed`; `v_0` = the caller's `start_vector` or a `random_normal` draw, normalised (a zero or non-finite one is `INVALID_VALUE`) | 1 (the norm) |
| `lanczos_extend` | resets the status block, then steps `first..ncv-1`: matvec, CGS2 (`gemv` `V^T w` then `w -= V h`, twice; `alpha_j` is the summed coefficient on `v_j`), `beta_j = nrm2(w)` in device pointer mode, `lanczos_step` | 0 |
| `lanczos_cycle_ritz` | `lanczos_ritz_extract` on the cycle's active dimension | 1, or 2 after an early breakdown |

The matvec callback always runs with the BLAS handle in **host** pointer mode;
the caller's mode is restored on return. Every callback call counts in
`LanczosResult::matvecs`.

**Breakdown inside a cycle.** The status block is read only at the cycle's
sync, so a breakdown at step `j < ncv - 1` still runs the remaining steps (on a
frozen, undivided tail) and they are discarded: `V(:, 0:j+1)` spans an invariant
subspace, its block `T(0:j+1, 0:j+1)` is decoupled, so it is repacked to ld
`j + 1` and re-extracted. Its Ritz pairs are exact (estimate `beta_j` ≈ 0), so
they converge. When that subspace holds fewer than `nev` pairs, the solve
returns them and reports not converged. A breakdown at the last step is the
`ncv == n` case: the whole space, every pair exact. Like any single-vector
Lanczos, an eigenvector absent from the start vector is never found.

Thick restart (Wu–Simon) keeps the `k` wanted Ritz vectors plus the residual
direction, writes the arrowhead into `T`, and continues from step `k + 1` —
`lanczos_extend` with `first = k`.

## Workspace

One caller-provided device buffer, carved by `make_lanczos_slices` through
`calaman::carve_workspace`, so sizing and carving share one code path. The
basis `V` (`n x (ncv + 1)`) dominates; the rest is `O(ncv^2)` plus `n`
generator states for the random start and breakdown-recovery vectors. The
`syevd` workspace is the single scratch region. The shape must satisfy
`nev >= 1` and `2 * nev + 1 <= ncv <= n`.

```cpp
import calaman.lanczos;
using namespace calaman;

std::size_t lwork = 0;
lanczos_bufferSize<double>(solver, n, nev, ncv, &lwork);
// ... allocate d_work (lwork bytes) ...
LanczosSlices<double> s;
make_lanczos_slices<double>(solver, n, nev, ncv, d_work, &s, &lwork);

LanczosResult<double> result;
lanczos_solve<double>(blas, solver, stream, n, nev, ncv, LanczosWhich::smallest, s,
                      matvec, d_eigenvectors, &result);
```

## Files

| File | Role |
|------|------|
| `types.cppm` | `:types` — options, result, end selection, matvec callback |
| `buffer_size.cppm` | `:buffer_size` — slices, workspace layout, sizing |
| `ritz.cppm` | `:ritz` — `syevd` on `T`, end selection, residual estimates, compaction, Ritz vectors |
| `solve.cppm` | `:solve` — `lanczos_solve` and its cycle stages |
| `lanczos_bridge.h` | `device::LanczosStatus` and the kernel-launch declarations |
| `lanczos.cu` | the device library: status reset, step write/normalise + breakdown guard, arrowhead |
| `interface.cppm` | primary interface; re-exports the partitions |
| `CMakeLists.txt` | build configuration |
