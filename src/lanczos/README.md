# calaman.lanczos

Thick-restart **Lanczos** for the `nev` extreme eigenpairs of a real symmetric
operator known only through a single-vector matrix-vector-product callback.

Not a LAPACK routine, so it is its own module rather than a partition of a
LAPACK-named one, like `calaman.davidson` and `calaman.feast`.

> **Status: complete.** Thick restart (Wu–Simon), breakdown recovery, the
> convergence loop under `max_restarts` / `fail_on_non_convergence`, and an
> opt-in true-residual pass. Single-vector: see the repeated-eigenvalue
> limitation below.

Real `float`/`double` only (`calaman::real_fp`). Hermitian/complex, block Lanczos
and a tridiagonal eigensolver are out of scope.

## Module

`calaman.lanczos` — one module, four partitions:

| Partition | Contents |
|---|---|
| `:types` | `LanczosWhich`, `LanczosOptions`, `LanczosResult`, `LanczosMatvecFn` |
| `:buffer_size` | `LanczosSlices`, `lanczos_shape_ok`, `lanczos_restart_keep`, `make_lanczos_slices` / `lanczos_bufferSize` |
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

`:solve` runs a cycle as module-internal stages (`calaman::detail`, not
exported) that the restart loop reuses:

| Stage | Does | Host syncs |
|---|---|---|
| `lanczos_start` | seeds `s.rng` from `options.seed`; `v_0` = the caller's `start_vector` or a `random_normal` draw, normalised (a zero or non-finite one is `INVALID_VALUE`) | 1 (the norm) |
| `lanczos_extend` | resets the status block, then steps `first..ncv-1`: matvec, CGS2 (`gemv` `V^T w` then `w -= V h`, twice; `alpha_j` is the summed coefficient on `v_j`), `beta_j = nrm2(w)` in device pointer mode, `lanczos_step` | 0 |
| `lanczos_cycle_ritz` | `lanczos_ritz_extract`; after an early breakdown, recovery and a re-extract | 1, plus 2 per recovery |
| `lanczos_restart_basis` | the thick restart below; run every cycle, so the wanted vectors are columns of `V` whether or not the loop goes on | 0, or 1 after a last-step breakdown |

The matvec callback always runs with the BLAS handle in **host** pointer mode;
the caller's mode is restored on return. Every callback call counts in
`LanczosResult::matvecs`.

**Breakdown recovery.** The status block is read only at the cycle's sync, so
a breakdown at step `j < ncv - 1` still runs the remaining steps (on a frozen,
undivided tail) and they are discarded: `V(:, 0:j+1)` spans an invariant
subspace and `T`'s coupling at `(j+1, j)` is already zero. `lanczos_inject`
draws a fresh `random_normal` vector from `s.rng`, CGS2-orthogonalises it
against `V(:, 0:j+1)`, normalises it into `V(:, j+1)`, and `lanczos_extend`
re-runs steps `j+1..ncv-1` with zero coupling — so `T` stays ld `ncv` and a
breakdown never shrinks the cycle. Each recovery costs the discarded tail's
matvecs (an operator that breaks down at every step costs `O(ncv^2)` per
cycle). A breakdown at the **last** step (the `ncv == n` case) makes every
estimate tiny; should the loop still go on, the restart injects a fresh `V(:, k)`
with a zero arrowhead coupling instead of using the void `v_ncv`.

### Thick restart and the convergence loop

After each cycle the `nev` wanted pairs are tested on their estimates. To
restart, the `k = lanczos_restart_keep(nev, ncv) = nev + (ncv - nev) / 2` pairs
at the same `LanczosWhich` are kept (`nev <= k <= ncv - 1` under the shape
contract; selections nest, so the wanted pairs are among them): `S` is compacted,
`V S_k` is staged in `s.keep` and copied over `V(:, 0:k)`, the residual
direction `v_ncv` becomes `V(:, k)`, and `lanczos_arrowhead` writes `T`'s
columns/rows `0..k-1` (`theta_i` on the diagonal, `beta_m s_{m,i}` in row `k`).
`lanczos_extend(first = k)` then runs the next `ncv - k` steps; the residual
estimates `|beta_m s_{m,i}|` stay valid across restarts.

The loop stops when every wanted estimate passes, or after
`options.max_restarts` restarts (0: one cycle). `LanczosResult` reports
`restarts` and `matvecs` (`ncv + restarts * (ncv - k)` without breakdowns or a
residual check). A non-converged solve still fills `eigenvalues` and the
vectors, and returns `INTERNAL_ERROR` only when `fail_on_non_convergence`
(the `DavidsonOptions` contract).

**True-residual pass** (`options.verify_residuals`, default off). Once the
estimates pass, each wanted `x_i` is checked by `||A x_i - theta_i x_i||_2`
against the same bound — `nev` matvecs into the spare `V(:, ncv)` and one sync;
a miss keeps restarting. Off by default because the estimate equals the true
residual up to rounding under full reorthogonalization, and a tolerance near
`eps` (e.g. the `float` default `1e-8`) that the estimates can reach the true
residual cannot.

### Limitation: multiple eigenvalues

Single-vector Lanczos sees a multiple eigenvalue's eigenspace only along the
start vector's projection onto it, so **in exact arithmetic it returns one copy
of each multiple eigenvalue** and fills the rest of the selection with the next
distinct ones. In floating point, rounding seeds the missing directions and
further copies may surface after more restarts — late, and not reliably (a
breakdown recovery also injects them). Do not rely on getting the full
multiplicity; a block method, or deflating the found vectors and re-solving, is
the remedy. For the same reason an eigenvector absent from the start vector is
found only if a breakdown recovery or rounding brings it in.

## Workspace

One caller-provided device buffer, carved by `make_lanczos_slices` through
`calaman::carve_workspace`, so sizing and carving share one code path. The
basis `V` (`n x (ncv + 1)`) and the restart's staging block `keep` (`n x k`)
dominate; the rest is `O(ncv^2)` plus `n` generator states for the random start and breakdown-recovery vectors. The
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
| `solve.cppm` | `:solve` — `lanczos_solve`, its cycle stages, the thick restart and breakdown recovery |
| `lanczos_bridge.h` | `device::LanczosStatus` and the kernel-launch declarations |
| `lanczos.cu` | the device library: status reset, step write/normalise + breakdown guard, arrowhead |
| `interface.cppm` | primary interface; re-exports the partitions |
| `CMakeLists.txt` | build configuration |
