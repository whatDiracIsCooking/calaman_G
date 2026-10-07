# calaman.feast

The **FEAST** eigensolver for real symmetric matrices, after Polizzi,
*Density-matrix-based algorithm for solving eigenvalue problems*, Phys. Rev. B
79, 115112 (2009), arXiv:0901.2665.

Finds the eigenpairs of `A x = λ x` with `λ ∈ [Emin, Emax]` — those and only
those, however much of the spectrum lies elsewhere.

Not a LAPACK routine (LAPACK ships no FEAST), so it is its own module rather than
a partition of a LAPACK-named one, like `calaman.expm`.

## Module

`calaman.feast` — one module, six partitions:

| Partition | Contents |
|---|---|
| `:quadrature` | Gauss–Legendre nodes and weights, `N = 4, 8` |
| `:compute_quadrature` | the contour `Z_e, w_e` for an interval, and the rational filter `ρ` it defines |
| `:buffer_size` | the driver's `O(n·m0)` workspace, the dense entry point's single-buffer layout, and its sizing |
| `:resolvent` | the `feast_resolvent` concept, and `DenseResolvent`: `ρ(A) Y` as `Ne` shifted solves in batched BLAS calls |
| `:rayleigh_ritz` | QR, projection, `syevd`, selection, residuals |
| `:driver` | the iteration |

Only `feast`, `feast_bufferSize`, `FeastOptions`, `FeastInfo`,
`FeastStopReason`, `feast_rational_filter` and the `feast_resolvent` concept
(with `linear_operator`, re-exported from `calaman.linear_operator`) are
exported. The rest — the per-iteration steps, `DenseResolvent`, `FeastSlices`,
the contour and quadrature tables — are
module-internal: reachable from the header-only `feast` template when an
importer instantiates it, but not nameable by that importer.

## The idea

The spectral projector onto the eigenvectors with eigenvalues inside a contour
`C` is a contour integral of the resolvent,

```
P = (1 / 2πi) ∮_C (zI − A)⁻¹ dz.
```

Take `C` to be the circle through `Emin` and `Emax`. For real symmetric `A` the
lower half of the circle contributes the conjugate of the upper half, and
Gauss–Legendre on the upper half gives

```
P ≈ ρ(A) = Σ_e Re[ w_e (Z_e I − A)⁻¹ ],   Z_e = c + r e^{iθ_e},   w_e = (ω_e/2) r e^{iθ_e}
```

with centre `c = (Emin + Emax)/2`, radius `r = (Emax − Emin)/2`. On an
eigenvector, `ρ(A)` multiplies by the rational function
`ρ(λ) = Σ_e Re[w_e / (Z_e − λ)]`: exactly 1 at the centre, ½ at either end, and
small outside — two radii from the centre it is 2.1·10⁻³ with 4 nodes and
1.6·10⁻⁵ with 8.

FEAST is subspace iteration with `ρ(A)`. Applied to `m0 ≥ m` vectors, it
amplifies the `m` wanted eigenvectors over all the others, and Rayleigh–Ritz on
the result extracts them.

## One iteration

```
1. basis = ρ(A) Q = Σ_e Re[ w_e (Z_e I − A)⁻¹ Q ]      Ne shifted solves
2. basis ← Q-factor of qr(basis)
3. H = basisᵀ A basis;  H = Y diag(ritz) Yᵀ             op.apply, gemm, syevd
4. rotate the pairs with ritz ∈ [Emin, Emax] to the front
5. Q = basis Y;  residuals of the pairs inside
6. stop once m repeats and every residual is below tol
```

## Resolvent models

The driver (`feast_iterate`) never reads `A`: it is generic over a
`feast_resolvent` model, a `linear_operator` (`Y = A X`, used by Rayleigh–Ritz)
that also has `prepare(stream, contour)`, once per solve, and
`filter(stream, contour, k, Y, out)`, `out = ρ(A) Y`, once per iteration. Each
model carves its own workspace; the driver's `FeastSlices` is only the
`O(n·m0)` Rayleigh–Ritz part. `feast` and `feast_bufferSize` are the dense
model, `DenseResolvent`, behind the original signatures. The plan for
matrix-free models is `docs/architecture.md` §9.

## One stream

The `Ne` shifted systems are independent, and each stage over them is a single
call: `getrfBatched` factors all `Ne` resolvents `Z_e I − A` once per solve —
neither `Z_e` nor `A` changes between iterations — and `getrsBatched` solves all
of them against `Q` once per iteration. Both are reached through the
backend-neutral wrappers in `wwr.wrappers.blas`, so the one-stream batched path
is the same under CUDA and HIP. The resolvents are complex symmetric, not
Hermitian, so the factorization is a general LU; it is never singular, since
every `Z_e` has a positive imaginary part and `A`'s spectrum is real.

Around the library calls sit a few small kernels (`feast.cu`): build the
resolvents from `A`'s stored triangle, broadcast the real `Q` into the `Ne`
complex right-hand sides, reduce the `Ne` solutions to `Re Σ w_e X_e`, select and
rotate the in-interval Ritz pairs, and compute their residuals. They are raw
`<<<>>>` kernels like `gebal.cu`, written **backend-neutral**: the complex
components go through `complex.h`'s `wwrCreal*`/`make_wwr*Complex` accessors
(never `.x`/`.y`, which `hipComplex` lacks), and the block reductions are
shared-memory tree-reduces rather than warp shuffles or the float-bits
`atomicMax` trick — both CUDA-only spellings. Everything the host needs to decide
an iteration's outcome — the count, the largest residual, the QR and `syevd`
info — lands in one small status block, so the loop synchronizes once per
iteration. The `Ne` LU infos are read once, by `DenseResolvent::prepare`, which
synchronizes once per solve.

`getrfBatched` is aimed at many small matrices. With only `Ne ≤ 8` large ones it
is not the fastest LU available, but the factorization happens once per solve,
and the batched solve that runs every iteration keeps all `Ne` systems on one
stream in one call.

## Stopping

| `info.reason` | Meaning |
|---|---|
| `Converged` | `m` is the same as last iteration and every residual is below `tol` |
| `MaxIterations` | the budget ran out |
| `SubspaceTooSmall` | every Ritz value landed inside the interval after the first iteration — raise `m0` |
| `NumericalFailure` | a factorization reported a breakdown, or a kernel failed |

`FeastInfo` derives from `calaman.iterative`'s `IterationInfo`, so
`converged(info)` reads `reason == Converged`. Non-convergence is an outcome,
not an error (the project-wide rule, `src/iterative/README.md`): every row but
`NumericalFailure` comes back with a successful `Status`.

The residual is each pair's normwise backward error,
`‖Ax − λx‖₁ / ((‖A‖₁ + |λ|) ‖x‖₁)`, so a tolerance on it means the same whatever
the scale of `A`. Defaults: `1e-5` in float, `1e-12` in double. It is not the
relative test lanczos and davidson share through `classify_ritz`; the three are
compared in [`docs/architecture.md` §8](../../docs/architecture.md).

Convergence needs `m` seen twice running, so a solve takes at least two
iterations. That is deliberate: from a random start, a Ritz value that has not
converged yet can sit inside the interval for an iteration or two, and without
the repeat it could end the iteration with the wrong count.

Only the leading `m` pairs are converged. The other `m0 − m` Ritz values in
`d_lambda` are whatever the rest of the subspace holds, with no accuracy
guarantee.

## Where this departs from the paper

1. **QR before Rayleigh–Ritz, not the reduced generalized problem.** The paper
   solves `H y = λ (basisᵀ basis) y`. But `basisᵀ basis` has eigenvalues going
   like `ρ(λ)²` for the least-amplified directions the subspace carries, so it
   grows more ill-conditioned the sharper the filter or the larger `m0`.
   Orthonormalizing first and calling `syevd` on `H` never squares that.
2. **A residual criterion, not the trace of the eigenvalues.** A stable trace
   says the eigenvalues stopped moving, not that the pairs are right; the
   backward error says the latter directly.
3. **The sign of the weights.** It does not affect FEAST — Rayleigh–Ritz sees
   only the span — so it is chosen to make `ρ` the projector's approximation
   (+1 inside) rather than its negative, which lets the tests check the device
   filter against `ρ` directly.

## Status type

Every entry point returns `calaman::Status`, like `calaman.expm`: the one
cross-domain return type, so a non-success solver result (the `syevd` /
`orthogonalize` / `getrf`-batched steps) now surfaces in its OWN solver domain
and a device-copy failure in the runtime domain, instead of the old
`WWRBLAS_STATUS_INTERNAL_ERROR` masquerade. Genuine host-side outcomes the
iteration itself reaches still carry an explicit BLAS-domain code deliberately:
a bad argument is `INVALID_VALUE`, a nonzero devInfo is whatever
`calaman::devinfo_verdict` (`calaman.error_handling`) says, and `info.reason`
distinguishes a `SUCCESS` return that covers `MaxIterations` /
`SubspaceTooSmall`.

## Usage

```cpp
import calaman.feast;
import wwr.blas;        // wwrblasHandle_t
import wwr.solver;      // wwrsolverDnHandle_t
import wwr.runtime_api; // wwrStream_t
using namespace calaman;

std::size_t lwork = 0;
feast_bufferSize<double, 8>(cusolver, n, m0, &lwork);
// ... allocate d_work; fill d_Q (n x m0) with random numbers ...

FeastInfo<double> info{};
feast<double, 8>(cublas, cusolver, stream, wwr::WWRBLAS_FILL_MODE_LOWER,
                 n, d_A, lda, Emin, Emax, m0,
                 d_lambda, d_Q, d_work, lwork, {}, &info);
// d_lambda[0 .. info.m) are the eigenvalues in [Emin, Emax], ascending;
// the leading info.m columns of d_Q are their eigenvectors.
```

Both handles must already be set to `stream`. Only the `uplo` triangle of `d_A`
is read. `m0` has to exceed the number of eigenvalues in the interval — about
1.5 times it is the usual choice.

## Workspace

Dominated by the `Ne` resolvents, `Ne·n²` complex elements: 32 MiB at `n = 512`
in double with `Ne = 8`. Everything else is `O(n·m0)`. The QR and `syevd`
workspaces never overlap in time, so they share one region.
