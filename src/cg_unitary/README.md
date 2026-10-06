# calaman.cg_unitary

Riemannian **conjugate gradient** for optimization under a unitary matrix
constraint, after Abrudan, Eriksson & Koivunen, *Conjugate gradient algorithm
for optimization under unitary matrix constraint*, Signal Processing 89 (2009)
1704–1714.

Minimize or maximize a real-valued `J(W)` subject to `W ∈ U(n)`, by conjugate
gradient along geodesics of the unitary group, with the step size chosen by one
of the paper's two almost-periodicity line searches.

## Module

`calaman.cg_unitary`

| Partition | Contents |
|---|---|
| `:cost_function` | `unitary_cost_function` concept — order, gradient, value |
| `:brockett` | `J(W) = trace{W^H R W N}`, the paper's §5.1 example |
| `:buffer_size` | the single-buffer workspace layout and its sizing |
| `:geodesic_search` | the line searches of Tables 1 and 2 |
| `:cg_solver` | the iteration of Table 3 |

## The idea

A constrained problem on `C^{n×n}` becomes an *unconstrained* one on the
parameter space the constraint defines — the Lie group `U(n)`, which is both a
Riemannian manifold and a matrix group. Working there satisfies the constraint by
construction rather than by reprojection, and the group structure makes the
pieces cheap:

- **Geodesics are the matrix exponential**: `Γ(t) = exp(tS)W` for `S ∈ u(n)`
  (eq. 3), via [`calaman.expm`](../expm/README.md). No retraction approximation.
- **The gradient is a skew-Hermitian part**: `G = Psi W^H − W Psi^H` (eq. 2),
  one `geam` (the solver's `skew_hermitian_part` helper).
- **Parallel transport of the velocity vector is a right multiplication**
  (eq. 5), which is what makes conjugacy affordable here at all.

Riemannian steepest descent turns 90° at every iteration just as its Euclidean
counterpart does, which is slow in a narrow valley (§2.3, Fig. 1). CG combines
the new gradient with the previous direction instead. The complication on a
manifold is that those two vectors live in different tangent spaces; with the
approximate Polak–Ribière formula of eq. (10) the transport disappears entirely.

## One iteration (Table 3)

```
2. Psi = dJ/dW(W);  G = Psi W^H − W Psi^H             eq. (2), in u(n)
   H := G every n² iterations, n² being dim U(n)
3. stop when <G, G> is small enough                    eq. (1)
4. mu_k by geodesic line search                        Tables 1 or 2
5. W ← exp(−mu_k H) W                                  eq. (11)
6. gamma_k = <G′ − G, G′> / <G, G>;  H ← G′ + gamma_k H    eqs. (10), (7)
7. reset H := G′ if <H, G′> < 0                         Remark 2
```

Setting `reset_period = 1` discards the conjugate direction every iteration,
recovering steepest descent — which is how the tests measure what conjugacy
buys.

## The line searches

Both rest on one observation (§2.2). A geodesic is `exp(mu S)W` with `S`
skew-Hermitian, so `S` has purely imaginary eigenvalues and `exp(mu S)` has
eigenvalues `exp(i omega_k mu)`: a smooth cost function restricted to a geodesic
is an **almost periodic** function of `mu`, and so is its derivative. The highest
frequency present is `q |omega_max|`, so

```
T_mu = 2 pi / (q |omega_max|)                          eq. (15)
```

is an interval in which the derivative completes at most one cycle of its fastest
component and therefore crosses zero at most twice (§3.1). That bound is what
makes a low-order approximation adequate, and it is why the searches approximate
the **derivative** and find its zeros rather than approximating the cost and
minimizing it.

The derivative needs no trace. With `A = Psi(RW)` and `B = HRW`, the second
factor of eq. (14) is `B^H`, so

```
dJhat/dmu = 2 sigma Re trace{A B^H} = 2 sigma Re dotc(B, A)
```

over the `n²` elements read as one vector — one level-1 reduction per sample.

| | Table 1 (`Polynomial`) | Table 2 (`Dft`) |
|---|---|---|
| Fits | degree-`P` polynomial over `[0, T_mu]` | Hann-windowed DFT over `N_T` periods |
| Gradients | `P + 1` | `N_DFT` |
| Returns | the **first** zero crossing | the **best of several** minima |
| Exponentials | 1 | 1 |

`exp(i·mu·H) = [exp(mu·H)]^i` (§3.3), so only the base exponential is computed and
the rest are its powers. Sample 0 is the current point, so it reuses the gradient
the solver already has — one gradient evaluation saved per search.

## What stays on the device

The samples are taken in **device** pointer mode, so they never reach the host.
The scalar stages — the Vandermonde solve, the windowed DFT, the root finding and
the step selection — all run in single-block kernels (`cg_unitary.cu`) on those
device values. A search therefore costs `O(1)` host synchronizations: one inside
the matrix exponential (which needs the matrix 1-norm on the host to pick its
Padé degree) and one to read the chosen step back for the solver's control flow,
rather than one per sample.

The root finders are this module's own device code — there is no shared
`poly_roots` module in calaman. Both are **bracketing + bisection** on a real
function, which needs no complex root solve and cannot diverge: Table 1 brackets
its degree-≤ 5 derivative polynomial over `(0, T_mu]`, and Table 2 brackets the
**real trigonometric polynomial** `D(θ) = Σ_k c_k e^{ikθ}` the windowed DFT
coefficients define (it is real because the derivative samples are, so
`c_{−k} = conj(c_k)`), collecting every zero crossing over `(0, 2π)` as a step
argument. The frequency bound `|omega_max|` is estimated by **power iteration**
on `H^H H` (`skew_spectral_radius`), with `calaman.expm`'s `matrix_norm1` as the
`use_spectral_radius = false` surrogate.

## Deliberate deviations from the paper

1. **`dft_factor` defaults to 5, not the paper's 3.** The paper's `N_DFT` is
   unbounded; the single-warp root finder holds at most 31 samples, and at that
   cap `K` and `N_T` trade against each other. `K = 3, N_T = 10` spends the 31
   samples on ten periods and stalls around −19 dB of diagonality on the Brockett
   criterion; `K = 5` spends them on six and reaches the exact maximum.
2. **The DFT search falls back to the polynomial search** when no root survives
   near the unit circle. The paper sets `mu_k = 0` and stops (Table 2 step 11),
   but in a solver that is a stall — and as the gradient shrinks towards the
   optimum the reconstructed Fourier polynomial degenerates and its roots drift
   off the circle, exactly when a single-period fit would still work.

## Usage

```cpp
import calaman.cg_unitary;
using namespace calaman;

brockett_cost<T> cost{d_R, n, d_N, n};      // N = diag(1..n), as n entries

std::size_t lwork = 0;
cg_unitary_bufferSize<T>(cusolver, n, cost.bufferSize(n), &lwork);
// ... allocate d_work; set W_0 = I, or a random unitary ...

CgInfo<ComplexToRealType<T>> info{};
cg_unitary<T>(cublas, cusolver, stream, n, d_W, cost,
              CgDirection::Maximize, d_work, lwork, {}, &info);
```

The leading dimension of `d_W` **must** be exactly `n`: the Frobenius reductions
and the cost functor's device-mode dot both read the matrices as flat `n²`
vectors. An `ok()` `Status` comes back whenever the iteration reached one of its
own stopping conditions — including `MaxIterations` and `LineSearchFailed`, which
are outcomes rather than errors. Read `converged(info)` or `info.reason` to
tell them apart; `CgInfo` derives from `calaman.iterative`'s `IterationInfo`,
which has the project-wide rule (`src/iterative/README.md`).

## Workspace

Ten `n×n` blocks plus `O(n)` vectors, and one scratch region sized for whichever
of the matrix exponential or the cost function needs more — they are never live
at the same instant, so they alias.

## Writing another cost function

Satisfy `unitary_cost_function<F, T>`: a static `order` (the paper's `q`), an
`euclidean_gradient`, a `value` writing `J(W)` to a **device** scalar, and a
`bufferSize`. `order` must be right — too small and `T_mu` overshoots, breaking
the at-most-one-cycle bound. Brockett has `q = 2`; the JADE criterion of §5.2 has
`q = 4`.
