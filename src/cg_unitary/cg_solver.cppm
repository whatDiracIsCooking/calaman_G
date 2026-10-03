/**
 * @file cg_solver.cppm
 * @brief Riemannian conjugate gradient on the Lie group U(n) -- paper Table 3
 *
 * Partition of calaman.cg_unitary.
 *
 * Minimizes (or maximizes) a real-valued J of a complex matrix argument subject
 * to W^H W = W W^H = I, by conjugate gradient along geodesics of U(n) (Abrudan,
 * Eriksson & Koivunen, "Conjugate gradient algorithm for optimization under
 * unitary matrix constraint", Signal Processing 89 (2009) 1704-1714).
 *
 * One iteration, with the paper's step numbers:
 *
 *   2. Psi = dJ/dW(W);  G = Psi W^H - W Psi^H            eq. (2), in u(n)
 *      H := G every n^2 iterations, n^2 being dim U(n)
 *   3. stop when <G, G> is small enough                   eq. (1)
 *   4. mu_k by geodesic line search                       Tables 1 or 2
 *   5. W <- exp(-mu_k H) W                                eq. (11)
 *   6. gamma_k = <G' - G, G'> / <G, G>;  H <- G' + gamma_k H    eqs. (10), (7)
 *   7. reset H := G' if <H, G'> < 0                        Remark 2
 *
 * ## Why conjugacy, and what it costs
 *
 * Riemannian steepest descent turns 90 degrees at every iteration just as its
 * Euclidean counterpart does, which is slow in a narrow valley (§2.3, Fig. 1).
 * CG combines the new gradient with the previous direction instead. The
 * complication on a manifold is that those two vectors live in different tangent
 * spaces, so the old direction must be parallel transported first; on U(n) the
 * transport of the velocity vector is just a right multiplication (eq. 5), and
 * with the approximate Polak-Ribiere formula of eq. (10) it disappears entirely.
 * That is the Lie group structure paying for itself. Setting gamma to zero
 * recovers steepest descent, which is how the tests measure what conjugacy buys.
 */

export module calaman.cg_unitary:cg_solver;

import std;
import wwr.blas;            // wwrblasHandle_t/Status, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.runtime_api;     // wwrStream_t, wwrMemcpy/MemsetAsync, wwrSuccess
import wwr.wrappers.common; // usual_fp, real_fp, ComplexToRealType
import wwr.wrappers.blas;   // gemm, axpy, scal
import calaman.common;      // kOne, kZero, kNegativeOne
import calaman.error_handling; // Status
import calaman.expm;        // expm, expm_bufferSize
import calaman.orthogonalize; // orthogonalize, orthogonalize_bufferSize
import :buffer_size;
import :cost_function;
import :geodesic_search;

export namespace calaman {

/// @brief Why the solver stopped.
enum class CgStopReason {
  Converged,        ///< <G, G> fell below the tolerance
  MaxIterations,    ///< the iteration budget ran out
  LineSearchFailed, ///< no step size found, even from the steepest direction
  NumericalFailure, ///< a BLAS, solver, or runtime call failed
};

/// @brief Tuning for cg_unitary. The defaults follow the paper.
template<typename R>
struct CgOptions {
  int max_iter = 300;

  /**
   * Stop when <G, G> = 0.5 ||G||_F^2 falls below this.
   *
   * The gradient norm, not the change in cost: on a compact manifold with an
   * almost periodic cost along every geodesic, a small step is not evidence of a
   * small gradient.
   */
  R tol = std::is_same_v<R, float> ? R(1e-10) : R(1e-20);

  LineSearchMethod method = LineSearchMethod::Polynomial;
  GeodesicSearchOptions search{};

  /**
   * Iterations between resets of H to the steepest direction. Zero uses the
   * paper's n^2, the real dimension of U(n) -- the usual "restart after a full
   * set of conjugate directions" rule.
   */
  int reset_period = 0;

  /**
   * Iterations between re-orthogonalizations of W; 0 disables.
   *
   * Normally unnecessary. The update is a matrix exponential of a skew-Hermitian
   * argument, which is unitary to roundoff, and the paper reports the departure
   * from unitarity sitting at machine precision (§5.1). Worth enabling for very
   * long runs in single precision.
   */
  int reorthogonalize_period = 0;
};

/// @brief What the solver did.
template<typename R>
struct CgInfo {
  int iterations = 0; ///< iterations completed
  int resets = 0;     ///< times H was reset to the steepest direction
  int line_search_failures = 0;
  R gradient_norm_sq = R(0); ///< final <G, G>
  R last_step = R(0);        ///< final mu_k
  CgStopReason reason = CgStopReason::MaxIterations;
};

/**
 * @brief Conjugate gradient on U(n) with the Polak-Ribiere factor. Paper Table 3.
 *
 * @tparam T     Element type (float, double, wwrFloatComplex, wwrDoubleComplex).
 *               The real types optimize over the orthogonal group instead, which
 *               the paper notes its algorithm also covers.
 * @tparam CostF A functor satisfying unitary_cost_function<CostF, T>.
 *
 * @param cublas_handle   Blas handle, its stream set to @p stream.
 * @param cusolver_handle Solver handle, its stream set to @p stream.
 * @param stream          Device stream.
 * @param n               Matrix dimension.
 * @param d_W             In/out: n x n unitary, packed (leading dimension n).
 *                        W_0 on entry -- the identity for the paper's own runs,
 *                        or a random unitary. Overwritten with the result.
 * @param cost            The cost function; see :cost_function.
 * @param dir             Minimize or maximize. Brockett is maximized.
 * @param d_work          Device workspace, from cg_unitary_bufferSize.
 * @param lwork_bytes     Its size.
 * @param opts            Tuning.
 * @param info            Host out, may be null.
 *
 * @return An ok() Status whenever the iteration ran to one of its own stopping
 *         conditions -- including MaxIterations and LineSearchFailed, which are
 *         outcomes rather than errors; read @p info to tell them apart. A failing
 *         Status means a BLAS, solver, or runtime call failed.
 *
 * @warning The leading dimension of @p d_W must be exactly n. The Frobenius
 *          reductions and the cost functor's device-mode dot both read the
 *          matrices as flat n^2 vectors.
 * @warning Synchronizes @p stream several times per iteration: the convergence
 *          test, the Polak-Ribiere factor and the step size are all host-side
 *          control flow over device-computed scalars. This is the same
 *          host-driven shape nnls/feast have, and is negligible against the
 *          O(n^3) products it sequences. Requires the blas handle's DEFAULT
 *          (host) pointer mode on entry.
 */
template<wwr::usual_fp T, typename CostF>
  requires unitary_cost_function<CostF, T>
Status cg_unitary(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                  wwr::wwrStream_t stream, const int n, T *d_W, const CostF &cost,
                  const CgDirection dir, void *d_work, const std::size_t lwork_bytes,
                  const CgOptions<wwr::ComplexToRealType<T>> &opts = {},
                  CgInfo<wwr::ComplexToRealType<T>> *info = nullptr) {
  using RealT = wwr::ComplexToRealType<T>;

  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (d_W == nullptr || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  CgInfo<RealT> local{};
  const auto publish = [&](const CgStopReason reason, const Status st) {
    local.reason = reason;
    if (info != nullptr) {
      *info = local;
    }
    return st;
  };

  std::size_t required = 0;
  const std::size_t cost_bytes = cost.bufferSize(n);
  Status status = cg_unitary_bufferSize<T>(cusolver_handle, n, cost_bytes, &required);
  if (!status.ok()) {
    return status;
  }
  if (lwork_bytes < required) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  std::size_t expm_bytes = 0;
  status = expm_bufferSize<T>(cusolver_handle, n, &expm_bytes);
  if (!status.ok()) {
    return status;
  }

  CgSlices<T> s = make_cg_slices<T>(d_work, n, expm_bytes, cost_bytes);

  const RealT sgn = cg_sign<RealT>(dir);
  const int reset_period = (opts.reset_period > 0) ? opts.reset_period : (n * n);
  const auto adj = wwr::real_fp<T> ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_C;
  const int nn = n * n;

  // Power iteration warm-start vector: zeroed so the first call reseeds it.
  if (wwr::wwrMemsetAsync(s.power_v, 0, static_cast<std::size_t>(n) * sizeof(T), stream) !=
      wwr::wwrSuccess) {
    return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
  }

  /// Psi = dJ/dW(W), then G = Psi W^H - W Psi^H into d_G. Paper step 2 / 6.
  const auto riemannian_gradient = [&](const T *d_point, T *d_G) -> wwr::wwrblasStatus_t {
    wwr::wwrblasStatus_t cst = cost.euclidean_gradient(cublas_handle, stream, n, d_point, n, s.psi,
                                                       n, s.scratch, s.scratch_bytes);
    if (cst != wwr::WWRBLAS_STATUS_SUCCESS) {
      return cst;
    }

    { // X = Psi W^H, in host pointer mode
      const cg_detail::HostPointerMode guard{cublas_handle};
      if (!guard.ok) {
        return wwr::WWRBLAS_STATUS_NOT_INITIALIZED;
      }
      cst = wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, adj, n, n, n, &kOne<T>, s.psi, n,
                              d_point, n, &kZero<T>, s.tmp, n);
    }
    if (cst != wwr::WWRBLAS_STATUS_SUCCESS) {
      return cst;
    }

    // G = X - X^H, the projection onto u(n).
    return cg_detail::skew_hermitian_part<T>(cublas_handle, n, RealT{1}, s.tmp, n, d_G, n);
  };

  // ── step 1: initialization ──────────────────────────────────────────────
  T *d_G = s.grad;
  T *d_G_next = s.grad_next;

  if (riemannian_gradient(d_W, d_G) != wwr::WWRBLAS_STATUS_SUCCESS) {
    return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }

  if (wwr::wwrMemcpyAsync(s.dir, d_G, static_cast<std::size_t>(nn) * sizeof(T),
                          wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
    return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
  }
  ++local.resets;

  // <G, G> = 0.5 ||G||_F^2, carrying the half so the convergence test -- the more
  // frequent reader -- needs no scaling.
  RealT g_norm{0};
  if (cg_detail::frobenius_norm<T>(cublas_handle, n, n, d_G, n, &g_norm) !=
      wwr::WWRBLAS_STATUS_SUCCESS) {
    return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
  }
  RealT gg = RealT{0.5} * g_norm * g_norm;
  local.gradient_norm_sq = gg;

  for (int k = 0; k < opts.max_iter; ++k) {
    // ── step 2: periodic reset to the steepest direction ──────────────────
    // k == 0 already did it above.
    if (k > 0 && (k % reset_period) == 0) {
      if (wwr::wwrMemcpyAsync(s.dir, d_G, static_cast<std::size_t>(nn) * sizeof(T),
                              wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      ++local.resets;
    }

    // ── step 3: convergence ───────────────────────────────────────────────
    if (gg < opts.tol) {
      return publish(CgStopReason::Converged, wwr::WWRBLAS_STATUS_SUCCESS);
    }

    // ── step 4: line search ───────────────────────────────────────────────
    RealT mu{0};
    status = geodesic_search<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W, s.dir, n,
                                       s.psi, cost, dir, opts.method, s, opts.search, &mu);
    if (!status.ok()) {
      return publish(CgStopReason::NumericalFailure, status);
    }

    if (!(mu > RealT{0})) {
      // No extremum along this geodesic. If the direction was conjugate, the
      // steepest one is still worth trying -- the standard recovery, and the
      // paper's Remark 2 resets for the same reason.
      ++local.line_search_failures;
      if (wwr::wwrMemcpyAsync(s.dir, d_G, static_cast<std::size_t>(nn) * sizeof(T),
                              wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      ++local.resets;

      status = geodesic_search<T, CostF>(cublas_handle, cusolver_handle, stream, n, d_W, s.dir, n,
                                         s.psi, cost, dir, opts.method, s, opts.search, &mu);
      if (!status.ok()) {
        return publish(CgStopReason::NumericalFailure, status);
      }
      if (!(mu > RealT{0})) {
        local.iterations = k;
        return publish(CgStopReason::LineSearchFailed, wwr::WWRBLAS_STATUS_SUCCESS);
      }
    }
    local.last_step = mu;

    // ── step 5: W <- exp(sigma mu H) W ────────────────────────────────────
    if (cg_detail::scale_into<T>(cublas_handle, n, sgn * mu, s.dir, n, s.tmp, n) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    status = expm<T>(cublas_handle, cusolver_handle, stream, n, s.tmp, n, s.rot, n, s.scratch,
                     s.scratch_bytes, s.ints + 3);
    if (!status.ok()) {
      return publish(CgStopReason::NumericalFailure, status);
    }

    {
      const cg_detail::HostPointerMode guard{cublas_handle};
      if (!guard.ok) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_NOT_INITIALIZED);
      }
      if (wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n, n, &kOne<T>,
                            s.rot, n, d_W, n, &kZero<T>, s.w_new, n) != wwr::WWRBLAS_STATUS_SUCCESS) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }
    }

    if (wwr::wwrMemcpyAsync(d_W, s.w_new, static_cast<std::size_t>(nn) * sizeof(T),
                            wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
    }

    if (opts.reorthogonalize_period > 0 && ((k + 1) % opts.reorthogonalize_period) == 0) {
      // orthogonalize() wants an int workspace count, and the scratch region is
      // shared with the exponential, which is not live here.
      int lwork_q = 0;
      if (orthogonalize_bufferSize<T>(cusolver_handle, n, n, &lwork_q).ok() &&
          static_cast<std::size_t>(lwork_q) * sizeof(T) <= s.scratch_bytes) {
        orthogonalize<T>(cusolver_handle, n, n, d_W, static_cast<T *>(s.scratch), lwork_q,
                         s.ints + 3, s.ints + 3);
      }
    }

    // ── step 6: new gradient and the Polak-Ribiere factor ─────────────────
    if (riemannian_gradient(d_W, d_G_next) != wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    RealT g_next_norm{0};
    if (cg_detail::frobenius_norm<T>(cublas_handle, n, n, d_G_next, n, &g_next_norm) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }
    const RealT gg_next = RealT{0.5} * g_next_norm * g_next_norm;

    // Gd = G' - G, in the scratch matrix the gradient assembly is done with.
    if (wwr::wwrMemcpyAsync(s.tmp, d_G_next, static_cast<std::size_t>(nn) * sizeof(T),
                            wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
    }
    {
      const cg_detail::HostPointerMode guard{cublas_handle};
      if (!guard.ok) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_NOT_INITIALIZED);
      }
      if (wwr::axpy<T, int>(cublas_handle, nn, &kNegativeOne<T>, d_G, 1, s.tmp, 1) !=
          wwr::WWRBLAS_STATUS_SUCCESS) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }
    }

    // gamma = <G' - G, G'> / <G, G>, eq. (10). Both inner products carry the same
    // factor of 0.5, so it cancels and the raw traces suffice -- hence the 2 * gg
    // below, gg being stored with the half in it.
    RealT numerator{0};
    if (cg_detail::frobenius_dot_real<T>(cublas_handle, n, n, s.tmp, n, d_G_next, n, &numerator) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    const RealT gamma = (gg > RealT{0}) ? (numerator / (RealT{2} * gg)) : RealT{0};

    // H <- G' + gamma H
    {
      const cg_detail::HostPointerMode guard{cublas_handle};
      if (!guard.ok) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_NOT_INITIALIZED);
      }
      const T gamma_t = cg_detail::cg_as_element<T>(gamma);
      wwr::wwrblasStatus_t cst = wwr::scal<T, int>(cublas_handle, nn, &gamma_t, s.dir, 1);
      if (cst == wwr::WWRBLAS_STATUS_SUCCESS) {
        cst = wwr::axpy<T, int>(cublas_handle, nn, &kOne<T>, d_G_next, 1, s.dir, 1);
      }
      if (cst != wwr::WWRBLAS_STATUS_SUCCESS) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
      }
    }

    std::swap(d_G, d_G_next);
    gg = gg_next;
    local.gradient_norm_sq = gg;

    // ── step 7: the direction must still point downhill (Remark 2) ────────
    RealT hg{0};
    if (cg_detail::frobenius_dot_real<T>(cublas_handle, n, n, s.dir, n, d_G, n, &hg) !=
        wwr::WWRBLAS_STATUS_SUCCESS) {
      return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
    }

    if (!(hg > RealT{0})) {
      if (wwr::wwrMemcpyAsync(s.dir, d_G, static_cast<std::size_t>(nn) * sizeof(T),
                              wwr::wwrMemcpyDeviceToDevice, stream) != wwr::wwrSuccess) {
        return publish(CgStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
      ++local.resets;
    }

    local.iterations = k + 1;
  }

  return publish(CgStopReason::MaxIterations, wwr::WWRBLAS_STATUS_SUCCESS);
}

} // namespace calaman
