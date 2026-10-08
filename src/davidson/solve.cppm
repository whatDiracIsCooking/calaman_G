/**
 * @file solve.cppm
 * @brief The block-Davidson solve entry point, its options/info types and its
 *        operator/preconditioner/metric callbacks
 *
 * The :solve partition of calaman.davidson.
 *
 * davidson_solve applies a linear_operator (calaman.linear_operator) to only the
 * newly appended columns each iteration, diagonalizes the subspace problem,
 * tests each Ritz residual, locks the converged, preconditions, collapses the
 * subspace before it would overflow max_subspace, and expands by the
 * twice-Gram-Schmidt-orthonormalized corrections. A davidson_sigma callable is
 * wrapped in DavidsonSigmaOperator, so it takes the same path.
 *
 * The metric's type picks the problem at compile time:
 *   - EUCLIDEAN (DavidsonNoMetric, the default): H = V^T Sigma_V, syevd.
 *   - GENERALIZED (any other metric): self-adjoint in <x,y>_M = x^T M y;
 *     H = (M V)^T Sigma_V, S = (M V)^T V, sygvd, M-norm residuals and
 *     M-orthonormal expansion. Needs a workspace sized with_metric.
 *
 * The solver owns neither the handles nor the stream (bind both to it first).
 * Non-convergence is an outcome (calaman.iterative): a failing Status means a
 * BLAS/solver/runtime fault or a bad argument, never an exhausted budget.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
// Resolved root-relative via the src/ root calaman.error_handling exports; needs
// calaman::Status visible at expansion, which the export import below supplies.
#include "error_handling/error_macros.h"

export module calaman.davidson:solve;

import std;
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR, WWRSOLVER_EIG_TYPE_1
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.wrappers.blas;   // gemm, axpy, dot, nrm2, scal
import wwr.wrappers.solver; // syevd, sygvd
import wwr.extension.blas;  // ScopedPointerMode (forces host mode for the solve)
import :buffer_size;        // DavidsonSlices
import calaman.common;      // kOne, kZero, kNegativeOne, real_fp
import calaman.ritz;        // RitzSelection, classify_ritz, ritz_rotate
export import calaman.error_handling;  // Status, PointerModeStatus
export import calaman.iterative;       // IterationInfo, stop_reason, converged
export import calaman.linear_operator; // linear_operator -- what the solve applies

export namespace calaman {

/// @brief Convergence/iteration knobs for one davidson_solve call.
template<calaman::real_fp T>
struct DavidsonOptions {
  /// @brief Relative: a pair converges when ||A x - theta x||_2 (the M-norm on
  ///        the metric path) <= tol * max(|theta|, ||H||_2), ||H||_2 = max |theta|
  ///        over the subspace spectrum (classify_ritz). Compared: architecture.md §8.
  T residual_tolerance = T{1e-8};
  /// @brief Subspace expansions before giving up.
  int max_iterations = 100;
  /// @brief A re-orthogonalized correction at or below this norm carries no
  ///        component outside the retained subspace and is dropped. Kept well
  ///        below residual_tolerance: a near-converged root's correction shrinks
  ///        with its residual, and a floor too close to the tolerance would drop
  ///        genuine (small) new directions as spurious linear dependence.
  T linear_dependence_floor = T{1e-12};
};

/// @brief Why davidson_solve stopped.
enum class DavidsonStopReason {
  Converged,        ///< every root's residual norm met the residual_tolerance bound
  MaxIterations,    ///< the iteration budget ran out
  Stagnated,        ///< no correction survived re-orthogonalization: the subspace stopped growing
  NumericalFailure, ///< a BLAS/solver/runtime call or a callback failed (Status says which)
};
static_assert(stop_reason<DavidsonStopReason>);

/// @brief What davidson_solve did: iterations are subspace expansions.
template<calaman::real_fp T>
struct DavidsonInfo : IterationInfo<DavidsonStopReason> {
  T max_residual_norm = T{0}; ///< largest root residual norm at the last Rayleigh-Ritz
};

/// @brief sigma(stream, block_size, b, sigma_out): out[:, :block_size] =
///        A b[:, :block_size], device-resident, n x block_size column-major (ld n).
template<typename F, typename T>
concept davidson_sigma =
    requires(const F &f, wwr::wwrStream_t stream, int block_size, const T *b, T *sigma_out) {
      { f(stream, block_size, b, sigma_out) } -> std::convertible_to<Status>;
    };

/// @brief The linear_operator over a davidson_sigma: apply(stream, k, X, Y) is
///        sigma(stream, k, X, Y), one call per apply. Holds a reference to it.
template<calaman::real_fp T, davidson_sigma<T> F>
struct DavidsonSigmaOperator {
  const F &sigma;

  Status apply(wwr::wwrStream_t stream, const int k, const T *X, T *Y) {
    return sigma(stream, k, X, Y);
  }
};

/// @brief precondition(stream, n_roots, theta, residual, correction): turn the
///        residual block (n x n_roots, device) into a correction block (same
///        shape, device), given the current Ritz values @p theta (n_roots, HOST).
template<typename F, typename T>
concept davidson_preconditioner = requires(const F &f, wwr::wwrStream_t stream, int n_roots,
                                           const T *theta, const T *residual, T *correction) {
  { f(stream, n_roots, theta, residual, correction) } -> std::convertible_to<Status>;
};

/// @brief The default metric argument: selects the Euclidean path at compile time.
struct DavidsonNoMetric {};

/// @brief OPTIONAL metric(stream, block_size, b, m_out): apply the SPD metric M,
///        same shape as davidson_sigma. DavidsonNoMetric (default) selects the
///        Euclidean path; any other type the generalized path, which requires the
///        workspace to have been sized with_metric.
template<typename F, typename T>
concept davidson_metric = std::same_as<F, DavidsonNoMetric> || davidson_sigma<F, T>;

} // namespace calaman

export namespace calaman {

/**
 * @brief Converge the lowest @p n_roots eigenpairs of the symmetric operator
 *        @p op (applied in HOST pointer mode), from an initial @p guess.
 *
 * @param stream          Stream both handles are bound to; must outlive the call.
 * @param n,n_roots,max_subspace  The shape @p s was carved for; n is also @p op's.
 * @param guess           n x guess_count device, column-major (ld n). Full column
 *                        rank; orthonormal (Euclidean) is the usual seed.
 * @param guess_count     n_roots <= guess_count <= max_subspace.
 * @param s               Workspace from make_davidson_slices; sized with_metric
 *                        iff @p metric is not DavidsonNoMetric.
 * @param eigenvalues_out  Out: the n_roots lowest Ritz values, ascending, device;
 *                        written on every successful return, converged or not.
 * @param eigenvectors_out  Out: their Ritz vectors, n x n_roots device (ld n);
 *                        also the per-iteration Ritz-vector scratch.
 * @param info            Out (host): iterations, stop reason, final residual.
 * @param metric          OPTIONAL; DavidsonNoMetric (default) selects the Euclidean
 *                        path, any other davidson_metric the generalized one.
 * @return success whenever the solve stops on its own (read info->reason);
 *         INVALID_VALUE for a null out-pointer, a bad guess_count, or a metric
 *         with a workspace not sized with_metric; else a propagated fault, with
 *         info->reason NumericalFailure.
 */
template<calaman::real_fp T, linear_operator<T> Op, davidson_preconditioner<T> Precondition,
         davidson_metric<T> Metric = DavidsonNoMetric>
Status davidson_solve(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                      wwr::wwrStream_t stream, int n, int n_roots, int max_subspace, const T *guess,
                      int guess_count, const DavidsonSlices<T> &s, Op &op,
                      const Precondition &precondition, T *eigenvalues_out, T *eigenvectors_out,
                      DavidsonInfo<T> *info, const DavidsonOptions<T> &options = {},
                      const Metric &metric = {}) {
  CLM_REQUIRE(info != nullptr && eigenvalues_out != nullptr && eigenvectors_out != nullptr,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(guess_count >= n_roots && guess_count <= max_subspace,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  constexpr bool use_metric = !std::same_as<Metric, DavidsonNoMetric>;
  // The generalized path needs the metric regions; a metric against an
  // Euclidean-sized workspace is a usage error, not a silent Euclidean solve.
  CLM_REQUIRE(!use_metric || (s.mv != nullptr && s.s_sub != nullptr && s.metric_scratch != nullptr),
              wwr::WWRBLAS_STATUS_INVALID_VALUE);

  // The level-1 reductions (dot/nrm2) and the scaled updates take HOST scalars and
  // write HOST results, so the handle is in host pointer mode across the solve.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  const auto nz = static_cast<std::size_t>(n);
  const std::size_t n_bytes = sizeof(T) * nz;
  const std::size_t block_bytes = n_bytes * static_cast<std::size_t>(n_roots);

  std::vector<T> theta(static_cast<std::size_t>(n_roots));
  std::vector<T> residual_norms(static_cast<std::size_t>(n_roots));

  // Every early (CLM_TRY) return from here on is a fault; each stop of the
  // solve's own overwrites the reason.
  *info = {};
  info->reason = DavidsonStopReason::NumericalFailure;

  CLM_TRY(wwr::wwrMemcpyAsync(s.v, guess, n_bytes * static_cast<std::size_t>(guess_count),
                              wwr::wwrMemcpyDeviceToDevice, stream));

  int dim = guess_count;
  int filled = 0; // columns of s.av already holding A s.v[:, :filled]

  // The last Rayleigh-Ritz left the n_roots lowest Ritz values in s.ritz; with
  // no iteration run there are none, so the output is zeroed.
  const auto finish = [&](int iters, DavidsonStopReason reason) -> Status {
    const std::size_t value_bytes = sizeof(T) * static_cast<std::size_t>(n_roots);
    if (iters == 0) {
      CLM_TRY(wwr::wwrMemsetAsync(eigenvalues_out, 0, value_bytes, stream));
    } else {
      CLM_TRY(wwr::wwrMemcpyAsync(eigenvalues_out, s.ritz, value_bytes,
                                  wwr::wwrMemcpyDeviceToDevice, stream));
    }
    info->iterations = iters;
    info->reason = reason;
    return wwr::WWRBLAS_STATUS_SUCCESS;
  };

  // M-norm helper: sqrt(x^T M x) = sqrt(x . Mx), guarding a tiny negative from roundoff.
  const auto m_norm = [](T dot_val) -> T { return dot_val > T{0} ? std::sqrt(dot_val) : T{0}; };

  for (int iter = 1; iter <= options.max_iterations; ++iter) {
    // 1. Extend Sigma_V (and M V on the metric path) for the new columns only:
    //    one op.apply per iteration.
    if (dim > filled) {
      const int new_count = dim - filled;
      const std::size_t offset = static_cast<std::size_t>(filled) * nz;
      CLM_TRY(op.apply(stream, new_count, s.v + offset, s.av + offset));
      if constexpr (use_metric) {
        CLM_TRY(metric(stream, new_count, s.v + offset, s.mv + offset));
      }
      filled = dim;
    }

    // 2. Rayleigh-Ritz. Euclidean: H = V^T Sigma_V, syevd. Metric: H = (MV)^T
    //    Sigma_V and S = (MV)^T V, generalized sygvd (H symmetric because Sigma is
    //    self-adjoint in M, S SPD).
    CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim, dim, n,
                               &kOne<T>, use_metric ? s.mv : s.v, n, s.av, n, &kZero<T>, s.h,
                               max_subspace)));
    if (use_metric) {
      CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim, dim, n,
                                 &kOne<T>, s.mv, n, s.v, n, &kZero<T>, s.s_sub, max_subspace)));
      CLM_TRY(wwr::sygvd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_TYPE_1,
                            wwr::WWRSOLVER_EIG_MODE_VECTOR, wwr::WWRBLAS_FILL_MODE_LOWER, dim, s.h,
                            max_subspace, s.s_sub, max_subspace, s.ritz, s.eig_scratch,
                            s.lwork_eig, s.info));
    } else {
      CLM_TRY(wwr::syevd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                            wwr::WWRBLAS_FILL_MODE_LOWER, dim, s.h, max_subspace, s.ritz,
                            s.eig_scratch, s.lwork_eig, s.info));
    }
    int info_host = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&info_host, s.info, sizeof(int), wwr::wwrMemcpyDeviceToHost,
                                stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    // devInfo's domain and code: calaman::devinfo_verdict (calaman.error_handling).
    CLM_TRY(devinfo_verdict(info_host));

    // 3. Ritz vectors X = V S_k and their operator images A X = Sigma_V S_k (the
    //    lowest n_roots: the leading columns of S, ld max_subspace).
    CLM_TRY(ritz_rotate<T>(cublas_handle, n, dim, n_roots, s.v, n, s.h, max_subspace,
                           eigenvectors_out, n));
    CLM_TRY(ritz_rotate<T>(cublas_handle, n, dim, n_roots, s.av, n, s.h, max_subspace, s.ritz_av,
                           n));
    CLM_TRY(wwr::wwrMemcpyAsync(theta.data(), s.ritz, sizeof(T) * static_cast<std::size_t>(n_roots),
                                wwr::wwrMemcpyDeviceToHost, stream));
    // ||H||_2 = max |theta| over the ascending subspace spectrum: its two ends.
    T theta_top{};
    CLM_TRY(wwr::wwrMemcpyAsync(&theta_top, s.ritz + (dim - 1), sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    const T h_norm = std::max(std::abs(theta.front()), std::abs(theta_top));

    // 4. Residual R = A X - X diag(theta), then its per-root norm. Euclidean:
    //    ||R_i||. Metric: sqrt(R_i^T M R_i) -- the norm the operator is symmetric
    //    in, which is where convergence must be measured.
    CLM_TRY(wwr::wwrMemcpyAsync(s.residual, s.ritz_av, block_bytes, wwr::wwrMemcpyDeviceToDevice,
                                stream));
    for (int i = 0; i < n_roots; ++i) {
      const std::size_t col = static_cast<std::size_t>(i) * nz;
      const T neg_theta = -theta[static_cast<std::size_t>(i)];
      CLM_TRY((wwr::axpy<T, int>(cublas_handle, n, &neg_theta, eigenvectors_out + col, 1,
                                 s.residual + col, 1)));
    }
    if constexpr (use_metric) {
      CLM_TRY(metric(stream, n_roots, s.residual, s.metric_scratch));
    }
    for (int i = 0; i < n_roots; ++i) {
      const std::size_t col = static_cast<std::size_t>(i) * nz;
      T norm{};
      if (use_metric) {
        T mnorm2{};
        CLM_TRY((wwr::dot<T, int>(cublas_handle, n, s.residual + col, 1, s.metric_scratch + col, 1,
                                  &mnorm2)));
        norm = m_norm(mnorm2);
      } else {
        CLM_TRY((wwr::nrm2<T, int>(cublas_handle, n, s.residual + col, 1, &norm)));
      }
      residual_norms[static_cast<std::size_t>(i)] = norm;
    }
    // Relative test at scale ||H||_2 (DavidsonOptions::residual_tolerance).
    const RitzSelection<T> sel =
        classify_ritz<T>(theta, residual_norms, options.residual_tolerance, h_norm);
    info->max_residual_norm = std::ranges::fold_left(
        sel.residuals, T{0}, [](const T a, const T b) { return std::max(a, b); });
    if (sel.all_converged()) {
      return finish(iter, DavidsonStopReason::Converged);
    }

    // 5. Precondition the full block (locked roots' output is discarded below).
    CLM_TRY(precondition(stream, n_roots, theta.data(), s.residual, s.correction));

    // 6. Collapse if a full new block would not fit; make_davidson_slices
    //    guarantees max_subspace >= 2 * n_roots, so this leaves room for one. The
    //    metric path keeps M V in lockstep, so its collapsed block's M image is
    //    refreshed (M X) with one metric apply.
    if (dim + n_roots > max_subspace) {
      CLM_TRY(wwr::wwrMemcpyAsync(s.v, eigenvectors_out, block_bytes, wwr::wwrMemcpyDeviceToDevice,
                                  stream));
      CLM_TRY(wwr::wwrMemcpyAsync(s.av, s.ritz_av, block_bytes, wwr::wwrMemcpyDeviceToDevice,
                                  stream));
      if constexpr (use_metric) {
        CLM_TRY(metric(stream, n_roots, s.v, s.mv));
      }
      dim = n_roots;
      filled = n_roots;
    }

    // On the metric path, precompute M applied to the correction block, carried in
    // lockstep with the correction through every projection/axpy below so the
    // orthogonalization is done in the M-inner-product.
    if constexpr (use_metric) {
      CLM_TRY(metric(stream, n_roots, s.correction, s.metric_scratch));
    }

    // 7. Modified Gram-Schmidt of the correction block against the retained
    //    subspace, twice (a single classical pass loses orthogonality here).
    //    Euclidean projection V^T C; metric (M V)^T C, subtracted from both C (via
    //    V) and M C (via M V) to keep them consistent.
    const T *proj_left = use_metric ? s.mv : s.v;
    for (int pass = 0; pass < 2; ++pass) {
      CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim, n_roots,
                                 n, &kOne<T>, proj_left, n, s.correction, n, &kZero<T>, s.proj,
                                 max_subspace)));
      CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots,
                                 dim, &kNegativeOne<T>, s.v, n, s.proj, max_subspace, &kOne<T>,
                                 s.correction, n)));
      if (use_metric) {
        CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots,
                                   dim, &kNegativeOne<T>, s.mv, n, s.proj, max_subspace, &kOne<T>,
                                   s.metric_scratch, n)));
      }
    }

    // 8. Sequential Gram-Schmidt of the surviving candidates against each other
    //    (in the active inner product), skipping locked roots and dropping a
    //    candidate with no component left outside the retained subspace.
    int kept = 0;
    for (int i = 0; i < n_roots; ++i) {
      if (sel.converged[static_cast<std::size_t>(i)]) {
        continue;
      }
      const std::size_t ci = static_cast<std::size_t>(i) * nz;
      T *candidate = s.correction + ci;
      // M candidate (metric path only; metric_scratch is null without a metric,
      // and offsetting a null pointer is UB).
      T *m_candidate = use_metric ? s.metric_scratch + ci : nullptr;
      for (int j = 0; j < kept; ++j) {
        const std::size_t aj = static_cast<std::size_t>(dim + j) * nz;
        T *accepted = s.v + aj;
        T *m_accepted = use_metric ? s.mv + aj : nullptr;
        // <accepted, candidate> in the active product: Euclidean accepted^T candidate;
        // metric accepted^T M candidate = (M accepted)^T candidate.
        T projection{};
        CLM_TRY((wwr::dot<T, int>(cublas_handle, n, use_metric ? m_accepted : accepted, 1,
                                  candidate, 1, &projection)));
        const T neg_projection = -projection;
        CLM_TRY((wwr::axpy<T, int>(cublas_handle, n, &neg_projection, accepted, 1, candidate, 1)));
        if (use_metric) {
          CLM_TRY((wwr::axpy<T, int>(cublas_handle, n, &neg_projection, m_accepted, 1, m_candidate,
                                     1)));
        }
      }
      T norm{};
      if (use_metric) {
        T mnorm2{};
        CLM_TRY((wwr::dot<T, int>(cublas_handle, n, candidate, 1, m_candidate, 1, &mnorm2)));
        norm = m_norm(mnorm2);
      } else {
        CLM_TRY((wwr::nrm2<T, int>(cublas_handle, n, candidate, 1, &norm)));
      }
      if (norm <= options.linear_dependence_floor) {
        continue;
      }
      const T inv_norm = T{1} / norm;
      CLM_TRY((wwr::scal<T, int>(cublas_handle, n, &inv_norm, candidate, 1)));
      CLM_TRY(wwr::wwrMemcpyAsync(s.v + static_cast<std::size_t>(dim + kept) * nz, candidate,
                                  n_bytes, wwr::wwrMemcpyDeviceToDevice, stream));
      if (use_metric) {
        // Keep this direction's M image for later candidates in THIS pass; step 1
        // recomputes M V for the whole appended block next iteration.
        CLM_TRY((wwr::scal<T, int>(cublas_handle, n, &inv_norm, m_candidate, 1)));
        CLM_TRY(wwr::wwrMemcpyAsync(s.mv + static_cast<std::size_t>(dim + kept) * nz, m_candidate,
                                    n_bytes, wwr::wwrMemcpyDeviceToDevice, stream));
      }
      ++kept;
    }

    // Stagnation: no independent direction survived before every root converged.
    if (kept == 0) {
      return finish(iter, DavidsonStopReason::Stagnated);
    }
    dim += kept;
  }

  return finish(std::max(options.max_iterations, 0), DavidsonStopReason::MaxIterations);
}

/// @brief davidson_solve over a davidson_sigma callable, through
///        DavidsonSigmaOperator: same calls, same contract.
template<calaman::real_fp T, davidson_sigma<T> Sigma, davidson_preconditioner<T> Precondition,
         davidson_metric<T> Metric = DavidsonNoMetric>
  requires(!linear_operator<Sigma, T>)
Status davidson_solve(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                      wwr::wwrStream_t stream, int n, int n_roots, int max_subspace, const T *guess,
                      int guess_count, const DavidsonSlices<T> &s, const Sigma &sigma,
                      const Precondition &precondition, T *eigenvalues_out, T *eigenvectors_out,
                      DavidsonInfo<T> *info, const DavidsonOptions<T> &options = {},
                      const Metric &metric = {}) {
  DavidsonSigmaOperator<T, Sigma> op{sigma};
  return davidson_solve<T>(cublas_handle, cusolver_handle, stream, n, n_roots, max_subspace, guess,
                           guess_count, s, op, precondition, eigenvalues_out, eigenvectors_out,
                           info, options, metric);
}

} // namespace calaman
