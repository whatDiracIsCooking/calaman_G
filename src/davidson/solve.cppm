/**
 * @file solve.cppm
 * @brief The block-Davidson solve entry point, its options/result types and its
 *        operator/preconditioner/metric callbacks
 *
 * The :solve partition of calaman.davidson.
 *
 * davidson_solve iterates: extend Sigma_V (and, on the metric path, M V) on only
 * the newly appended columns (one sigma call), form the subspace problem and
 * diagonalize it, rotate the lowest n_roots Ritz pairs and their operator images,
 * test each residual's norm, lock the converged, precondition the full block,
 * collapse the subspace when a full new block would overflow max_subspace, and
 * expand by the twice-modified-Gram-Schmidt-orthonormalized corrections.
 *
 * Two problems share the loop, chosen by whether @p metric is empty:
 *   - EUCLIDEAN (empty metric): H = V^T Sigma_V, standard syevd, residual and
 *     orthogonality in the Euclidean inner product.
 *   - GENERALIZED (non-empty metric): the operator is self-adjoint in the metric
 *     <x,y>_M = x^T M y. H = (M V)^T Sigma_V and S = (M V)^T V, generalized sygvd;
 *     the residual norm is sqrt(R^T M R) and the subspace is expanded
 *     M-orthonormally -- M C is carried in lockstep through every projection so
 *     the inner products stay in the metric. The workspace must have been sized
 *     with_metric (make_davidson_slices); it is a usage error otherwise.
 *
 * Like calaman.feast, the solver owns neither the handles nor the stream (bind
 * them with wwrblasSetStream/wwrsolverDnSetStream and pass them in) and reports
 * the converged pairs through out-pointers -- eigenvectors into @p
 * eigenvectors_out, eigenvalues and the converged flag into @p result. A Status
 * failure means a genuine BLAS/solver/runtime fault; whether failing to converge
 * is ALSO surfaced as a non-success Status is DavidsonOptions::fail_on_non_convergence.
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
export import calaman.error_handling; // Status, PointerModeStatus

export namespace calaman {

/// @brief Convergence/iteration knobs for one davidson_solve call.
template<calaman::real_fp T>
struct DavidsonOptions {
  /// @brief A root is converged when its residual norm (Euclidean, or the M-norm
  ///        on the metric path) is at or below this.
  T residual_tolerance = T{1e-8};
  /// @brief Subspace expansions before giving up.
  int max_iterations = 100;
  /// @brief A re-orthogonalized correction at or below this norm carries no
  ///        component outside the retained subspace and is dropped. Kept well
  ///        below residual_tolerance: a near-converged root's correction shrinks
  ///        with its residual, and a floor too close to the tolerance would drop
  ///        genuine (small) new directions as spurious linear dependence.
  T linear_dependence_floor = T{1e-12};
  /// @brief Whether failing to converge (exhausting max_iterations, or the
  ///        subspace stagnating with no independent correction left) is surfaced
  ///        as a non-success Status. true (default): davidson_solve returns
  ///        WWRBLAS_STATUS_INTERNAL_ERROR, with @p result still filled with the
  ///        best-effort Ritz values for inspection. false: it returns success and
  ///        the caller reads DavidsonResult::converged. Either way @p result is
  ///        filled; this only chooses whether a CLM_TRY-style caller stops.
  bool fail_on_non_convergence = true;
};

/// @brief Outcome of a davidson_solve call.
template<calaman::real_fp T>
struct DavidsonResult {
  bool converged = false;
  int iterations = 0;
  /// @brief The n_roots lowest eigenvalues, ascending (host). Filled on every
  ///        completed run (its accuracy is only guaranteed when converged).
  std::vector<T> eigenvalues;
};

/// @brief sigma(stream, block_size, b, sigma_out): out[:, :block_size] =
///        A b[:, :block_size], device-resident, n x block_size column-major (ld n).
template<calaman::real_fp T>
using DavidsonSigmaFn =
    std::function<Status(wwr::wwrStream_t stream, int block_size, const T *b, T *sigma_out)>;

/// @brief precondition(stream, n_roots, theta, residual, correction): turn the
///        residual block (n x n_roots, device) into a correction block (same
///        shape, device), given the current Ritz values @p theta (n_roots, HOST).
template<calaman::real_fp T>
using DavidsonPreconditionFn = std::function<Status(wwr::wwrStream_t stream, int n_roots,
                                                    const T *theta, const T *residual,
                                                    T *correction)>;

/// @brief OPTIONAL metric(stream, block_size, b, m_out): apply the SPD metric M,
///        same shape as DavidsonSigmaFn. Empty (default) selects the Euclidean
///        path; non-empty selects the generalized path and requires the workspace
///        to have been sized with_metric.
template<calaman::real_fp T>
using DavidsonMetricFn =
    std::function<Status(wwr::wwrStream_t stream, int block_size, const T *b, T *m_out)>;

} // namespace calaman

export namespace calaman {

/**
 * @brief Converge the lowest @p n_roots eigenpairs of a symmetric operator known
 *        only through @p sigma, from an initial @p guess.
 *
 * @param stream          Stream both handles are bound to; must outlive the call.
 * @param n,n_roots,max_subspace  The shape @p s was carved for.
 * @param guess           n x guess_count device, column-major (ld n). Full column
 *                        rank; orthonormal (Euclidean) is the usual seed.
 * @param guess_count     n_roots <= guess_count <= max_subspace.
 * @param s               Workspace from make_davidson_slices; sized with_metric
 *                        iff @p metric is non-empty.
 * @param eigenvectors_out  Out: converged Ritz vectors, n x n_roots device (ld n);
 *                        also the per-iteration Ritz-vector scratch.
 * @param result          Out (host): eigenvalues, iteration count, converged flag.
 * @param metric          OPTIONAL; empty selects the Euclidean path, non-empty the
 *                        generalized (metric) one.
 * @return success on a converged solve; WWRBLAS_STATUS_INTERNAL_ERROR on a
 *         non-converged one when options.fail_on_non_convergence (result still
 *         filled); INVALID_VALUE for a bad guess_count, or a non-empty metric with
 *         a workspace not sized with_metric; or a propagated BLAS/solver/runtime
 *         failure.
 */
template<calaman::real_fp T>
Status davidson_solve(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                      wwr::wwrStream_t stream, int n, int n_roots, int max_subspace, const T *guess,
                      int guess_count, const DavidsonSlices<T> &s, const DavidsonSigmaFn<T> &sigma,
                      const DavidsonPreconditionFn<T> &precondition, T *eigenvectors_out,
                      DavidsonResult<T> *result, const DavidsonOptions<T> &options = {},
                      const DavidsonMetricFn<T> &metric = {}) {
  CLM_REQUIRE(result != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(guess_count >= n_roots && guess_count <= max_subspace,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const bool use_metric = static_cast<bool>(metric);
  // The generalized path needs the metric regions; a non-empty metric against an
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

  result->converged = false;
  result->iterations = 0;
  result->eigenvalues.assign(static_cast<std::size_t>(n_roots), T{0});

  CLM_TRY(wwr::wwrMemcpyAsync(s.v, guess, n_bytes * static_cast<std::size_t>(guess_count),
                              wwr::wwrMemcpyDeviceToDevice, stream));

  int dim = guess_count;
  int filled = 0; // columns of s.av already holding sigma(s.v[:, :filled])

  const auto finish = [&](int iters) -> Status {
    result->converged = false;
    result->iterations = iters;
    result->eigenvalues.assign(theta.begin(), theta.end());
    return options.fail_on_non_convergence ? Status{wwr::WWRBLAS_STATUS_INTERNAL_ERROR}
                                           : Status{wwr::WWRBLAS_STATUS_SUCCESS};
  };

  // M-norm helper: sqrt(x^T M x) = sqrt(x . Mx), guarding a tiny negative from roundoff.
  const auto m_norm = [](T dot_val) -> T { return dot_val > T{0} ? std::sqrt(dot_val) : T{0}; };

  for (int iter = 1; iter <= options.max_iterations; ++iter) {
    // 1. Extend Sigma_V (and M V on the metric path) for the new columns only.
    if (dim > filled) {
      const int new_count = dim - filled;
      const std::size_t offset = static_cast<std::size_t>(filled) * nz;
      CLM_TRY(sigma(stream, new_count, s.v + offset, s.av + offset));
      if (use_metric) {
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
    // info != 0 is a synthetic host-side verdict on the eigensolver's devInfo, so
    // (like feast's own host-side conditions) it carries a BLAS-domain code -- the
    // solver-domain status constants are macros, not importable through wwr.solver.
    CLM_REQUIRE(info_host == 0, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);

    // 3. Ritz vectors X = V S_k and their operator images A X = Sigma_V S_k.
    CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots, dim,
                               &kOne<T>, s.v, n, s.h, max_subspace, &kZero<T>, eigenvectors_out,
                               n)));
    CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots, dim,
                               &kOne<T>, s.av, n, s.h, max_subspace, &kZero<T>, s.ritz_av, n)));
    CLM_TRY(wwr::wwrMemcpyAsync(theta.data(), s.ritz, sizeof(T) * static_cast<std::size_t>(n_roots),
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));

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
    if (use_metric) {
      CLM_TRY(metric(stream, n_roots, s.residual, s.metric_scratch));
    }
    bool all_converged = true;
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
      if (norm > options.residual_tolerance) {
        all_converged = false;
      }
    }
    if (all_converged) {
      result->converged = true;
      result->iterations = iter;
      result->eigenvalues.assign(theta.begin(), theta.end());
      return wwr::WWRBLAS_STATUS_SUCCESS;
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
      if (use_metric) {
        CLM_TRY(metric(stream, n_roots, s.v, s.mv));
      }
      dim = n_roots;
      filled = n_roots;
    }

    // On the metric path, precompute M applied to the correction block, carried in
    // lockstep with the correction through every projection/axpy below so the
    // orthogonalization is done in the M-inner-product.
    if (use_metric) {
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
      if (residual_norms[static_cast<std::size_t>(i)] <= options.residual_tolerance) {
        continue;
      }
      const std::size_t ci = static_cast<std::size_t>(i) * nz;
      T *candidate = s.correction + ci;
      T *m_candidate = s.metric_scratch + ci; // M candidate (metric path only)
      for (int j = 0; j < kept; ++j) {
        const std::size_t aj = static_cast<std::size_t>(dim + j) * nz;
        T *accepted = s.v + aj;
        T *m_accepted = s.mv + aj;
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
      return finish(iter);
    }
    dim += kept;
  }

  return finish(options.max_iterations);
}

} // namespace calaman
