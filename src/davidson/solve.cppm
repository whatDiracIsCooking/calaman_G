/**
 * @file solve.cppm
 * @brief The block-Davidson solve entry point, its options/result types and its
 *        operator/preconditioner/metric callbacks
 *
 * The :solve partition of calaman.davidson.
 *
 * This unit pins the public interface -- the callback shapes, the convergence
 * knobs, the result -- and implements the EUCLIDEAN path of davidson_solve over
 * the DavidsonSlices workspace from :buffer_size. Per iteration: extend Sigma_V
 * on only the newly appended columns (one sigma call), form H = V^T Sigma_V and
 * diagonalize it with syevd, rotate the lowest n_roots Ritz pairs and their
 * operator images, test each residual's norm, lock the converged, precondition
 * the full block, collapse the subspace when a full new block would overflow
 * max_subspace, and expand by the twice-modified-Gram-Schmidt-orthogonalized
 * corrections. The generalized (metric, sygvd) path is NOT YET implemented: a
 * non-empty @p metric is rejected with WWRBLAS_STATUS_NOT_SUPPORTED.
 *
 * Like calaman.feast, the solver owns neither the handles nor the stream (bind
 * them with wwrblasSetStream/wwrsolverDnSetStream and pass them in) and reports
 * the converged pairs through out-pointers -- eigenvectors into @p
 * eigenvectors_out, eigenvalues and the converged flag into @p result. A Status
 * failure means a genuine BLAS/solver/runtime fault; whether failing to converge
 * is ALSO surfaced as a non-success Status is the caller's choice, through
 * DavidsonOptions::fail_on_non_convergence.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
// Resolved root-relative via the src/ root calaman.error_handling exports; needs
// calaman::Status visible at expansion, which the export import below supplies.
#include "error_handling/error_macros.h"

export module calaman.davidson:solve;

import std;
import wwr.blas;            // wwrblasHandle_t, pointer-mode get/set, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.wrappers.common; // real_fp
import wwr.wrappers.blas;   // gemm, axpy, dot, nrm2, scal
import wwr.wrappers.solver; // syevd
import :buffer_size;        // DavidsonSlices
import calaman.common;      // kOne, kZero, kNegativeOne
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/// @brief Convergence/iteration knobs for one davidson_solve call.
template<wwr::real_fp T>
struct DavidsonOptions {
  /// @brief A root is converged when ||A X_i - theta_i X_i|| is at or below this.
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
template<wwr::real_fp T>
struct DavidsonResult {
  bool converged = false;
  int iterations = 0;
  /// @brief The n_roots lowest eigenvalues, ascending (host). Filled on every
  ///        completed run (its accuracy is only guaranteed when converged).
  std::vector<T> eigenvalues;
};

/// @brief sigma(stream, block_size, b, sigma_out): out[:, :block_size] =
///        A b[:, :block_size], device-resident, n x block_size column-major (ld n).
template<wwr::real_fp T>
using DavidsonSigmaFn =
    std::function<Status(wwr::wwrStream_t stream, int block_size, const T *b, T *sigma_out)>;

/// @brief precondition(stream, n_roots, theta, residual, correction): turn the
///        residual block (n x n_roots, device) into a correction block (same
///        shape, device), given the current Ritz values @p theta (n_roots, HOST).
template<wwr::real_fp T>
using DavidsonPreconditionFn = std::function<Status(wwr::wwrStream_t stream, int n_roots,
                                                    const T *theta, const T *residual,
                                                    T *correction)>;

/// @brief OPTIONAL metric(stream, block_size, b, m_out): apply the SPD metric M,
///        same shape as DavidsonSigmaFn. Empty (default) selects the Euclidean
///        path. NOT YET IMPLEMENTED: a non-empty metric is rejected.
template<wwr::real_fp T>
using DavidsonMetricFn =
    std::function<Status(wwr::wwrStream_t stream, int block_size, const T *b, T *m_out)>;

} // namespace calaman

namespace calaman::davidson_detail {

/// Host pointer mode for a scope, restoring whatever the caller had set -- the
/// level-1 reductions (dot/nrm2) and the scaled updates take HOST scalars and
/// write HOST results, so the handle must be in host pointer mode across the
/// solve. Mirrors calaman.feast's guard.
class HostPointerMode {
public:
  explicit HostPointerMode(wwr::wwrblasHandle_t handle) : handle_(handle) {
    ok_ = wwr::wwrblasGetPointerMode(handle_, &saved_) == wwr::WWRBLAS_STATUS_SUCCESS &&
          wwr::wwrblasSetPointerMode(handle_, wwr::WWRBLAS_POINTER_MODE_HOST) ==
              wwr::WWRBLAS_STATUS_SUCCESS;
  }
  ~HostPointerMode() {
    if (ok_) {
      wwr::wwrblasSetPointerMode(handle_, saved_);
    }
  }
  HostPointerMode(const HostPointerMode &) = delete;
  HostPointerMode &operator=(const HostPointerMode &) = delete;
  [[nodiscard]] bool ok() const { return ok_; }

private:
  wwr::wwrblasHandle_t handle_;
  wwr::wwrblasPointerMode_t saved_ = wwr::WWRBLAS_POINTER_MODE_HOST;
  bool ok_ = false;
};

} // namespace calaman::davidson_detail

export namespace calaman {

/**
 * @brief Converge the lowest @p n_roots eigenpairs of a symmetric operator known
 *        only through @p sigma, from an orthonormal initial @p guess.
 *
 * @param stream          Stream both handles are bound to; must outlive the call.
 * @param n,n_roots,max_subspace  The shape @p s was carved for.
 * @param guess           n x guess_count device, column-major (ld n), ORTHONORMAL.
 * @param guess_count     n_roots <= guess_count <= max_subspace.
 * @param s               Workspace from make_davidson_slices (Euclidean sizing).
 * @param eigenvectors_out  Out: converged Ritz vectors, n x n_roots device (ld n);
 *                        also the per-iteration Ritz-vector scratch.
 * @param result          Out (host): eigenvalues, iteration count, converged flag.
 * @param metric          OPTIONAL; must be empty (metric path not yet implemented).
 * @return success on a converged solve; WWRBLAS_STATUS_INTERNAL_ERROR on a
 *         non-converged one when options.fail_on_non_convergence (result still
 *         filled); INVALID_VALUE for a bad guess_count; NOT_SUPPORTED for a
 *         non-empty metric; or a propagated BLAS/solver/runtime failure.
 */
template<wwr::real_fp T>
Status davidson_solve(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                      wwr::wwrStream_t stream, int n, int n_roots, int max_subspace, const T *guess,
                      int guess_count, const DavidsonSlices<T> &s, const DavidsonSigmaFn<T> &sigma,
                      const DavidsonPreconditionFn<T> &precondition, T *eigenvectors_out,
                      DavidsonResult<T> *result, const DavidsonOptions<T> &options = {},
                      const DavidsonMetricFn<T> &metric = {}) {
  CLM_REQUIRE(result != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(guess_count >= n_roots && guess_count <= max_subspace,
              wwr::WWRBLAS_STATUS_INVALID_VALUE);
  // The generalized (metric) path lands in a follow-up; reject it rather than
  // silently ignoring M and solving the wrong (Euclidean) problem.
  CLM_REQUIRE(!metric, wwr::WWRBLAS_STATUS_NOT_SUPPORTED);

  const davidson_detail::HostPointerMode mode(cublas_handle);
  CLM_REQUIRE(mode.ok(), wwr::WWRBLAS_STATUS_INTERNAL_ERROR);

  const auto nz = static_cast<std::size_t>(n);
  const std::size_t n_bytes = sizeof(T) * nz;
  const std::size_t block_bytes = n_bytes * static_cast<std::size_t>(n_roots);

  std::vector<T> theta(static_cast<std::size_t>(n_roots));
  std::vector<T> residual_norms(static_cast<std::size_t>(n_roots));

  // result holds the best-effort answer for every exit, converged or not.
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

  for (int iter = 1; iter <= options.max_iterations; ++iter) {
    // 1. Extend Sigma_V for only the columns appended since the last iteration.
    if (dim > filled) {
      const int new_count = dim - filled;
      const std::size_t offset = static_cast<std::size_t>(filled) * nz;
      CLM_TRY(sigma(stream, new_count, s.v + offset, s.av + offset));
      filled = dim;
    }

    // 2. Rayleigh-Ritz: H = V^T Sigma_V (dim x dim, ld max_subspace), syevd.
    CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim, dim, n,
                               &kOne<T>, s.v, n, s.av, n, &kZero<T>, s.h, max_subspace)));
    CLM_TRY(wwr::syevd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                          wwr::WWRBLAS_FILL_MODE_LOWER, dim, s.h, max_subspace, s.ritz,
                          s.eig_scratch, s.lwork_eig, s.info));
    int info_host = 0;
    CLM_TRY(wwr::wwrMemcpyAsync(&info_host, s.info, sizeof(int), wwr::wwrMemcpyDeviceToHost,
                                stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    // info != 0 is a synthetic host-side verdict on syevd's devInfo, so (like
    // feast's own host-side conditions) it carries a BLAS-domain code -- the
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

    // 4. Residual R = A X - X diag(theta), then its per-root norm.
    CLM_TRY(wwr::wwrMemcpyAsync(s.residual, s.ritz_av, block_bytes, wwr::wwrMemcpyDeviceToDevice,
                                stream));
    bool all_converged = true;
    for (int i = 0; i < n_roots; ++i) {
      const std::size_t col = static_cast<std::size_t>(i) * nz;
      const T neg_theta = -theta[static_cast<std::size_t>(i)];
      CLM_TRY((wwr::axpy<T, int>(cublas_handle, n, &neg_theta, eigenvectors_out + col, 1,
                                 s.residual + col, 1)));
      T norm{};
      CLM_TRY((wwr::nrm2<T, int>(cublas_handle, n, s.residual + col, 1, &norm)));
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
    //    guarantees max_subspace >= 2 * n_roots, so this leaves room for one.
    if (dim + n_roots > max_subspace) {
      CLM_TRY(wwr::wwrMemcpyAsync(s.v, eigenvectors_out, block_bytes, wwr::wwrMemcpyDeviceToDevice,
                                  stream));
      CLM_TRY(wwr::wwrMemcpyAsync(s.av, s.ritz_av, block_bytes, wwr::wwrMemcpyDeviceToDevice,
                                  stream));
      dim = n_roots;
      filled = n_roots;
    }

    // 7. Modified Gram-Schmidt of the correction block against the retained
    //    subspace, twice (a single classical pass loses orthogonality here):
    //    proj = V^T C; C -= V proj.
    for (int pass = 0; pass < 2; ++pass) {
      CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, dim, n_roots,
                                 n, &kOne<T>, s.v, n, s.correction, n, &kZero<T>, s.proj,
                                 max_subspace)));
      CLM_TRY((wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n_roots,
                                 dim, &kNegativeOne<T>, s.v, n, s.proj, max_subspace, &kOne<T>,
                                 s.correction, n)));
    }

    // 8. Sequential Gram-Schmidt of the surviving candidates against each other,
    //    skipping locked roots and dropping a candidate with no component left
    //    outside the retained subspace.
    int kept = 0;
    for (int i = 0; i < n_roots; ++i) {
      if (residual_norms[static_cast<std::size_t>(i)] <= options.residual_tolerance) {
        continue;
      }
      T *candidate = s.correction + static_cast<std::size_t>(i) * nz;
      for (int j = 0; j < kept; ++j) {
        const T *accepted = s.v + static_cast<std::size_t>(dim + j) * nz;
        T projection{};
        CLM_TRY((wwr::dot<T, int>(cublas_handle, n, accepted, 1, candidate, 1, &projection)));
        const T neg_projection = -projection;
        CLM_TRY((wwr::axpy<T, int>(cublas_handle, n, &neg_projection, accepted, 1, candidate, 1)));
      }
      T norm{};
      CLM_TRY((wwr::nrm2<T, int>(cublas_handle, n, candidate, 1, &norm)));
      if (norm <= options.linear_dependence_floor) {
        continue;
      }
      const T inv_norm = T{1} / norm;
      CLM_TRY((wwr::scal<T, int>(cublas_handle, n, &inv_norm, candidate, 1)));
      CLM_TRY(wwr::wwrMemcpyAsync(s.v + static_cast<std::size_t>(dim + kept) * nz, candidate,
                                  n_bytes, wwr::wwrMemcpyDeviceToDevice, stream));
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
