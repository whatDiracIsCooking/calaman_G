/**
 * @file solve.cppm
 * @brief The block-Davidson solve entry point, its options/result types and its
 *        operator/preconditioner/metric callbacks
 *
 * The :solve partition of calaman.davidson.
 *
 * SKELETON. This unit pins the public interface -- the callback shapes, the
 * convergence knobs, the result, and davidson_solve's signature over the
 * DavidsonSlices workspace from :buffer_size -- so a downstream caller can be
 * written against it and the workspace layout is fixed. The iteration itself
 * (subspace expansion, Rayleigh-Ritz via syevd/sygvd, residual/locking,
 * collapse, twice-modified Gram-Schmidt) lands in a follow-up PR; until then
 * davidson_solve returns WWRBLAS_STATUS_NOT_SUPPORTED and writes no output.
 *
 * The design mirrors calaman.feast: the solver owns neither the handles nor the
 * stream (bind them with wwrblasSetStream/wwrsolverDnSetStream and pass them in),
 * returns a calaman::Status, and reports the converged pairs through out-pointers
 * -- the eigenvectors into @p eigenvectors_out and the eigenvalues/convergence
 * flag into @p result -- rather than a Result<T>. Non-convergence and subspace
 * stagnation are DATA in @p result (converged == false), not Status errors; a
 * Status failure means a genuine BLAS/solver/runtime fault.
 */

export module calaman.davidson:solve;

import std;
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_STATUS_NOT_SUPPORTED
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.runtime_api;     // wwrStream_t
import wwr.wrappers.common; // real_fp
import :buffer_size;        // DavidsonSlices
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/// @brief Convergence/iteration knobs for one davidson_solve call.
template<wwr::real_fp T>
struct DavidsonOptions {
  /// @brief A root is converged when ||A X_i - theta_i X_i|| (M-norm on the
  ///        metric path) is at or below this.
  T residual_tolerance = T{1e-8};
  /// @brief Subspace expansions before giving up.
  int max_iterations = 100;
  /// @brief A re-orthogonalized correction at or below this norm carries no
  ///        component outside the retained subspace and is dropped. Kept well
  ///        below residual_tolerance: a near-converged root's correction shrinks
  ///        with its residual, and a floor too close to the tolerance would drop
  ///        genuine (small) new directions as spurious linear dependence.
  T linear_dependence_floor = T{1e-12};
  /// @brief On exhausting max_iterations while still converging normally: true
  ///        (default) leaves converged == false for the caller to treat as it
  ///        likes; subspace stagnation is unaffected (always converged == false).
  bool fail_on_non_convergence = true;
};

/// @brief Outcome of a davidson_solve call.
template<wwr::real_fp T>
struct DavidsonResult {
  bool converged = false;
  int iterations = 0;
  /// @brief The n_roots lowest eigenvalues, ascending (host). Valid on success.
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
///        path; non-empty selects the generalized (sygvd) path and must match the
///        with_metric the slices/workspace were sized with.
template<wwr::real_fp T>
using DavidsonMetricFn =
    std::function<Status(wwr::wwrStream_t stream, int block_size, const T *b, T *m_out)>;

/**
 * @brief Converge the lowest @p n_roots eigenpairs of a symmetric operator known
 *        only through @p sigma, from an orthonormal initial @p guess.
 *
 * SKELETON: not yet implemented -- returns WWRBLAS_STATUS_NOT_SUPPORTED. The
 * signature and preconditions are final.
 *
 * @param stream          Stream both handles are bound to; must outlive the call.
 * @param n,n_roots,max_subspace  The shape @p s was carved for.
 * @param guess           n x guess_count device, column-major (ld n), ORTHONORMAL.
 * @param guess_count     n_roots <= guess_count <= max_subspace.
 * @param s               Workspace from make_davidson_slices (same with_metric).
 * @param eigenvectors_out  Out: converged Ritz vectors, n x n_roots device (ld n).
 * @param result          Out (host): eigenvalues, iteration count, converged flag.
 * @param metric          OPTIONAL; empty selects the Euclidean path.
 */
template<wwr::real_fp T>
Status davidson_solve([[maybe_unused]] wwr::wwrblasHandle_t cublas_handle,
                      [[maybe_unused]] wwr::wwrsolverDnHandle_t cusolver_handle,
                      [[maybe_unused]] wwr::wwrStream_t stream, [[maybe_unused]] int n,
                      [[maybe_unused]] int n_roots, [[maybe_unused]] int max_subspace,
                      [[maybe_unused]] const T *guess, [[maybe_unused]] int guess_count,
                      [[maybe_unused]] const DavidsonSlices<T> &s,
                      [[maybe_unused]] const DavidsonSigmaFn<T> &sigma,
                      [[maybe_unused]] const DavidsonPreconditionFn<T> &precondition,
                      [[maybe_unused]] T *eigenvectors_out,
                      [[maybe_unused]] DavidsonResult<T> *result,
                      [[maybe_unused]] const DavidsonOptions<T> &options = {},
                      [[maybe_unused]] const DavidsonMetricFn<T> &metric = {}) {
  return wwr::WWRBLAS_STATUS_NOT_SUPPORTED;
}

} // namespace calaman
