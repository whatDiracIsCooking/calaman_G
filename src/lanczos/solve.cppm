/**
 * @file solve.cppm
 * @brief The thick-restart Lanczos solve entry point
 *
 * The :solve partition of calaman.lanczos. lanczos_solve converges the nev
 * extreme eigenpairs of a symmetric operator known only through a single-vector
 * matvec callback: Lanczos steps with full CGS2 reorthogonalization against V,
 * the projected problem diagonalised by syevd, the :ritz stages, and a thick
 * restart (Wu-Simon) that keeps k = lanczos_restart_keep(nev, ncv) Ritz pairs.
 *
 * The cycle stages are module-internal (calaman::detail). One host sync per
 * cycle, plus one per breakdown recovery and one for the start vector's norm.
 * See README.md for the restart and the breakdown recovery.
 *
 * Like calaman.davidson, the solver owns neither the handles nor the stream (bind
 * both handles to @p stream before the call).
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF; they
// need calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"
#include "lanczos_bridge.h"

#include <cstddef>

export module calaman.lanczos:solve;

import std;
import wwr.blas;                    // wwrblasHandle_t, wwrblasSetPointerMode, WWRBLAS_*
import wwr.solver;                  // wwrsolverDnHandle_t
import wwr.runtime_api;             // wwrStream_t, wwrMemcpyAsync, wwrMemsetAsync
import wwr.wrappers.blas;           // gemv, axpy, nrm2, scal
import wwr.extension.blas;          // ScopedPointerMode
import wwr.extension.init_state;    // init_state
import wwr.extension.random_normal; // random_normal
import :buffer_size;                // LanczosSlices, lanczos_shape_ok, lanczos_restart_keep
import :ritz;                       // LanczosRitz, lanczos_ritz_*
import :types;                      // LanczosWhich, LanczosOptions, LanczosResult, lanczos_matvec
import calaman.common;              // kOne, kZero, kNegativeOne, real_fp
export import calaman.error_handling; // Status, PointerModeStatus

// Module-internal cycle stages: not exported, reached by the restart loop.
namespace calaman::detail {

/// @brief v = v / ||v||_2 in HOST pointer mode (one sync).
/// @return @p bad_norm for a zero or non-finite norm.
template<calaman::real_fp T>
Status lanczos_normalise(wwr::wwrblasHandle_t blas_handle, const int n, T *v,
                         const Status bad_norm) {
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);
  T norm{};
  CLM_TRY((wwr::nrm2<T, int>(blas_handle, n, v, 1, &norm)));
  if (!(norm > T{0} && std::isfinite(norm))) {
    return bad_norm;
  }
  const T inv_norm = T{1} / norm;
  CLM_TRY((wwr::scal<T, int>(blas_handle, n, &inv_norm, v, 1)));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Seed s.rng from options.seed, then v_0 = start / ||start|| (the
///        caller's options.start_vector, else a random normal draw). One sync.
/// @return INVALID_VALUE for a start vector of zero or non-finite norm.
template<calaman::real_fp T>
Status lanczos_start(wwr::wwrblasHandle_t blas_handle, wwr::wwrStream_t stream, const int n,
                     const LanczosSlices<T> &s, const LanczosOptions<T> &options) {
  const auto nz = static_cast<std::size_t>(n);
  wwr::extension::init_state(stream, nz, s.rng, options.seed);
  if (options.start_vector != nullptr) {
    CLM_TRY(wwr::wwrMemcpyAsync(s.v, options.start_vector, sizeof(T) * nz,
                                wwr::wwrMemcpyDeviceToDevice, stream));
  } else {
    wwr::extension::random_normal<T>(stream, nz, s.rng, s.v);
  }
  // Both launchers return void: the runtime's sticky error is the only report.
  CLM_TRY(wwr::wwrGetLastError());
  return lanczos_normalise<T>(blas_handle, n, s.v, wwr::WWRBLAS_STATUS_INVALID_VALUE);
}

/// @brief Breakdown recovery: V(:, col) = a fresh random normal draw, CGS2'd
///        against V(:, 0:col) and normalised. One sync; needs col < n.
/// @return INTERNAL_ERROR if the draw vanishes against V (not seen for col < n).
template<calaman::real_fp T>
Status lanczos_inject(wwr::wwrblasHandle_t blas_handle, wwr::wwrStream_t stream, const int n,
                      const int col, const LanczosSlices<T> &s) {
  const auto nz = static_cast<std::size_t>(n);
  T *v = s.v + static_cast<std::size_t>(col) * nz;
  wwr::extension::random_normal<T>(stream, nz, s.rng, v);
  CLM_TRY(wwr::wwrGetLastError());
  {
    wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
    const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                 PointerModeStatus{&pm_status}};
    CLM_TRY(pm_status);
    for (int pass = 0; pass < 2 && col > 0; ++pass) {
      CLM_TRY((wwr::gemv<T, int>(blas_handle, wwr::WWRBLAS_OP_T, n, col, &kOne<T>, s.v, n, v, 1,
                                 &kZero<T>, s.coeffs, 1)));
      CLM_TRY((wwr::gemv<T, int>(blas_handle, wwr::WWRBLAS_OP_N, n, col, &kNegativeOne<T>, s.v, n,
                                 s.coeffs, 1, &kOne<T>, v, 1)));
    }
  }
  return lanczos_normalise<T>(blas_handle, n, v, wwr::WWRBLAS_STATUS_INTERNAL_ERROR);
}

/**
 * @brief Lanczos steps @p first..ncv-1 of one cycle, after resetting its status
 *        block. No host sync; @p matvec runs with the handle in HOST mode.
 *
 * Needs V(:, 0:first+1) orthonormal and T's columns/rows 0..first-1 written
 * (a fresh cycle: first = 0; after a restart: the arrowhead, first = k).
 * @param matvecs In/out: incremented once per @p matvec call.
 */
template<calaman::real_fp T, lanczos_matvec<T> Matvec>
Status lanczos_extend(wwr::wwrblasHandle_t blas_handle, wwr::wwrStream_t stream, const int n,
                      const int ncv, const int first, const LanczosSlices<T> &s,
                      const Matvec &matvec, int *matvecs) {
  const auto nz = static_cast<std::size_t>(n);
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  device::lanczos_status_reset(stream, s.status);
  for (int j = first; j < ncv; ++j) {
    const auto jz = static_cast<std::size_t>(j);
    T *w = s.v + (jz + 1) * nz;
    CLM_TRY(matvec(stream, s.v + jz * nz, w));
    ++*matvecs;

    // CGS2 against V(:, 0:j+1): h = V^T w, w -= V h, twice. T(j,j) = alpha_j is
    // the total coefficient on v_j, h1[j] + h2[j] (h1[j] = v_j^T w is the plain
    // Lanczos alpha; h2[j] is its roundoff-level correction).
    for (int pass = 0; pass < 2; ++pass) {
      CLM_TRY((wwr::gemv<T, int>(blas_handle, wwr::WWRBLAS_OP_T, n, j + 1, &kOne<T>, s.v, n, w, 1,
                                 &kZero<T>, s.coeffs, 1)));
      CLM_TRY((wwr::gemv<T, int>(blas_handle, wwr::WWRBLAS_OP_N, n, j + 1, &kNegativeOne<T>, s.v, n,
                                 s.coeffs, 1, &kOne<T>, w, 1)));
      if (pass == 0) {
        CLM_TRY(wwr::wwrMemcpyAsync(s.alpha + jz, s.coeffs + jz, sizeof(T),
                                    wwr::wwrMemcpyDeviceToDevice, stream));
      } else {
        CLM_TRY((wwr::axpy<T, int>(blas_handle, 1, &kOne<T>, s.coeffs + jz, 1, s.alpha + jz, 1)));
      }
    }

    // beta_j = ||w|| lands on the device; the outer scope restores the caller's
    // mode if this returns early.
    CLM_TRY(wwr::wwrblasSetPointerMode(blas_handle, wwr::WWRBLAS_POINTER_MODE_DEVICE));
    CLM_TRY((wwr::nrm2<T, int>(blas_handle, n, w, 1, s.beta + jz)));
    CLM_TRY(wwr::wwrblasSetPointerMode(blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST));

    device::lanczos_step<T>(stream, n, ncv, j, s.alpha, s.beta, w, s.t, s.status);
    CLM_TRY(wwr::wwrGetLastError());
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief The cycle's Ritz snapshot, recovering from early breakdowns: while the
 *        first tripped step j is below ncv - 1, V(:, 0:j+1) is invariant, so
 *        inject a fresh v_{j+1} (T's coupling there is already 0), re-run steps
 *        j+1.. and re-extract. On return any breakdown is at the last step.
 */
template<calaman::real_fp T, lanczos_matvec<T> Matvec>
Status lanczos_cycle_ritz(wwr::wwrblasHandle_t blas_handle, wwr::wwrsolverDnHandle_t solver_handle,
                          wwr::wwrStream_t stream, const int n, const int ncv,
                          const LanczosSlices<T> &s, const Matvec &matvec, int *matvecs,
                          LanczosRitz<T> *ritz) {
  CLM_TRY(lanczos_ritz_extract<T>(solver_handle, stream, ncv, s, ritz));
  while (ritz->breakdown && ritz->breakdown_step < ncv - 1) {
    const int next = ritz->breakdown_step + 1;
    CLM_TRY(lanczos_inject<T>(blas_handle, stream, n, next, s));
    CLM_TRY(lanczos_extend<T>(blas_handle, stream, n, ncv, next, s, matvec, matvecs));
    CLM_TRY(lanczos_ritz_extract<T>(solver_handle, stream, ncv, s, ritz));
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Thick-restart the basis on the @p kept pairs (a k-selection's index):
 *        V(:, 0:k) = V S_k (staged in s.keep), V(:, k) = v_ncv, and the
 *        arrowhead into T, so lanczos_extend(first = k) resumes. After a
 *        last-step breakdown v_ncv is void: the coupling is 0, V(:, k) fresh.
 */
template<calaman::real_fp T>
Status lanczos_restart_basis(wwr::wwrblasHandle_t blas_handle, wwr::wwrStream_t stream, const int n,
                             const int ncv, const std::vector<int> &kept, const bool breakdown,
                             const LanczosSlices<T> &s) {
  const auto nz = static_cast<std::size_t>(n);
  const int k = static_cast<int>(kept.size());
  const auto kz = static_cast<std::size_t>(k);
  CLM_TRY(lanczos_ritz_compact<T>(stream, ncv, kept, s));
  CLM_TRY(lanczos_ritz_vectors<T>(blas_handle, n, ncv, k, s, s.keep, n));
  CLM_TRY(
      wwr::wwrMemcpyAsync(s.v, s.keep, sizeof(T) * nz * kz, wwr::wwrMemcpyDeviceToDevice, stream));
  T *beta_m = s.beta + (ncv - 1);
  if (breakdown) {
    CLM_TRY(wwr::wwrMemsetAsync(beta_m, 0, sizeof(T), stream));
    CLM_TRY(lanczos_inject<T>(blas_handle, stream, n, k, s));
  } else {
    CLM_TRY(wwr::wwrMemcpyAsync(s.v + kz * nz, s.v + static_cast<std::size_t>(ncv) * nz,
                                sizeof(T) * nz, wwr::wwrMemcpyDeviceToDevice, stream));
  }
  device::lanczos_arrowhead<T>(stream, ncv, k, s.theta, s.s + (ncv - 1), ncv, beta_m, s.t);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief The true-residual check: ||A x_j - theta_j x_j||_2 <= tolerance *
 *        max(|theta_j|, t_norm) for x_j = V(:, @p cols[j]), one matvec each
 *        into the spare V(:, ncv) and one sync. Call after lanczos_restart_basis.
 */
template<calaman::real_fp T, lanczos_matvec<T> Matvec>
Status lanczos_true_residuals(wwr::wwrblasHandle_t blas_handle, wwr::wwrStream_t stream,
                              const int n, const int ncv, const std::vector<int> &cols,
                              const std::vector<T> &theta, const T bound_scale, const T t_norm,
                              const LanczosSlices<T> &s, const Matvec &matvec, int *matvecs,
                              bool *all_ok) {
  const auto nz = static_cast<std::size_t>(n);
  T *w = s.v + static_cast<std::size_t>(ncv) * nz;
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);
  for (std::size_t j = 0; j < cols.size(); ++j) {
    const T *x = s.v + static_cast<std::size_t>(cols[j]) * nz;
    CLM_TRY(matvec(stream, x, w));
    ++*matvecs;
    const T minus_theta = -theta[j];
    CLM_TRY((wwr::axpy<T, int>(blas_handle, n, &minus_theta, x, 1, w, 1)));
    CLM_TRY(wwr::wwrblasSetPointerMode(blas_handle, wwr::WWRBLAS_POINTER_MODE_DEVICE));
    CLM_TRY((wwr::nrm2<T, int>(blas_handle, n, w, 1, s.coeffs + j)));
    CLM_TRY(wwr::wwrblasSetPointerMode(blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST));
  }
  std::vector<T> norms(cols.size());
  CLM_TRY(wwr::wwrMemcpyAsync(norms.data(), s.coeffs, sizeof(T) * norms.size(),
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  *all_ok = true;
  for (std::size_t j = 0; j < norms.size(); ++j) {
    *all_ok = *all_ok && norms[j] <= bound_scale * std::max(std::abs(theta[j]), t_norm);
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman::detail

export namespace calaman {

/**
 * @brief Converge the @p nev eigenpairs at the @p which end(s) of the spectrum of
 *        the symmetric operator behind @p matvec (called in HOST pointer mode),
 *        thick-restarting up to options.max_restarts times.
 *
 * @param n,nev,ncv         The shape @p s was carved for (make_lanczos_slices).
 * @param eigenvectors_out  Out: Ritz vectors, n x nev device (ld n), in the order
 *                          of result->eigenvalues, written on @p stream; null to skip.
 * @param result            Out (host): eigenvalues, counts, converged flag.
 * @return INVALID_VALUE for a rejected shape, a null @p result or a zero start
 *         vector; INTERNAL_ERROR on non-convergence when
 *         options.fail_on_non_convergence; else a propagated fault or success.
 */
template<calaman::real_fp T, lanczos_matvec<T> Matvec>
Status lanczos_solve(wwr::wwrblasHandle_t blas_handle, wwr::wwrsolverDnHandle_t solver_handle,
                     wwr::wwrStream_t stream, const int n, const int nev, const int ncv,
                     const LanczosWhich which, const LanczosSlices<T> &s, const Matvec &matvec,
                     T *eigenvectors_out, LanczosResult<T> *result,
                     const LanczosOptions<T> &options = {}) {
  CLM_REQUIRE(result != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(lanczos_shape_ok(n, nev, ncv), wwr::WWRBLAS_STATUS_INVALID_VALUE);

  result->converged = false;
  result->restarts = 0;
  result->matvecs = 0;
  result->eigenvalues.clear();

  const int k = lanczos_restart_keep(nev, ncv);
  CLM_TRY(detail::lanczos_start<T>(blas_handle, stream, n, s, options));
  CLM_TRY(detail::lanczos_extend<T>(blas_handle, stream, n, ncv, 0, s, matvec, &result->matvecs));

  LanczosRitz<T> ritz;
  LanczosRitzSelection<T> sel;
  std::vector<int> cols(static_cast<std::size_t>(nev)); // the wanted pairs' columns of V(:, 0:k)
  for (;;) {
    CLM_TRY(detail::lanczos_cycle_ritz<T>(blas_handle, solver_handle, stream, n, ncv, s, matvec,
                                          &result->matvecs, &ritz));
    // Selections nest, so the wanted pairs sit inside the kept ones; their
    // estimates must be read before the restart compacts S.
    sel = lanczos_ritz_select(ritz, which, nev, options.tolerance);
    const std::vector<int> kept = lanczos_select(which, ncv, k);
    for (std::size_t j = 0; j < cols.size(); ++j) {
      cols[j] = static_cast<int>(std::ranges::lower_bound(kept, sel.index[j]) - kept.begin());
    }

    CLM_TRY(detail::lanczos_restart_basis<T>(blas_handle, stream, n, ncv, kept, ritz.breakdown, s));
    bool converged = sel.all_converged();
    if (converged && options.verify_residuals) {
      CLM_TRY(detail::lanczos_true_residuals<T>(blas_handle, stream, n, ncv, cols, sel.values,
                                                options.tolerance, ritz.t_norm, s, matvec,
                                                &result->matvecs, &converged));
    }
    if (converged || result->restarts >= options.max_restarts) {
      result->converged = converged;
      break;
    }
    ++result->restarts;
    CLM_TRY(detail::lanczos_extend<T>(blas_handle, stream, n, ncv, k, s, matvec, &result->matvecs));
  }
  result->eigenvalues = sel.values;

  if (eigenvectors_out != nullptr) {
    const auto nz = static_cast<std::size_t>(n);
    for (std::size_t j = 0; j < cols.size(); ++j) {
      CLM_TRY(wwr::wwrMemcpyAsync(eigenvectors_out + j * nz,
                                  s.v + static_cast<std::size_t>(cols[j]) * nz, sizeof(T) * nz,
                                  wwr::wwrMemcpyDeviceToDevice, stream));
    }
  }
  if (!result->converged && options.fail_on_non_convergence) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
