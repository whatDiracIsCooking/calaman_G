/**
 * @file solve.cppm
 * @brief The thick-restart Lanczos solve entry point
 *
 * The :solve partition of calaman.lanczos. lanczos_solve converges the nev
 * extreme eigenpairs of a symmetric operator known only through a single-vector
 * matvec callback: Lanczos steps with full CGS2 reorthogonalization against V,
 * the projected problem diagonalised by syevd, and the :ritz stages.
 *
 * Status: a single cycle of ncv steps, no restart yet (see README.md). The cycle
 * is split into module-internal stages -- lanczos_start, lanczos_extend (steps
 * first..ncv-1, so a restart can resume from step k) and lanczos_cycle_ritz --
 * that the restart loop reuses. One host sync per cycle, plus one for the start
 * vector's norm (and one more after an early breakdown).
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
import wwr.runtime_api;             // wwrStream_t, wwrMemcpyAsync, wwrMemcpy2DAsync
import wwr.wrappers.blas;           // gemv, axpy, nrm2, scal
import wwr.extension.blas;          // ScopedPointerMode
import wwr.extension.init_state;    // init_state
import wwr.extension.random_normal; // random_normal
import :buffer_size;                // LanczosSlices, lanczos_shape_ok
import :ritz;                       // LanczosRitz, lanczos_ritz_*
import :types;                      // LanczosWhich, LanczosOptions, LanczosResult, LanczosMatvecFn
import calaman.common;              // kOne, kZero, kNegativeOne, real_fp
export import calaman.error_handling; // Status, PointerModeStatus

// Module-internal cycle stages: not exported, reached by the restart loop.
namespace calaman::detail {

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

  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{blas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);
  T norm{};
  CLM_TRY((wwr::nrm2<T, int>(blas_handle, n, s.v, 1, &norm)));
  CLM_REQUIRE(norm > T{0} && std::isfinite(norm), wwr::WWRBLAS_STATUS_INVALID_VALUE);
  const T inv_norm = T{1} / norm;
  CLM_TRY((wwr::scal<T, int>(blas_handle, n, &inv_norm, s.v, 1)));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Lanczos steps @p first..ncv-1 of one cycle, after resetting its status
 *        block. No host sync; @p matvec runs with the handle in HOST mode.
 *
 * Needs V(:, 0:first+1) orthonormal and T's columns/rows 0..first-1 written
 * (a fresh cycle: first = 0; after a restart: the arrowhead, first = k).
 * @param matvecs In/out: incremented once per @p matvec call.
 */
template<calaman::real_fp T>
Status lanczos_extend(wwr::wwrblasHandle_t blas_handle, wwr::wwrStream_t stream, const int n,
                      const int ncv, const int first, const LanczosSlices<T> &s,
                      const LanczosMatvecFn<T> &matvec, int *matvecs) {
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
 * @brief The cycle's Ritz snapshot over its active dimension @p m: ncv, or
 *        breakdown_step + 1 when an earlier step broke down -- V(:, 0:m) then
 *        spans an invariant subspace, T's leading m x m block is repacked to
 *        ld m and re-extracted (a second sync), and the trailing steps are void.
 */
template<calaman::real_fp T>
Status lanczos_cycle_ritz(wwr::wwrsolverDnHandle_t solver_handle, wwr::wwrStream_t stream,
                          const int ncv, const LanczosSlices<T> &s, LanczosRitz<T> *ritz, int *m) {
  CLM_TRY(lanczos_ritz_extract<T>(solver_handle, stream, ncv, s, ritz));
  *m = ncv;
  if (!ritz->breakdown || ritz->breakdown_step >= ncv - 1) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  const int active = ritz->breakdown_step + 1;
  const auto az = static_cast<std::size_t>(active);
  // T(0:m, 0:m) from ld ncv to ld m, staged through s.s (in place would overlap).
  CLM_TRY(wwr::wwrMemcpy2DAsync(s.s, sizeof(T) * az, s.t, sizeof(T) * static_cast<std::size_t>(ncv),
                                sizeof(T) * az, az, wwr::wwrMemcpyDeviceToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(s.t, s.s, sizeof(T) * az * az, wwr::wwrMemcpyDeviceToDevice, stream));
  CLM_TRY(lanczos_ritz_extract<T>(solver_handle, stream, active, s, ritz));
  *m = active;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman::detail

export namespace calaman {

/**
 * @brief Converge the @p nev eigenpairs at the @p which end(s) of the spectrum of
 *        the symmetric operator behind @p matvec (called in HOST pointer mode).
 *
 * @param n,nev,ncv         The shape @p s was carved for (make_lanczos_slices).
 * @param eigenvectors_out  Out: Ritz vectors, n x nev device (ld n), in the order
 *                          of result->eigenvalues, written on @p stream; null to skip.
 * @param result            Out (host): eigenvalues, counts, converged flag.
 * @return INVALID_VALUE for a rejected shape, a null @p result, an empty @p matvec
 *         or a zero start vector; INTERNAL_ERROR on non-convergence when
 *         options.fail_on_non_convergence; else a propagated fault or success.
 */
template<calaman::real_fp T>
Status lanczos_solve(wwr::wwrblasHandle_t blas_handle, wwr::wwrsolverDnHandle_t solver_handle,
                     wwr::wwrStream_t stream, const int n, const int nev, const int ncv,
                     const LanczosWhich which, const LanczosSlices<T> &s,
                     const LanczosMatvecFn<T> &matvec, T *eigenvectors_out,
                     LanczosResult<T> *result, const LanczosOptions<T> &options = {}) {
  CLM_REQUIRE(result != nullptr, wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(static_cast<bool>(matvec), wwr::WWRBLAS_STATUS_INVALID_VALUE);
  CLM_REQUIRE(lanczos_shape_ok(n, nev, ncv), wwr::WWRBLAS_STATUS_INVALID_VALUE);

  result->converged = false;
  result->restarts = 0;
  result->matvecs = 0;
  result->eigenvalues.clear();

  CLM_TRY(detail::lanczos_start<T>(blas_handle, stream, n, s, options));
  CLM_TRY(detail::lanczos_extend<T>(blas_handle, stream, n, ncv, 0, s, matvec, &result->matvecs));
  LanczosRitz<T> ritz;
  int m = ncv;
  CLM_TRY(detail::lanczos_cycle_ritz<T>(solver_handle, stream, ncv, s, &ritz, &m));

  // An invariant subspace smaller than nev yields only m pairs: not converged.
  const int count = std::min(nev, m);
  const LanczosRitzSelection<T> sel = lanczos_ritz_select(ritz, which, count, options.tolerance);
  result->converged = count == nev && sel.all_converged();
  result->eigenvalues = sel.values;

  CLM_TRY(lanczos_ritz_compact<T>(stream, m, sel.index, s));
  if (eigenvectors_out != nullptr) {
    CLM_TRY(lanczos_ritz_vectors<T>(blas_handle, n, m, count, s, eigenvectors_out, n));
  }
  if (!result->converged && options.fail_on_non_convergence) {
    return wwr::WWRBLAS_STATUS_INTERNAL_ERROR;
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
