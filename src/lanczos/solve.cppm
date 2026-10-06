/**
 * @file solve.cppm
 * @brief The thick-restart Lanczos solve entry point
 *
 * The :solve partition of calaman.lanczos. lanczos_solve converges the nev
 * extreme eigenpairs of a symmetric operator known only through a single-vector
 * matvec callback: Lanczos steps with full CGS2 reorthogonalization against V,
 * the projected problem diagonalised by syevd, and Wu-Simon thick restart.
 *
 * Status: skeleton. The interface is pinned; after its argument checks the solve
 * returns WWRBLAS_STATUS_NOT_SUPPORTED. See README.md.
 *
 * Like calaman.davidson, the solver owns neither the handles nor the stream (bind
 * both handles to @p stream before the call).
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the GMF; it needs
// calaman::Status visible at expansion, which the export import supplies.
#include "error_handling/error_macros.h"

export module calaman.lanczos:solve;

import std;
import wwr.blas;        // wwrblasHandle_t, WWRBLAS_STATUS_*
import wwr.solver;      // wwrsolverDnHandle_t
import wwr.runtime_api; // wwrStream_t
import :buffer_size;    // LanczosSlices, lanczos_shape_ok
import :types;          // LanczosWhich, LanczosOptions, LanczosResult, LanczosMatvecFn
import calaman.common;  // real_fp
export import calaman.error_handling; // Status

export namespace calaman {

/**
 * @brief Converge the @p nev eigenpairs at the @p which end(s) of the spectrum of
 *        the symmetric operator behind @p matvec.
 *
 * @param n,nev,ncv         The shape @p s was carved for (make_lanczos_slices).
 * @param eigenvectors_out  Out: Ritz vectors, n x nev device (ld n), in the order
 *                          of result->eigenvalues; null to skip them.
 * @param result            Out (host): eigenvalues, counts, converged flag.
 * @return INVALID_VALUE for a rejected shape, a null @p result or an empty
 *         @p matvec; otherwise NOT_SUPPORTED until the solve lands.
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

  (void)blas_handle;
  (void)solver_handle;
  (void)stream;
  (void)which;
  (void)s;
  (void)eigenvectors_out;
  (void)options;

  result->converged = false;
  result->restarts = 0;
  result->matvecs = 0;
  result->eigenvalues.clear();
  return wwr::WWRBLAS_STATUS_NOT_SUPPORTED;
}

} // namespace calaman
