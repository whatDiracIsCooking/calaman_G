/**
 * @file interface.cppm
 * @brief Primary interface for calaman.sterf -- eigenvalues of a symmetric
 *        tridiagonal matrix, LAPACK's ?sterf
 *
 * Overwrites the device diagonal @p d with the eigenvalues, ascending, of the
 * symmetric tridiagonal (d, e) by the root-free Pal-Walker-Kahan QL/QR
 * iteration; @p e is destroyed. INFO is a device int, as calaman.lahqr's: 0, or
 * the count of off-diagonals not yet zero after 30*n sweeps (d then unsorted).
 * Enqueued on the stream; returns WITHOUT synchronizing. Allocates nothing: one
 * single-thread kernel. Real only (float, double), as the reference.
 *
 * Mapping from DSTERF (docs/architecture.md §4): s/d become one template over
 * T; INFO = -1 (N < 0) becomes an invalid-value Status. A kernel that needs the
 * same work in-thread #includes "lapack/sterf/sterf.cuh" (sterf_serial).
 *
 * Usage:
 *   import calaman.sterf;     // also re-exports calaman::Status
 *   // d_d: n device diagonal, d_e: n-1 device off-diagonal, d_info: device int
 *   calaman::sterf<double>(stream, n, d_d, d_e, d_info);
 */

module;

// CLM_TRY / CLM_REQUIRE arrive by #include in the GMF; sterf_bridge.h declares
// the .cu launcher there too.
#include "error_handling/error_macros.h"

#include "sterf_bridge.h"

export module calaman.sterf;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError, wwrSuccess
import calaman.common;  // real_fp

// export import: sterf RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lahqr does.
export import calaman.error_handling;

namespace calaman {

/// @brief Eigenvalues of the order-@p n symmetric tridiagonal (d, e) (?sterf)
///
/// Enqueues the iteration on @p stream and returns without synchronizing.
/// Writes the ascending eigenvalues to @p d and INFO to the device int @p info.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param n Order of the tridiagonal; n < 0 returns an invalid-value Status
/// @param d Device diagonal, length n; overwritten by the eigenvalues
/// @param e Device off-diagonal, length n-1 (unread when n <= 1); destroyed
/// @param info Device int; 0, or the count of unconverged off-diagonals
/// @return Success, or the runtime error the kernel launch reported
export template<calaman::real_fp T>
Status sterf(const wwr::wwrStream_t stream, const int n, T *const d, T *const e, int *const info) {
  CLM_REQUIRE(n >= 0, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(info != nullptr, wwr::wwrErrorInvalidValue);
  device::sterf(stream, n, d, e, info);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status sterf<float>(wwr::wwrStream_t, int, float *, float *, int *);
extern template Status sterf<double>(wwr::wwrStream_t, int, double *, double *, int *);

} // namespace calaman
