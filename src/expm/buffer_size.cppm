/**
 * @file buffer_size.cppm
 * @brief Device-workspace sizing for pade() and expm()
 *
 * The :buffer_size partition of calaman.expm. Both queries size the SAME carve
 * the routines run (PadeWorkspace / ExpmWorkspace, from :detail), so a query and
 * its routine cannot drift: pade_bufferSize sizes the chosen degree, while
 * expm_bufferSize reserves the ladder's worst case because the degree is picked
 * from the norm at run time.
 */

module;

#include "expm_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import supplies.
#include "error_handling/error_macros.h"

export module calaman.expm:buffer_size;

import std;
import wwr.blas;            // wwrblasStatus_t, WWRBLAS_* (invalid-value / success returns)
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex (the extern-template list)
import wwr.wrappers.common; // usual_fp, ComplexToRealType
import wwr.wrappers.solver; // getrf_bufferSize
import calaman.common;      // carve_workspace
import calaman.error_handling; // Status
import :detail;             // PadeWorkspace, ExpmWorkspace

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace.

/**
 * @brief Device workspace, in bytes, required by pade() at degree @p m.
 *
 * Grows with the degree -- three n*n blocks at m = 3, six at m = 9 and m = 13 --
 * so a caller who knows its matrices are small-normed can ask for far less than
 * expm_bufferSize reserves.
 *
 * @param handle      Solver handle (queried for the getrf workspace size).
 * @param m           Pade degree; one of 3, 5, 7, 9, 13.
 * @param n           Matrix dimension.
 * @param lwork_bytes Output: required workspace in bytes.
 * @return Status: SUCCESS, WWRBLAS_STATUS_INVALID_VALUE for a bad degree or
 *         dimension, or the solver-domain status if the getrf query fails.
 */
export template<wwr::usual_fp T>
Status pade_bufferSize(wwr::wwrsolverDnHandle_t handle, const int m, const int n,
                       std::size_t *lwork_bytes) {
  if (pade_coeffs(m) == nullptr || n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  CLM_TRY(wwr::getrf_bufferSize<T>(handle, n, n, static_cast<T *>(nullptr), n, &lwork_getrf));

  // Null base: size only, from the same carve pade() runs.
  *lwork_bytes = carve_workspace<PadeWorkspace<T>>(nullptr, nullptr, n, m, lwork_getrf);
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Device workspace, in bytes, required by expm().
 *
 * The degree is chosen at run time from the matrix norm, so this reserves the
 * ladder's worst case: one n*n block for the scaled matrix, the six degree 13
 * needs, and a handful of O(n) vectors. Seven n*n blocks in total.
 *
 * @param handle      Solver handle (queried for the getrf workspace size).
 * @param n           Matrix dimension.
 * @param lwork_bytes Output: required workspace in bytes.
 */
export template<wwr::usual_fp T>
Status expm_bufferSize(wwr::wwrsolverDnHandle_t handle, const int n, std::size_t *lwork_bytes) {
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  int lwork_getrf = 0;
  CLM_TRY(wwr::getrf_bufferSize<T>(handle, n, n, static_cast<T *>(nullptr), n, &lwork_getrf));

  // As + colsum, then the pade region sized for the WORST degree on the ladder:
  // run the same PadeWorkspace carve over every degree and take the largest, so
  // expm() can carve any chosen degree into the region that follows.
  std::size_t worst_pade = 0;
  for (int i = 0; i < kNumPadeDegrees; ++i) {
    worst_pade = std::max(worst_pade, carve_workspace<PadeWorkspace<T>>(
                                          nullptr, nullptr, n, kPadeDegrees[i], lwork_getrf));
  }

  *lwork_bytes = carve_workspace<ExpmWorkspace<T>>(nullptr, nullptr, n) + worst_pade;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

// Instantiated once in instantiations.cpp, so an importer never re-instantiates.
extern template Status pade_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, int, std::size_t *);
extern template Status pade_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, int, std::size_t *);
extern template Status pade_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int, int,
                                                             std::size_t *);
extern template Status pade_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int, int,
                                                              std::size_t *);

extern template Status expm_bufferSize<float>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
extern template Status expm_bufferSize<double>(wwr::wwrsolverDnHandle_t, int, std::size_t *);
extern template Status expm_bufferSize<wwr::wwrFloatComplex>(wwr::wwrsolverDnHandle_t, int,
                                                             std::size_t *);
extern template Status expm_bufferSize<wwr::wwrDoubleComplex>(wwr::wwrsolverDnHandle_t, int,
                                                              std::size_t *);

} // namespace calaman
