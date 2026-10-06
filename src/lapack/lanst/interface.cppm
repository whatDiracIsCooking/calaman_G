/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lanst -- the norm of a symmetric
 *        tridiagonal matrix, LAPACK's ?lanst
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm or Frobenius norm of
 * the symmetric tridiagonal with diagonal d and off-diagonal e, selected by a
 * MatrixNorm (as ?lanst selects on its NORM char). The result is written to a
 * device scalar and the call returns WITHOUT synchronizing; n == 0 writes 0, as
 * DLANST returns 0. Allocates nothing: one single-block launch (lanst.cu).
 *
 * Mapping from DLANST (docs/architecture.md §4): NORM becomes MatrixNorm; s/d
 * become one template over T. Divergence: the Frobenius norm is a plain sum of
 * squares, not DLANST's scaled ?lassq -- correct for well-scaled inputs, the
 * calaman.lange precedent.
 *
 * A kernel that needs lanst('M') of a sub-block (?sterf, ?steqr) does not call
 * this: it #includes "lapack/lanst/lanst.h" for the per-thread lanst_max_abs.
 *
 * Usage:
 *   import calaman.lanst;      // also re-exports calaman::MatrixNorm and Status
 *   // d_d: n device diagonal, d_e: n-1 device off-diagonal, d_result: scalar
 *   calaman::lanst<double>(stream, calaman::MatrixNorm::one, n, d_d, d_e, d_result);
 */

module;

// CLM_TRY's header is not needed: the body has one fallible step, returned
// directly. lanst_bridge.h declares the .cu launcher in the GMF.
#include "lanst_bridge.h"

export module calaman.lanst;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import calaman.common;  // MatrixNorm (:enums)

// export import: lanst RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lange does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lanst;` alone names calaman::MatrixNorm.
export using calaman::MatrixNorm;

/// @brief Norm of the n-by-n symmetric tridiagonal (d, e) on @p stream (?lanst)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries a launch or memset failure.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param n        Order of the tridiagonal
/// @param d_d      Device diagonal, length n
/// @param d_e      Device off-diagonal, length n-1 (unread when n <= 1)
/// @param d_result Device scalar receiving the norm
/// @return Success, or the memset/launch error
export template<typename T>
Status lanst(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
             const T *const d_d, const T *const d_e, T *const d_result) {
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(T), stream);
  }
  device::lanst<T>(stream, which, n, d_d, d_e, d_result);
  return wwr::wwrGetLastError();
}

extern template Status lanst<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                                    const float *, float *);
extern template Status lanst<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                                     const double *, double *);

} // namespace calaman
