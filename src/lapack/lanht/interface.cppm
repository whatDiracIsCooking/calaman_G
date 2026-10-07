/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lanht -- the norm of a complex Hermitian
 *        tridiagonal matrix, LAPACK's ?lanht
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm (equal to the
 * 1-norm, by symmetry) or Frobenius norm of the Hermitian tridiagonal with real
 * diagonal d and complex off-diagonal e, selected by a MatrixNorm. The result is
 * written to a real device scalar and the call returns WITHOUT synchronizing;
 * n == 0 writes 0, as ZLANHT returns 0. Allocates nothing: one single-block
 * launch.
 *
 * The kernel is calaman.lanst's (lapack/lanst/lanst_bridge.h), templated on the
 * off-diagonal type -- |e| is the complex modulus, the only difference.
 *
 * Mapping from ZLANHT (docs/architecture.md §4): NORM becomes MatrixNorm; c/z
 * become one template over T, d and the norm ComplexToRealType<T>. Divergence:
 * the Frobenius norm is a plain sum of squares, not ZLANHT's scaled ?lassq --
 * the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lanht;  // also re-exports MatrixNorm and Status
 *   calaman::lanht<wwr::wwrDoubleComplex>(stream, calaman::MatrixNorm::one, n, d_d, d_e,
 *                                         d_result);
 */

module;

// CLM_TRY's header is not needed: the body has one fallible step, returned
// directly. lanst_bridge.h declares the .cu launcher in the GMF.
#include "lapack/lanst/lanst_bridge.h"

export module calaman.lanht;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // MatrixNorm (:enums), ComplexToRealType

// export import: lanht RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lanst does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lanht;` alone names calaman::MatrixNorm.
export using calaman::MatrixNorm;

/// @brief Norm of the n-by-n Hermitian tridiagonal (d, e) on @p stream (?lanht)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries a launch or memset failure.
///
/// @tparam T Off-diagonal type; wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param n        Order of the tridiagonal
/// @param d_d      Device real diagonal, length n
/// @param d_e      Device complex off-diagonal, length n-1 (unread when n <= 1)
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the memset/launch error
export template<typename T>
Status lanht(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
             const ComplexToRealType<T> *const d_d, const T *const d_e,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  device::lanst<T, R>(stream, which, n, d_d, d_e, d_result);
  return wwr::wwrGetLastError();
}

extern template Status lanht<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   const float *, const wwr::wwrFloatComplex *,
                                                   float *);
extern template Status lanht<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                    const double *,
                                                    const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
