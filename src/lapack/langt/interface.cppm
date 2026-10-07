/**
 * @file interface.cppm
 * @brief Primary interface for calaman.langt -- the norm of a general
 *        tridiagonal matrix, LAPACK's ?langt
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm or Frobenius norm
 * of the n-by-n tridiagonal with sub-diagonal dl, diagonal d and super-diagonal
 * du, selected by a MatrixNorm. Unlike ?lanst the 1- and infinity-norms differ:
 * a column sums du(i-1), d(i), dl(i); a row sums dl(i-1), d(i), du(i). The
 * result is written to a real device scalar and the call returns WITHOUT
 * synchronizing; n == 0 writes 0, as DLANGT returns 0. Allocates nothing: one
 * single-block launch (langt.cu).
 *
 * Mapping from DLANGT (docs/architecture.md §4): NORM becomes MatrixNorm;
 * s/d/c/z become one template over T, whose norm is ComplexToRealType<T>.
 * Divergence: the Frobenius norm is a plain sum of squares, not DLANGT's scaled
 * ?lassq -- the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.langt;  // also re-exports MatrixNorm and Status
 *   calaman::langt<double>(stream, calaman::MatrixNorm::one, n, d_dl, d_d, d_du, d_result);
 */

module;

// CLM_TRY's header is not needed: the body has one fallible step, returned
// directly. langt_bridge.h declares the .cu launcher in the GMF.
#include "langt_bridge.h"

export module calaman.langt;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm (:enums), ComplexToRealType

// export import: langt RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lange does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.langt;` alone names calaman::MatrixNorm.
export using calaman::MatrixNorm;

/// @brief Norm of the n-by-n tridiagonal (dl, d, du) on @p stream (?langt)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries a launch or memset failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param n        Order of the tridiagonal
/// @param d_dl     Device sub-diagonal, length n-1 (unread when n <= 1)
/// @param d_d      Device diagonal, length n
/// @param d_du     Device super-diagonal, length n-1 (unread when n <= 1)
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the memset/launch error
export template<typename T>
Status langt(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
             const T *const d_dl, const T *const d_d, const T *const d_du,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  device::langt<T, R>(stream, which, n, d_dl, d_d, d_du, d_result);
  return wwr::wwrGetLastError();
}

extern template Status langt<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                                    const float *, const float *, float *);
extern template Status langt<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                                     const double *, const double *, double *);
extern template Status langt<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   const wwr::wwrFloatComplex *,
                                                   const wwr::wwrFloatComplex *,
                                                   const wwr::wwrFloatComplex *, float *);
extern template Status langt<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                    const wwr::wwrDoubleComplex *,
                                                    const wwr::wwrDoubleComplex *,
                                                    const wwr::wwrDoubleComplex *, double *);

} // namespace calaman
