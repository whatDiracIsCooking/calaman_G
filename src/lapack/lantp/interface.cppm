/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lantp -- the norm of a triangular matrix
 *        in packed storage, LAPACK's ?lantp
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm or Frobenius norm
 * of the n-by-n upper or lower triangular A held in packed storage -- the
 * n(n+1)/2-element AP of calaman.trttp, the triangle column by column. With
 * Diag::U each diagonal entry counts as 1 and its AP slot is not read. The
 * result is written to a device scalar and the call returns WITHOUT
 * synchronizing; n == 0 writes 0, as DLANTP returns 0. Like calaman.lantr, it
 * allocates an n-element scratch on @p stream.
 *
 * The kernel is calaman.lantr's (lapack/lantr/lantr_bridge.h) with
 * TriStorage::packed, the slot from calaman::tri_index's packed_index.
 *
 * Mapping from DLANTP (docs/architecture.md §4): NORM becomes MatrixNorm, UPLO
 * Uplo, DIAG Diag; s/d/c/z become one template over T, whose norm is always
 * real (ComplexToRealType<T>); WORK is the internal scratch. Divergence: the
 * Frobenius norm is a plain sum of squares, not DLANTP's scaled ?lassq -- the
 * calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lantp;  // also re-exports MatrixNorm, Uplo, Diag and Status
 *   calaman::lantp<double>(stream, calaman::MatrixNorm::inf, calaman::Uplo::L,
 *                          calaman::Diag::U, n, d_AP, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lapack/lantr/lantr_bridge.h"

export module calaman.lantp;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm, Uplo, Diag (:enums), ComplexToRealType

// export import: lantp RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lantr does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lantp;` alone names every selector.
export using calaman::MatrixNorm;
export using calaman::Uplo;
export using calaman::Diag;

/// @brief Norm of the n-by-n triangular A in packed storage @p d_AP (?lantp)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries an allocation or launch failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param uplo     Upper or lower triangle
/// @param diag     Diag::U: the diagonal is all ones and not read
/// @param n        Order of A
/// @param d_AP     Device packed triangle, n(n+1)/2 elements
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lantp(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
             const Diag diag, const std::size_t n, const T *const d_AP,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(R), stream));

  device::lantr<T, R>(stream, which, uplo, diag, device::TriStorage::packed, n, n, /*k=*/0, d_AP,
                      /*ld=*/0, d_result, d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lantr.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lantp<float>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                                    const float *, float *);
extern template Status lantp<double>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                                     const double *, double *);
extern template Status lantp<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                                   std::size_t, const wwr::wwrFloatComplex *,
                                                   float *);
extern template Status lantp<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                                    std::size_t, const wwr::wwrDoubleComplex *,
                                                    double *);

} // namespace calaman
