/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lantr -- the norm of a triangular
 *        (trapezoidal) matrix, LAPACK's ?lantr
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm or Frobenius norm
 * of the m-by-n upper (i <= j) or lower (i >= j) trapezoid A, reading ONLY that
 * trapezoid -- the opposite triangle and the lda padding may hold anything.
 * With Diag::U each diagonal entry counts as 1 and is not read. The result is
 * written to a device scalar and the call returns WITHOUT synchronizing;
 * min(m, n) == 0 writes 0, as DLANTR returns 0. Like calaman.lanhs, it
 * allocates an m- (inf) or n-element scratch on @p stream.
 *
 * Mapping from DLANTR (docs/architecture.md §4): NORM becomes MatrixNorm, UPLO
 * Uplo, DIAG Diag; s/d/c/z become one template over T, whose norm is always
 * real (ComplexToRealType<T>); WORK is the internal scratch. The kernel is
 * shared with calaman.lantp and calaman.lantb (lantr_bridge.h's TriStorage).
 * Divergence: the Frobenius norm is a plain sum of squares, not DLANTR's
 * scaled ?lassq -- the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lantr;  // also re-exports MatrixNorm, Uplo, Diag and Status
 *   calaman::lantr<double>(stream, calaman::MatrixNorm::one, calaman::Uplo::U,
 *                          calaman::Diag::N, m, n, d_A, lda, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lantr_bridge.h"

export module calaman.lantr;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm, Uplo, Diag (:enums), ComplexToRealType

// export import: lantr RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lanhs does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lantr;` alone names every selector.
export using calaman::MatrixNorm;
export using calaman::Uplo;
export using calaman::Diag;

/// @brief Norm of the m-by-n upper or lower trapezoid A (?lantr)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p m or @p n is 0. The return carries an allocation or launch failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param uplo     Upper (i <= j) or lower (i >= j) trapezoid
/// @param diag     Diag::U: the diagonal is all ones and not read
/// @param m        Rows of A
/// @param n        Columns of A
/// @param d_A      Device matrix, column-major; only the trapezoid is read
/// @param lda      Leading dimension of @p d_A; lda >= m
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lantr(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
             const Diag diag, const std::size_t m, const std::size_t n, const T *const d_A,
             const std::size_t lda, ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (m == 0 || n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  const std::size_t lines = which == MatrixNorm::inf ? m : n;
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), lines * sizeof(R), stream));

  device::lantr<T, R>(stream, which, uplo, diag, device::TriStorage::full, m, n, /*k=*/0, d_A,
                      lda, d_result, d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lanhs.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lantr<float>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                                    std::size_t, const float *, std::size_t, float *);
extern template Status lantr<double>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, std::size_t,
                                     std::size_t, const double *, std::size_t, double *);
extern template Status lantr<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                                   std::size_t, std::size_t,
                                                   const wwr::wwrFloatComplex *, std::size_t,
                                                   float *);
extern template Status lantr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                                    std::size_t, std::size_t,
                                                    const wwr::wwrDoubleComplex *, std::size_t,
                                                    double *);

} // namespace calaman
