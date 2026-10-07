/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lansy -- the norm of a symmetric matrix
 *        from one triangle, LAPACK's ?lansy
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm (equal to the
 * 1-norm, by symmetry) or Frobenius norm of the n-by-n symmetric A, reading ONLY
 * the triangle @p uplo names -- the other is never addressed, so it may hold
 * anything (calaman.lange would read it). The result is written to a device
 * scalar and the call returns WITHOUT synchronizing; n == 0 writes 0, as DLANSY
 * returns 0. Like calaman.lange, it allocates an n-element scratch on @p stream
 * (wwrMallocAsync/FreeAsync, stream-ordered) for the per-column partials.
 *
 * Mapping from DLANSY (docs/architecture.md §4): NORM becomes MatrixNorm, UPLO
 * becomes Uplo; s/d/c/z become one template over T, whose norm is always real
 * (ComplexToRealType<T>); complex is symmetric, not Hermitian (calaman.lanhe).
 * Divergence: the Frobenius norm is a plain sum of squares (off-diagonal counted
 * twice), not DLANSY's scaled ?lassq -- the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lansy;  // also re-exports MatrixNorm, Uplo and Status
 *   // d_A: n-by-n device matrix, column-major, upper triangle filled
 *   calaman::lansy<double>(stream, calaman::MatrixNorm::max_abs, calaman::Uplo::U,
 *                          n, d_A, lda, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lansy_bridge.h"

export module calaman.lansy;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm, Uplo (:enums), ComplexToRealType

// export import: lansy RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lange does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lansy;` alone names both selectors.
export using calaman::MatrixNorm;
export using calaman::Uplo;

/// @brief Norm of the n-by-n symmetric A from its @p uplo triangle (?lansy)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries an allocation or launch failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param uplo     Which triangle of A holds the matrix; the other is not read
/// @param n        Order of A
/// @param d_A      Device matrix, column-major, leading dimension @p lda
/// @param lda      Leading dimension of @p d_A; lda >= n
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lansy(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
             const std::size_t n, const T *const d_A, const std::size_t lda,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(R), stream));

  device::lansy<T, R>(stream, which, uplo, /*hermitian=*/false, n, d_A, lda, d_result,
                      d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lange.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lansy<float>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                    const float *, std::size_t, float *);
extern template Status lansy<double>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                     const double *, std::size_t, double *);
extern template Status lansy<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo,
                                                   std::size_t, const wwr::wwrFloatComplex *,
                                                   std::size_t, float *);
extern template Status lansy<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo,
                                                    std::size_t, const wwr::wwrDoubleComplex *,
                                                    std::size_t, double *);

} // namespace calaman
