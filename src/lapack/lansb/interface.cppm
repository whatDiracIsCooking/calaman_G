/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lansb -- the norm of a symmetric band
 *        matrix, LAPACK's ?lansb
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm (equal to the
 * 1-norm, by symmetry) or Frobenius norm of the n-by-n symmetric band matrix A
 * with k super-diagonals, of which one triangle is held in LAPACK band storage:
 * A(i,j) at AB(k+i-j, j) for Uplo::U, at AB(i-j, j) for Uplo::L. Only in-matrix
 * band entries of that triangle are read, so the unused corner of AB and its
 * padding may hold anything. The result is written to a device scalar and the
 * call returns WITHOUT synchronizing; n == 0 writes 0, as DLANSB returns 0.
 * Like calaman.lansp, it allocates an n-element scratch on @p stream.
 *
 * Mapping from DLANSB (docs/architecture.md §4): NORM becomes MatrixNorm, UPLO
 * becomes Uplo; s/d/c/z become one template over T, whose norm is always real
 * (ComplexToRealType<T>); complex is symmetric, not Hermitian (calaman.lanhb,
 * on the same kernel); WORK is the internal scratch. Divergence: the Frobenius
 * norm is a plain sum of squares, not DLANSB's scaled ?lassq -- the
 * calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lansb;  // also re-exports MatrixNorm, Uplo and Status
 *   // d_AB: (k+1)-by-n (or taller) device band storage, column-major
 *   calaman::lansb<double>(stream, calaman::MatrixNorm::one, calaman::Uplo::U,
 *                          n, k, d_AB, ldab, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lansb_bridge.h"

export module calaman.lansb;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm, Uplo (:enums), ComplexToRealType

// export import: lansb RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lansp does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lansb;` alone names both selectors.
export using calaman::MatrixNorm;
export using calaman::Uplo;

/// @brief Norm of the n-by-n symmetric band matrix A in band storage @p d_AB (?lansb)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries an allocation or launch failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param uplo     Which triangle of A the band storage holds
/// @param n        Order of A
/// @param k        Number of super- (Uplo::U) or sub-diagonals (Uplo::L) of A
/// @param d_AB     Device band storage, column-major (layout in the file header)
/// @param ldab     Leading dimension of @p d_AB; ldab >= k+1
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lansb(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
             const std::size_t n, const std::size_t k, const T *const d_AB, const std::size_t ldab,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(R), stream));

  device::lansb<T, R>(stream, which, uplo, /*hermitian=*/false, n, k, d_AB, ldab, d_result,
                      d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lansp.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lansb<float>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t, std::size_t,
                                    const float *, std::size_t, float *);
extern template Status lansb<double>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t, std::size_t,
                                     const double *, std::size_t, double *);
extern template Status lansb<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                                   std::size_t, const wwr::wwrFloatComplex *,
                                                   std::size_t, float *);
extern template Status lansb<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                                    std::size_t, const wwr::wwrDoubleComplex *,
                                                    std::size_t, double *);

} // namespace calaman
