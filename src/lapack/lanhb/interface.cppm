/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lanhb -- the norm of a complex Hermitian
 *        band matrix, LAPACK's ?lanhb
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm (equal to the
 * 1-norm, by symmetry) or Frobenius norm of the n-by-n Hermitian band matrix A
 * with k super-diagonals, of which one triangle is held in LAPACK band storage
 * (calaman.lansb's layout); of each diagonal entry only the real part is read,
 * as ZLANHB assumes it real. The result is written to a real device scalar and
 * the call returns WITHOUT synchronizing; n == 0 writes 0. Like calaman.lansb,
 * it allocates an n-element scratch on @p stream.
 *
 * The kernel is calaman.lansb's (lapack/lansb/lansb_bridge.h) with its
 * `hermitian` flag set -- for these norms the diagonal is the only difference.
 *
 * Mapping from ZLANHB (docs/architecture.md §4): NORM becomes MatrixNorm, UPLO
 * becomes Uplo; c/z become one template over T, the norm ComplexToRealType<T>;
 * WORK is the internal scratch. Divergence: the Frobenius norm is a plain sum
 * of squares, not ZLANHB's scaled ?lassq -- the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lanhb;  // also re-exports MatrixNorm, Uplo and Status
 *   calaman::lanhb<wwr::wwrDoubleComplex>(stream, calaman::MatrixNorm::one,
 *       calaman::Uplo::L, n, k, d_AB, ldab, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lapack/lansb/lansb_bridge.h"

export module calaman.lanhb;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // MatrixNorm, Uplo (:enums), ComplexToRealType

// export import: lanhb RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lansb does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lanhb;` alone names both selectors.
export using calaman::MatrixNorm;
export using calaman::Uplo;

/// @brief Norm of the n-by-n Hermitian band matrix A in band storage @p d_AB (?lanhb)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries an allocation or launch failure.
///
/// @tparam T Element type; wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param uplo     Which triangle of A the band storage holds
/// @param n        Order of A
/// @param k        Number of super- (Uplo::U) or sub-diagonals (Uplo::L) of A
/// @param d_AB     Device band storage, column-major; diagonal imaginary parts ignored
/// @param ldab     Leading dimension of @p d_AB; ldab >= k+1
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lanhb(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
             const std::size_t n, const std::size_t k, const T *const d_AB, const std::size_t ldab,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(R), stream));

  device::lansb<T, R>(stream, which, uplo, /*hermitian=*/true, n, k, d_AB, ldab, d_result,
                      d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lansb.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lanhb<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                                   std::size_t, const wwr::wwrFloatComplex *,
                                                   std::size_t, float *);
extern template Status lanhb<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Uplo, std::size_t,
                                                    std::size_t, const wwr::wwrDoubleComplex *,
                                                    std::size_t, double *);

} // namespace calaman
