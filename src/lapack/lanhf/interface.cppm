/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lanhf -- the norm of a complex Hermitian
 *        matrix in Rectangular Full Packed storage, LAPACK's ?lanhf
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm (equal to the
 * 1-norm, by symmetry) or Frobenius norm of the n-by-n Hermitian A held in the
 * n(n+1)/2-element RFP array ARF; of each diagonal entry only the real part is
 * read, as ZLANHF assumes it real. The result is written to a real device
 * scalar and the call returns WITHOUT synchronizing; n == 0 writes 0. Like
 * calaman.lansf, it allocates an n-element scratch on @p stream.
 *
 * The kernel is calaman.lansf's (lapack/lansf/lansf_bridge.h) with its
 * `hermitian` flag set -- for these norms the diagonal is the only difference.
 *
 * Mapping from ZLANHF (docs/architecture.md §4): NORM becomes MatrixNorm,
 * TRANSR becomes Trans (N or C), UPLO becomes Uplo; c/z become one template
 * over T, the norm ComplexToRealType<T>; WORK is the internal scratch.
 * Divergence: the Frobenius norm is a plain sum of squares, not ZLANHF's
 * scaled ?lassq -- the calaman.lange precedent.
 *
 * Usage:
 *   import calaman.lanhf;  // also re-exports MatrixNorm, Trans, Uplo and Status
 *   calaman::lanhf<wwr::wwrDoubleComplex>(stream, calaman::MatrixNorm::one,
 *       calaman::Trans::C, calaman::Uplo::L, n, d_arf, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lapack/lansf/lansf_bridge.h"

export module calaman.lanhf;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // MatrixNorm, Trans, Uplo (:enums), ComplexToRealType

// export import: lanhf RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lansf does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lanhf;` alone names every selector.
export using calaman::MatrixNorm;
export using calaman::Trans;
export using calaman::Uplo;

/// @brief Norm of the n-by-n Hermitian A in RFP storage @p d_arf (?lanhf)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries a bad @p transr, or an
/// allocation or launch failure.
///
/// @tparam T Element type; wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param transr   Trans::N, or Trans::C for the conjugate-transposed RFP layout
/// @param uplo     Which triangle of A the RFP array holds
/// @param n        Order of A
/// @param d_arf    Device RFP array, n(n+1)/2 elements; diagonal imaginary parts ignored
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, InvalidValue for a bad @p transr, or the first failing step's error
export template<typename T>
Status lanhf(const wwr::wwrStream_t stream, const MatrixNorm which, const Trans transr,
             const Uplo uplo, const std::size_t n, const T *const d_arf,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (transr != Trans::N && transr != Trans::C) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }
  R *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(R), stream));

  device::lansf<T, R>(stream, which, transr != Trans::N, uplo, /*hermitian=*/true, n, d_arf,
                      d_result, d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lansf.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lanhf<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo,
                                                   std::size_t, const wwr::wwrFloatComplex *,
                                                   float *);
extern template Status lanhf<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo,
                                                    std::size_t, const wwr::wwrDoubleComplex *,
                                                    double *);

} // namespace calaman
