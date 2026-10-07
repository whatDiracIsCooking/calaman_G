/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lansf -- the norm of a real symmetric
 *        matrix in Rectangular Full Packed storage, LAPACK's ?lansf
 *
 * Returns the largest absolute entry, 1-norm, infinity-norm (equal to the
 * 1-norm, by symmetry) or Frobenius norm of the n-by-n symmetric A held in the
 * n(n+1)/2-element RFP array ARF (the layout of src/lapack/trttf/README.md).
 * The result is written to a device scalar and the call returns WITHOUT
 * synchronizing; n == 0 writes 0, as DLANSF returns 0. Like calaman.lansy, it
 * allocates an n-element scratch on @p stream (stream-ordered).
 *
 * Mapping from DLANSF (docs/architecture.md §4): NORM becomes MatrixNorm,
 * TRANSR becomes Trans (N or T), UPLO becomes Uplo; s/d become one template
 * over T; WORK is the internal scratch. Divergence: the Frobenius norm is a
 * plain sum of squares, not DLANSF's scaled ?lassq -- the calaman.lange
 * precedent. The Hermitian counterpart is calaman.lanhf, on the same kernel.
 *
 * Usage:
 *   import calaman.lansf;  // also re-exports MatrixNorm, Trans, Uplo and Status
 *   // d_arf: n(n+1)/2 device elements, RFP, TRANSR = 'N', UPLO = 'U'
 *   calaman::lansf<double>(stream, calaman::MatrixNorm::one, calaman::Trans::N,
 *                          calaman::Uplo::U, n, d_arf, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

#include "lansf_bridge.h"

export module calaman.lansf;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import calaman.common;  // MatrixNorm, Trans, Uplo (:enums)

// export import: lansf RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export lansy does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lansf;` alone names every selector.
export using calaman::MatrixNorm;
export using calaman::Trans;
export using calaman::Uplo;

/// @brief Norm of the n-by-n symmetric A in RFP storage @p d_arf (?lansf)
///
/// Writes the @p which norm to @p d_result and returns without synchronizing;
/// writes 0 when @p n is 0. The return carries a bad @p transr, or an
/// allocation or launch failure.
///
/// @tparam T Element type; float or double
/// @param stream   Stream the work is enqueued on; all pointers live on its device
/// @param which    Which norm to compute (max_abs, one, inf, frobenius)
/// @param transr   Trans::N, or Trans::T for the transposed RFP layout
/// @param uplo     Which triangle of A the RFP array holds
/// @param n        Order of A
/// @param d_arf    Device RFP array, n(n+1)/2 elements
/// @param d_result Device scalar receiving the norm
/// @return Success, InvalidValue for a bad @p transr, or the first failing step's error
export template<typename T>
Status lansf(const wwr::wwrStream_t stream, const MatrixNorm which, const Trans transr,
             const Uplo uplo, const std::size_t n, const T *const d_arf, T *const d_result) {
  if (transr != Trans::N && transr != Trans::T) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(T), stream);
  }
  T *d_scratch = nullptr;
  CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), n * sizeof(T), stream));

  device::lansf<T, T>(stream, which, transr != Trans::N, uplo, /*hermitian=*/false, n, d_arf,
                      d_result, d_scratch);
  // Capture the sticky launch error BEFORE the free so a free failure cannot
  // mask it; the stream-ordered free's own failure is discarded, as in lansy.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lansf<float>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo, std::size_t,
                                    const float *, float *);
extern template Status lansf<double>(wwr::wwrStream_t, MatrixNorm, Trans, Uplo, std::size_t,
                                     const double *, double *);

} // namespace calaman
