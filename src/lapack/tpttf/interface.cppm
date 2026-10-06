/**
 * @file interface.cppm
 * @brief Primary interface for calaman.tpttf -- convert a triangular matrix
 *        from packed (tp) to Rectangular Full Packed storage (tf), LAPACK's
 *        ?tpttf
 *
 * One device routine over the four usual_fp element types. Enqueued on the
 * given stream and returns without synchronizing; nothing is allocated. TRANSR
 * is Trans::N or, as in LAPACK, Trans::T for a real T and Trans::C for a
 * complex T.
 *
 * Mapping from LAPACK's ?TPTTF (docs/architecture.md §4):
 *
 * | ?TPTTF           | calaman::tpttf                             |
 * |------------------|--------------------------------------------|
 * | CHARACTER TRANSR | calaman::Trans enum (N; T real, C complex) |
 * | CHARACTER UPLO   | calaman::Uplo enum                         |
 * | s/d/c/z variants | one template over T (usual_fp)             |
 * | INTEGER N        | std::size_t                                |
 * | INFO             | returned Status (InvalidValue: transr)     |
 *
 * Usage:
 *   import calaman.tpttf;     // also re-exports Trans, Uplo and Status
 *   import wwr.runtime_api;
 *   // d_ap: n(n+1)/2 packed elements; d_arf: n(n+1)/2 RFP elements
 *   calaman::tpttf(stream, calaman::Trans::N, calaman::Uplo::U, n, d_ap, d_arf);
 */

module;

// CLM_TRY arrives by #include: a GMF cannot import a macro.
#include "error_handling/error_macros.h"

#include "tpttf_bridge.h"

export module calaman.tpttf;

import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common; // Trans, Uplo, usual_fp, complex_fp

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Trans;
export using calaman::Uplo;

/// @brief Convert the packed @p uplo triangle @p d_ap to RFP @p d_arf on @p stream
///
/// Enqueues nothing when @p n is 0. Both arrays live on @p stream's device.
///
/// @param transr Trans::N, or the transposed RFP layout: Trans::T (real T) or
///        Trans::C (complex T)
/// @param d_ap Device packed triangle, n(n+1)/2 elements, column-major
/// @param d_arf Device RFP array, n(n+1)/2 elements
/// @return Success, InvalidValue for a bad @p transr, or the launch error
export template<calaman::usual_fp T>
Status tpttf(const wwr::wwrStream_t stream, const Trans transr, const Uplo uplo,
             const std::size_t n, const T *const d_ap, T *const d_arf) {
  constexpr Trans kTransposed = calaman::complex_fp<T> ? Trans::C : Trans::T;
  if (transr != Trans::N && transr != kTransposed) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::tpttf(stream, transr != Trans::N, uplo, n, d_ap, d_arf);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

// Paired with instantiations.cpp: the body names the GMF-declared launcher, so
// it is instantiated once inside this library, never by an importer.
extern template Status tpttf<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *,
                                    float *);
extern template Status tpttf<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                                     double *);
extern template Status tpttf<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                                   const wwr::wwrFloatComplex *,
                                                   wwr::wwrFloatComplex *);
extern template Status tpttf<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                                    const wwr::wwrDoubleComplex *,
                                                    wwr::wwrDoubleComplex *);

} // namespace calaman
