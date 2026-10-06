/**
 * @file interface.cppm
 * @brief Primary interface for calaman.tfttr -- convert a triangular matrix
 *        from Rectangular Full Packed (tf) to full storage (tr), LAPACK's ?tfttr
 *
 * One device routine over the four usual_fp element types. Enqueued on the
 * given stream and returns without synchronizing; nothing is allocated, and the
 * opposite triangle of A is untouched. TRANSR is Trans::N or, as in LAPACK,
 * Trans::T for a real T and Trans::C for a complex T.
 *
 * Mapping from LAPACK's ?TFTTR (docs/architecture.md §4):
 *
 * | ?TFTTR           | calaman::tfttr                                |
 * |------------------|-----------------------------------------------|
 * | CHARACTER TRANSR | calaman::Trans enum (N; T real, C complex)    |
 * | CHARACTER UPLO   | calaman::Uplo enum                            |
 * | s/d/c/z variants | one template over T (usual_fp)                |
 * | INTEGER N, LDA   | std::size_t                                   |
 * | INFO             | returned Status (InvalidValue: transr or lda) |
 *
 * Usage:
 *   import calaman.tfttr;     // also re-exports Trans, Uplo and Status
 *   import wwr.runtime_api;
 *   // d_arf: n(n+1)/2 RFP elements; d_a: n-by-n device matrix (lda)
 *   calaman::tfttr(stream, calaman::Trans::N, calaman::Uplo::U, n, d_arf, d_a, lda);
 */

module;

// CLM_TRY arrives by #include: a GMF cannot import a macro.
#include "error_handling/error_macros.h"

#include "tfttr_bridge.h"

export module calaman.tfttr;

import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common; // Trans, Uplo, usual_fp, complex_fp

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Trans;
export using calaman::Uplo;

/// @brief Convert RFP @p d_arf into the @p uplo triangle of @p d_a on @p stream
///
/// Writes only the @p uplo triangle of the n-by-n A; enqueues nothing when
/// @p n is 0. Both arrays live on @p stream's device.
///
/// @param transr Trans::N, or the transposed RFP layout: Trans::T (real T) or
///        Trans::C (complex T)
/// @param d_arf Device RFP array, n(n+1)/2 elements
/// @param d_a Device matrix, column-major, leading dimension @p lda
/// @param lda Leading dimension of A; lda >= max(1, n)
/// @return Success, InvalidValue for a bad @p transr or @p lda, or the launch error
export template<calaman::usual_fp T>
Status tfttr(const wwr::wwrStream_t stream, const Trans transr, const Uplo uplo,
             const std::size_t n, const T *const d_arf, T *const d_a, const std::size_t lda) {
  constexpr Trans kTransposed = calaman::complex_fp<T> ? Trans::C : Trans::T;
  if ((transr != Trans::N && transr != kTransposed) || lda < std::max<std::size_t>(1, n)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::tfttr(stream, transr != Trans::N, uplo, n, d_arf, d_a, lda);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

// Paired with instantiations.cpp: the body names the GMF-declared launcher, so
// it is instantiated once inside this library, never by an importer.
extern template Status tfttr<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *,
                                    float *, std::size_t);
extern template Status tfttr<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                                     double *, std::size_t);
extern template Status tfttr<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                                   const wwr::wwrFloatComplex *,
                                                   wwr::wwrFloatComplex *, std::size_t);
extern template Status tfttr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                                    const wwr::wwrDoubleComplex *,
                                                    wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
