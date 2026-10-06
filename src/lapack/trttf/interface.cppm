/**
 * @file interface.cppm
 * @brief Primary interface for calaman.trttf -- convert a triangular matrix
 *        from full storage (tr) to Rectangular Full Packed (tf), LAPACK's ?trttf
 *
 * One device routine over the four usual_fp element types. Enqueued on the
 * given stream and returns without synchronizing; nothing is allocated, and the
 * opposite triangle of A is never read. TRANSR is Trans::N or, as in LAPACK,
 * Trans::T for a real T and Trans::C for a complex T.
 *
 * Mapping from LAPACK's ?TRTTF (docs/architecture.md §4):
 *
 * | ?TRTTF           | calaman::trttf                                |
 * |------------------|-----------------------------------------------|
 * | CHARACTER TRANSR | calaman::Trans enum (N; T real, C complex)    |
 * | CHARACTER UPLO   | calaman::Uplo enum                            |
 * | s/d/c/z variants | one template over T (usual_fp)                |
 * | INTEGER N, LDA   | std::size_t                                   |
 * | INFO             | returned Status (InvalidValue: transr or lda) |
 *
 * Usage:
 *   import calaman.trttf;     // also re-exports Trans, Uplo and Status
 *   import wwr.runtime_api;
 *   // d_a: n-by-n device matrix (lda); d_arf: n(n+1)/2 RFP elements
 *   calaman::trttf(stream, calaman::Trans::N, calaman::Uplo::U, n, d_a, lda, d_arf);
 */

module;

// CLM_TRY arrives by #include: a GMF cannot import a macro.
#include "error_handling/error_macros.h"

#include "trttf_bridge.h"

export module calaman.trttf;

import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common; // Trans, Uplo, usual_fp, complex_fp

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Trans;
export using calaman::Uplo;

/// @brief Convert the @p uplo triangle of @p d_a to RFP @p d_arf on @p stream
///
/// Reads only the @p uplo triangle of the n-by-n A; enqueues nothing when
/// @p n is 0. Both arrays live on @p stream's device.
///
/// @param transr Trans::N, or the transposed RFP layout: Trans::T (real T) or
///        Trans::C (complex T)
/// @param d_a Device matrix, column-major, leading dimension @p lda
/// @param lda Leading dimension of A; lda >= max(1, n)
/// @param d_arf Device RFP array, n(n+1)/2 elements
/// @return Success, InvalidValue for a bad @p transr or @p lda, or the launch error
export template<calaman::usual_fp T>
Status trttf(const wwr::wwrStream_t stream, const Trans transr, const Uplo uplo,
             const std::size_t n, const T *const d_a, const std::size_t lda, T *const d_arf) {
  constexpr Trans kTransposed = calaman::complex_fp<T> ? Trans::C : Trans::T;
  if ((transr != Trans::N && transr != kTransposed) || lda < std::max<std::size_t>(1, n)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::trttf(stream, transr != Trans::N, uplo, n, d_a, lda, d_arf);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

// Paired with instantiations.cpp: the body names the GMF-declared launcher, so
// it is instantiated once inside this library, never by an importer.
extern template Status trttf<float>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const float *,
                                    std::size_t, float *);
extern template Status trttf<double>(wwr::wwrStream_t, Trans, Uplo, std::size_t, const double *,
                                     std::size_t, double *);
extern template Status trttf<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                                   const wwr::wwrFloatComplex *, std::size_t,
                                                   wwr::wwrFloatComplex *);
extern template Status trttf<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, Uplo, std::size_t,
                                                    const wwr::wwrDoubleComplex *, std::size_t,
                                                    wwr::wwrDoubleComplex *);

} // namespace calaman
