/**
 * @file interface.cppm
 * @brief Primary interface for calaman.trttp -- pack a triangular matrix from
 *        full storage (tr) into packed storage (tp), LAPACK's ?trttp
 *
 * One device routine over the four usual_fp element types. Enqueued on the
 * given stream and returns without synchronizing; nothing is allocated, and the
 * opposite triangle of A is never read.
 *
 * Mapping from LAPACK's ?TRTTP (docs/architecture.md §4):
 *
 * | ?TRTTP           | calaman::trttp                           |
 * |------------------|------------------------------------------|
 * | CHARACTER UPLO   | calaman::Uplo enum                       |
 * | s/d/c/z variants | one template over T (usual_fp)           |
 * | INTEGER N, LDA   | std::size_t                              |
 * | INFO             | returned Status (InvalidValue: bad lda)  |
 *
 * Usage:
 *   import calaman.trttp;     // also re-exports calaman::Uplo and Status
 *   import wwr.runtime_api;
 *   // d_a: n-by-n device matrix (lda); d_ap: n(n+1)/2 packed elements
 *   calaman::trttp(stream, calaman::Uplo::U, n, d_a, lda, d_ap);
 */

module;

// CLM_TRY arrives by #include: a GMF cannot import a macro.
#include "error_handling/error_macros.h"

#include "trttp_bridge.h"

export module calaman.trttp;

import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common; // Uplo, usual_fp

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Uplo;

/// @brief Pack the @p uplo triangle of @p d_a into @p d_ap on @p stream
///
/// Reads only the @p uplo triangle of the n-by-n A; enqueues nothing when
/// @p n is 0. Both arrays live on @p stream's device.
///
/// @param d_a Device matrix, column-major, leading dimension @p lda
/// @param lda Leading dimension of A; lda >= max(1, n)
/// @param d_ap Device packed triangle, n(n+1)/2 elements, column-major
/// @return Success, InvalidValue for a bad @p lda, or the launch error
export template<calaman::usual_fp T>
Status trttp(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n,
             const T *const d_a, const std::size_t lda, T *const d_ap) {
  if (lda < std::max<std::size_t>(1, n)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::trttp(stream, uplo, n, d_a, lda, d_ap);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

// Paired with instantiations.cpp: the body names the GMF-declared launcher, so
// it is instantiated once inside this library, never by an importer.
extern template Status trttp<float>(wwr::wwrStream_t, Uplo, std::size_t, const float *,
                                    std::size_t, float *);
extern template Status trttp<double>(wwr::wwrStream_t, Uplo, std::size_t, const double *,
                                     std::size_t, double *);
extern template Status trttp<wwr::wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                                   const wwr::wwrFloatComplex *, std::size_t,
                                                   wwr::wwrFloatComplex *);
extern template Status trttp<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                                    const wwr::wwrDoubleComplex *, std::size_t,
                                                    wwr::wwrDoubleComplex *);

} // namespace calaman
