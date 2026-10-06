/**
 * @file interface.cppm
 * @brief Primary interface for calaman.tpttr -- unpack a triangular matrix from
 *        packed storage (tp) into full storage (tr), LAPACK's ?tpttr
 *
 * One device routine over the four usual_fp element types. Enqueued on the
 * given stream and returns without synchronizing; nothing is allocated, and the
 * opposite triangle of A is untouched.
 *
 * Mapping from LAPACK's ?TPTTR (docs/architecture.md §4):
 *
 * | ?TPTTR           | calaman::tpttr                           |
 * |------------------|------------------------------------------|
 * | CHARACTER UPLO   | calaman::Uplo enum                       |
 * | s/d/c/z variants | one template over T (usual_fp)           |
 * | INTEGER N, LDA   | std::size_t                              |
 * | INFO             | returned Status (InvalidValue: bad lda)  |
 *
 * Usage:
 *   import calaman.tpttr;     // also re-exports calaman::Uplo and Status
 *   import wwr.runtime_api;
 *   // d_ap: n(n+1)/2 packed elements; d_a: n-by-n device matrix (lda)
 *   calaman::tpttr(stream, calaman::Uplo::U, n, d_ap, d_a, lda);
 */

module;

// CLM_TRY arrives by #include: a GMF cannot import a macro.
#include "error_handling/error_macros.h"

#include "tpttr_bridge.h"

export module calaman.tpttr;

import std;
import wwr.runtime_api;
import wwr.complex;    // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common; // Uplo, usual_fp

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

export using calaman::Uplo;

/// @brief Unpack the @p uplo triangle of packed @p d_ap into @p d_a on @p stream
///
/// Writes only the @p uplo triangle of the n-by-n A; enqueues nothing when
/// @p n is 0. Both arrays live on @p stream's device.
///
/// @param d_ap Device packed triangle, n(n+1)/2 elements, column-major
/// @param d_a Device matrix, column-major, leading dimension @p lda
/// @param lda Leading dimension of A; lda >= max(1, n)
/// @return Success, InvalidValue for a bad @p lda, or the launch error
export template<calaman::usual_fp T>
Status tpttr(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n,
             const T *const d_ap, T *const d_a, const std::size_t lda) {
  if (lda < std::max<std::size_t>(1, n)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::tpttr(stream, uplo, n, d_ap, d_a, lda);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

// Paired with instantiations.cpp: the body names the GMF-declared launcher, so
// it is instantiated once inside this library, never by an importer.
extern template Status tpttr<float>(wwr::wwrStream_t, Uplo, std::size_t, const float *, float *,
                                    std::size_t);
extern template Status tpttr<double>(wwr::wwrStream_t, Uplo, std::size_t, const double *,
                                     double *, std::size_t);
extern template Status tpttr<wwr::wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                                   const wwr::wwrFloatComplex *,
                                                   wwr::wwrFloatComplex *, std::size_t);
extern template Status tpttr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                                    const wwr::wwrDoubleComplex *,
                                                    wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
