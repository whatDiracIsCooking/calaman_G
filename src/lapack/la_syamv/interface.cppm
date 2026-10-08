/**
 * @file interface.cppm
 * @brief Primary interface for calaman.la_syamv -- absolute-value matrix-vector
 *        product of a symmetric matrix held in one triangle, LAPACK's ?la_syamv
 *
 * y := alpha * |A| * |x| + beta * |y| for the n-by-n symmetric A whose @p uplo
 * triangle is stored, then calaman.la_geamv's symbolic-zero nudge. Only that
 * triangle is read. |.| is CABS1 for a complex element, so the complex
 * symmetric and Hermitian readings agree: calaman.la_heamv runs this kernel.
 * Enqueued on the stream and returns WITHOUT synchronizing; allocates nothing.
 *
 * Mapping from DLA_SYAMV (docs/architecture.md §4) is la_geamv's (its
 * README.md): the INTEGER UPLO (ILAUPLO's 121/122) becomes Uplo, real
 * alpha/beta/y, signed INCX/INCY, validation before the quick return, bitwise
 * the reference loop. Where the reference's strided-x branch tests the wrong
 * x(j) for a symbolic zero, this tests the x it multiplies (README.md).
 *
 * Usage:
 *   import calaman.la_syamv;  // also re-exports Uplo and Status
 *   calaman::la_syamv<double>(stream, calaman::Uplo::L, n, 1.0, d_a, lda,
 *                             d_x, 1, 0.0, d_y, 1);
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the global module
// fragment; it needs calaman::Status visible (the export import below).
#include "error_handling/error_macros.h"

#include "la_syamv_bridge.h"

export module calaman.la_syamv;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // Uplo (:enums), ComplexToRealType

// export import: la_syamv RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.la_syamv;` alone names calaman::Uplo.
export using calaman::Uplo;

/// @brief y := alpha * |A| * |x| + beta * |y| for symmetric A in one triangle, plus the nudge
///
/// Returns without synchronizing; enqueues nothing when @p n is 0, or when
/// alpha == 0 and beta == 1 (y is then left exactly as given). beta == 0 never
/// reads y; alpha == 0 never reads A or x.
///
/// @tparam T Element type of A and x; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the work is enqueued on; all pointers live on its device
/// @param uplo   Which triangle of A is stored; the other is never read
/// @param n      Order of A
/// @param alpha  Real scalar on |A| * |x|
/// @param d_a    Device A, column-major, leading dimension @p lda >= max(1, n)
/// @param lda    Leading dimension of A
/// @param d_x    Device x: n elements, stride @p incx
/// @param incx   Stride of x; non-zero, negative walks from the far end
/// @param beta   Real scalar on |y|
/// @param d_y    Device real y: n elements, stride @p incy; updated
/// @param incy   Stride of y; non-zero, negative walks from the far end
/// @return Success; wwrErrorInvalidValue for lda < max(1, n), a zero stride, or
///         a null pointer on a call that launches; or the launch error
export template<typename T>
Status la_syamv(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n,
                const ComplexToRealType<T> alpha, const T *const d_a, const std::size_t lda,
                const T *const d_x, const int incx, const ComplexToRealType<T> beta,
                ComplexToRealType<T> *const d_y, const int incy) {
  using R = ComplexToRealType<T>;
  // The reference validates before its quick return, in this order.
  CLM_REQUIRE(lda >= std::max<std::size_t>(1, n) && incx != 0 && incy != 0,
              wwr::wwrErrorInvalidValue);
  if (n == 0 || (alpha == R{0} && beta == R{1})) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d_a != nullptr && d_x != nullptr && d_y != nullptr, wwr::wwrErrorInvalidValue);
  // KX / KY: a negative stride starts at the far end of the vector.
  const auto first = [n](const int inc) -> std::ptrdiff_t {
    return inc > 0 ? 0 : static_cast<std::ptrdiff_t>(n - 1) * -static_cast<std::ptrdiff_t>(inc);
  };
  // SAFE1 = (N+1) * ?LAMCH('S').
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  device::la_syamv<T, R>(stream, uplo, n, alpha, d_a, lda, d_x + first(incx), incx, beta,
                         d_y + first(incy), incy, safe1);
  return wwr::wwrGetLastError();
}

extern template Status la_syamv<float>(wwr::wwrStream_t, Uplo, std::size_t, float, const float *,
                                       std::size_t, const float *, int, float, float *, int);
extern template Status la_syamv<double>(wwr::wwrStream_t, Uplo, std::size_t, double, const double *,
                                        std::size_t, const double *, int, double, double *, int);
extern template Status la_syamv<wwr::wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t, float,
                                                      const wwr::wwrFloatComplex *, std::size_t,
                                                      const wwr::wwrFloatComplex *, int, float,
                                                      float *, int);
extern template Status la_syamv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t, double,
                                                       const wwr::wwrDoubleComplex *, std::size_t,
                                                       const wwr::wwrDoubleComplex *, int, double,
                                                       double *, int);

} // namespace calaman
