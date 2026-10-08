/**
 * @file interface.cppm
 * @brief Primary interface for calaman.la_heamv -- absolute-value matrix-vector
 *        product of a Hermitian matrix held in one triangle, LAPACK's ?la_heamv
 *
 * y := alpha * |A| * |x| + beta * |y| for the n-by-n Hermitian A whose @p uplo
 * triangle is stored, then calaman.la_geamv's symbolic-zero nudge. Only that
 * triangle is read. |.| is CABS1, and CABS1(conj(a)) == CABS1(a), so the
 * kernel is calaman.la_syamv's, with no Hermitian flag: the two routines agree
 * bit for bit on the same stored triangle. As in CLA_HEAMV, a diagonal entry's
 * imaginary part IS read (CABS1), unlike ?hemv, which assumes it zero.
 * Enqueued on the stream and returns WITHOUT synchronizing; allocates nothing.
 *
 * Mapping from ZLA_HEAMV is la_syamv's (lapack/la_syamv/README.md); c/z only,
 * as the reference has no s/d form.
 *
 * Usage:
 *   import calaman.la_heamv;  // also re-exports Uplo and Status
 *   calaman::la_heamv<wwr::wwrDoubleComplex>(stream, calaman::Uplo::U, n, 1.0,
 *                                            d_a, lda, d_x, 1, 0.0, d_y, 1);
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the global module
// fragment; it needs calaman::Status visible (the export import below).
#include "error_handling/error_macros.h"

#include "lapack/la_syamv/la_syamv_bridge.h"

export module calaman.la_heamv;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // Uplo (:enums), ComplexToRealType

// export import: la_heamv RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.la_heamv;` alone names calaman::Uplo.
export using calaman::Uplo;

/// @brief y := alpha * |A| * |x| + beta * |y| for Hermitian A in one triangle, plus the nudge
///
/// Returns without synchronizing; enqueues nothing when @p n is 0, or when
/// alpha == 0 and beta == 1 (y is then left exactly as given). beta == 0 never
/// reads y; alpha == 0 never reads A or x.
///
/// @tparam T Element type of A and x; wwrFloatComplex or wwrDoubleComplex
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
  requires(!std::is_same_v<T, ComplexToRealType<T>>)
Status la_heamv(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n,
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

extern template Status la_heamv<wwr::wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t, float,
                                                      const wwr::wwrFloatComplex *, std::size_t,
                                                      const wwr::wwrFloatComplex *, int, float,
                                                      float *, int);
extern template Status la_heamv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t, double,
                                                       const wwr::wwrDoubleComplex *, std::size_t,
                                                       const wwr::wwrDoubleComplex *, int, double,
                                                       double *, int);

} // namespace calaman
