/**
 * @file interface.cppm
 * @brief Primary interface for calaman.la_geamv -- absolute-value matrix-vector
 *        product of a general matrix, LAPACK's ?la_geamv
 *
 * y := alpha * |op(A)| * |x| + beta * |y|, then every y(i) that is not a
 * symbolic zero is nudged away from zero by (n + 1) * the underflow threshold
 * -- the bound ?gerfsx-style refinement wants. |.| is CABS1 (|re| + |im|) for a
 * complex element, never the modulus. Enqueued on the stream and returns
 * WITHOUT synchronizing; allocates nothing (one launch, no workspace).
 *
 * Mapping from DLA_GEAMV (docs/architecture.md §4): the INTEGER TRANS
 * (ILATRANS's 111/112/113) becomes Trans; s/d/c/z become one template over T
 * with alpha, beta and y real (ComplexToRealType<T>) as in the reference; LDA
 * and the signed INCX/INCY are kept, a negative stride walking from the far
 * end. A bad LDA or a zero stride is wwrErrorInvalidValue (the reference's
 * XERBLA). The result is bitwise the reference loop's (la_geamv.cu); README.md
 * fixes the conventions ?la_gbamv and ?la_heamv/?la_syamv copy.
 *
 * Usage:
 *   import calaman.la_geamv;  // also re-exports Trans and Status
 *   calaman::la_geamv<double>(stream, calaman::Trans::N, m, n, 1.0, d_a, lda,
 *                             d_x, 1, 0.0, d_y, 1);
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the global module
// fragment; it needs calaman::Status visible (the export import below).
#include "error_handling/error_macros.h"

#include "la_geamv_bridge.h"

export module calaman.la_geamv;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // Trans (:enums), ComplexToRealType

// export import: la_geamv RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.la_geamv;` alone names calaman::Trans.
export using calaman::Trans;

/// @brief y := alpha * |op(A)| * |x| + beta * |y|, plus the symbolic-zero nudge
///
/// Returns without synchronizing; enqueues nothing when @p m or @p n is 0, or
/// when alpha == 0 and beta == 1 (y is then left exactly as given). beta == 0
/// never reads y; alpha == 0 never reads A or x.
///
/// @tparam T Element type of A and x; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the work is enqueued on; all pointers live on its device
/// @param trans  op(A): A (N), or A^T (T or C -- the same here, |conj(a)| == |a|)
/// @param m      Row count of A
/// @param n      Column count of A
/// @param alpha  Real scalar on |op(A)| * |x|
/// @param d_a    Device A, column-major, leading dimension @p lda >= max(1, m)
/// @param lda    Leading dimension of A
/// @param d_x    Device x: n elements for N, m for T/C, stride @p incx
/// @param incx   Stride of x; non-zero, negative walks from the far end
/// @param beta   Real scalar on |y|
/// @param d_y    Device real y: m elements for N, n for T/C, stride @p incy; updated
/// @param incy   Stride of y; non-zero, negative walks from the far end
/// @return Success; wwrErrorInvalidValue for lda < max(1, m), a zero stride, or
///         a null pointer on a call that launches; or the launch error
export template<typename T>
Status la_geamv(const wwr::wwrStream_t stream, const Trans trans, const std::size_t m,
                const std::size_t n, const ComplexToRealType<T> alpha, const T *const d_a,
                const std::size_t lda, const T *const d_x, const int incx,
                const ComplexToRealType<T> beta, ComplexToRealType<T> *const d_y, const int incy) {
  using R = ComplexToRealType<T>;
  // The reference validates before its quick return; so does this.
  CLM_REQUIRE(lda >= std::max<std::size_t>(1, m) && incx != 0 && incy != 0,
              wwr::wwrErrorInvalidValue);
  if (m == 0 || n == 0 || (alpha == R{0} && beta == R{1})) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d_a != nullptr && d_x != nullptr && d_y != nullptr, wwr::wwrErrorInvalidValue);
  const std::size_t lenx = trans == Trans::N ? n : m;
  const std::size_t leny = trans == Trans::N ? m : n;
  // KX / KY: a negative stride starts at the far end of the vector.
  const auto first = [](const std::size_t len, const int inc) -> std::ptrdiff_t {
    return inc > 0 ? 0 : static_cast<std::ptrdiff_t>(len - 1) * -static_cast<std::ptrdiff_t>(inc);
  };
  // SAFE1 = (N+1) * ?LAMCH('S'): N is A's column count whatever trans is.
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  device::la_geamv<T, R>(stream, trans, lenx, leny, alpha, d_a, lda, d_x + first(lenx, incx), incx,
                         beta, d_y + first(leny, incy), incy, safe1);
  return wwr::wwrGetLastError();
}

extern template Status la_geamv<float>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, float,
                                       const float *, std::size_t, const float *, int, float,
                                       float *, int);
extern template Status la_geamv<double>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, double,
                                        const double *, std::size_t, const double *, int, double,
                                        double *, int);
extern template Status la_geamv<wwr::wwrFloatComplex>(wwr::wwrStream_t, Trans, std::size_t,
                                                      std::size_t, float,
                                                      const wwr::wwrFloatComplex *, std::size_t,
                                                      const wwr::wwrFloatComplex *, int, float,
                                                      float *, int);
extern template Status la_geamv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, Trans, std::size_t,
                                                       std::size_t, double,
                                                       const wwr::wwrDoubleComplex *, std::size_t,
                                                       const wwr::wwrDoubleComplex *, int, double,
                                                       double *, int);

} // namespace calaman
