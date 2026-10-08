/**
 * @file interface.cppm
 * @brief Primary interface for calaman.la_gbamv -- absolute-value matrix-vector
 *        product of a general band matrix, LAPACK's ?la_gbamv
 *
 * y := alpha * |op(A)| * |x| + beta * |y| for A m-by-n with kl sub- and ku
 * super-diagonals in LAPACK band storage, A(i, j) = AB(ku + i - j, j) (0-based),
 * then calaman.la_geamv's symbolic-zero nudge. Only the band is read: AB's
 * off-band corners and padding rows may hold anything. Enqueued on the stream
 * and returns WITHOUT synchronizing; allocates nothing (one launch).
 *
 * Mapping from DLA_GBAMV (docs/architecture.md §4) is la_geamv's (its
 * README.md): Trans for the INTEGER TRANS, real alpha/beta/y, signed
 * INCX/INCY, validation before the quick return, bitwise the reference loop.
 * Where the reference loop indexes the transposed band or a strided x wrongly,
 * this computes the documented product instead (README.md).
 *
 * Usage:
 *   import calaman.la_gbamv;  // also re-exports Trans and Status
 *   calaman::la_gbamv<double>(stream, calaman::Trans::N, m, n, kl, ku, 1.0,
 *                             d_ab, ldab, d_x, 1, 0.0, d_y, 1);
 */

module;

// CLM_REQUIRE -- a macro, so it arrives by #include in the global module
// fragment; it needs calaman::Status visible (the export import below).
#include "error_handling/error_macros.h"

#include "la_gbamv_bridge.h"

export module calaman.la_gbamv;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // Trans (:enums), ComplexToRealType

// export import: la_gbamv RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.la_gbamv;` alone names calaman::Trans.
export using calaman::Trans;

/// @brief y := alpha * |op(A)| * |x| + beta * |y| for a band A, plus the symbolic-zero nudge
///
/// Returns without synchronizing; enqueues nothing when @p m or @p n is 0, or
/// when alpha == 0 and beta == 1 (y is then left exactly as given). beta == 0
/// never reads y; alpha == 0 never reads AB or x.
///
/// @tparam T Element type of A and x; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the work is enqueued on; all pointers live on its device
/// @param trans  op(A): A (N), or A^T (T or C -- the same here, |conj(a)| == |a|)
/// @param m      Row count of A
/// @param n      Column count of A
/// @param kl     Sub-diagonal count, kl < max(1, m)
/// @param ku     Super-diagonal count, ku < max(1, n)
/// @param alpha  Real scalar on |op(A)| * |x|
/// @param d_ab   Device AB, (kl + ku + 1)-by-n band of A, column-major, leading dim @p ldab
/// @param ldab   Leading dimension of AB, >= kl + ku + 1
/// @param d_x    Device x: n elements for N, m for T/C, stride @p incx
/// @param incx   Stride of x; non-zero, negative walks from the far end
/// @param beta   Real scalar on |y|
/// @param d_y    Device real y: m elements for N, n for T/C, stride @p incy; updated
/// @param incy   Stride of y; non-zero, negative walks from the far end
/// @return Success; wwrErrorInvalidValue for a bad kl, ku or ldab, a zero
///         stride, or a null pointer on a call that launches; or the launch error
export template<typename T>
Status la_gbamv(const wwr::wwrStream_t stream, const Trans trans, const std::size_t m,
                const std::size_t n, const std::size_t kl, const std::size_t ku,
                const ComplexToRealType<T> alpha, const T *const d_ab, const std::size_t ldab,
                const T *const d_x, const int incx, const ComplexToRealType<T> beta,
                ComplexToRealType<T> *const d_y, const int incy) {
  using R = ComplexToRealType<T>;
  // The reference validates before its quick return, in this order; its
  // KL > M-1 / KU > N-1 would also reject every m == 0 or n == 0 call.
  CLM_REQUIRE(kl < std::max<std::size_t>(1, m) && ku < std::max<std::size_t>(1, n) &&
                  ldab >= kl + ku + 1 && incx != 0 && incy != 0,
              wwr::wwrErrorInvalidValue);
  if (m == 0 || n == 0 || (alpha == R{0} && beta == R{1})) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d_ab != nullptr && d_x != nullptr && d_y != nullptr, wwr::wwrErrorInvalidValue);
  const std::size_t lenx = trans == Trans::N ? n : m;
  const std::size_t leny = trans == Trans::N ? m : n;
  // KX / KY: a negative stride starts at the far end of the vector.
  const auto first = [](const std::size_t len, const int inc) -> std::ptrdiff_t {
    return inc > 0 ? 0 : static_cast<std::ptrdiff_t>(len - 1) * -static_cast<std::ptrdiff_t>(inc);
  };
  // SAFE1 = (N+1) * ?LAMCH('S'): N is A's column count whatever trans is.
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  device::la_gbamv<T, R>(stream, trans, m, n, kl, ku, alpha, d_ab, ldab, d_x + first(lenx, incx),
                         incx, beta, d_y + first(leny, incy), incy, safe1);
  return wwr::wwrGetLastError();
}

extern template Status la_gbamv<float>(wwr::wwrStream_t, Trans, std::size_t, std::size_t,
                                       std::size_t, std::size_t, float, const float *,
                                       std::size_t, const float *, int, float, float *, int);
extern template Status la_gbamv<double>(wwr::wwrStream_t, Trans, std::size_t, std::size_t,
                                        std::size_t, std::size_t, double, const double *,
                                        std::size_t, const double *, int, double, double *, int);
extern template Status la_gbamv<wwr::wwrFloatComplex>(
    wwr::wwrStream_t, Trans, std::size_t, std::size_t, std::size_t, std::size_t, float,
    const wwr::wwrFloatComplex *, std::size_t, const wwr::wwrFloatComplex *, int, float, float *,
    int);
extern template Status la_gbamv<wwr::wwrDoubleComplex>(
    wwr::wwrStream_t, Trans, std::size_t, std::size_t, std::size_t, std::size_t, double,
    const wwr::wwrDoubleComplex *, std::size_t, const wwr::wwrDoubleComplex *, int, double,
    double *, int);

} // namespace calaman
