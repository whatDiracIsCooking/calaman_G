/**
 * @file la_amv.cuh
 * @brief The per-y(i) body every ?la_*amv kernel shares: beta * |y(i)| plus
 *        alpha * sum |op(A)(i,j)| * |x(j)|, with the symbolic-zero nudge
 *
 * Header-only `__device__` helpers for calaman.la_geamv and calaman.la_gbamv;
 * a kernel supplies only its storage scheme, as the j range and an accessor
 * for op(A)(i, j). |.| is CABS1 (|re| + |im|) for a complex element.
 *
 * Every multiply and add is rounded on its own, never contracted into an FMA,
 * so the result is bitwise the reference loop's (la_geamv/README.md, decision
 * 2): CUDA spells the _rn intrinsics; HIP spells plain operators under
 * `fp contract(off)`, because HIP's own _rn functions contract under its
 * default fp-contract=fast.
 *
 * Reached root-relative as "lapack/la_geamv/la_amv.cuh"; link the INTERFACE
 * target calaman::la_amv. Device-only: include it from a .cu.
 *
 * Usage:
 *   #include "lapack/la_geamv/la_amv.cuh"
 *
 *   la_amv_row<T, R>(y + i * incy, alpha, beta, safe1, x, incx, j0, j1,
 *                    [=](std::size_t j) { return a[i + j * lda]; });
 */

#pragma once

#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

/// @brief a * b, rounded once and never fused with a neighbouring add
template<typename R>
__device__ __forceinline__ R mul_rn(const R a, const R b) {
#if defined(__HIP__)
#pragma clang fp contract(off)
  return a * b;
#else
  if constexpr (std::is_same_v<R, float>) {
    return __fmul_rn(a, b);
  } else {
    return __dmul_rn(a, b);
  }
#endif
}

/// @brief a + b, rounded once and never fused with a neighbouring multiply
template<typename R>
__device__ __forceinline__ R add_rn(const R a, const R b) {
#if defined(__HIP__)
#pragma clang fp contract(off)
  return a + b;
#else
  if constexpr (std::is_same_v<R, float>) {
    return __fadd_rn(a, b);
  } else {
    return __dadd_rn(a, b);
  }
#endif
}

/// @brief BLAS CABS1: |a| for a real a, |Re a| + |Im a| for a complex one
template<typename T, typename R>
__device__ __forceinline__ R abs1(const T a) {
  using ops = elem_ops<T>;
  if constexpr (std::is_same_v<T, R>) {
    return wwr::fabs(a);
  } else {
    return add_rn(wwr::fabs(ops::real_part(a)), wwr::fabs(ops::imag_part(a)));
  }
}

/// @brief One y(i) of ?la_*amv: the reference's loop body, SYMB_ZERO and all
///
/// Sums j = j0 .. j1-1 in ascending order. beta == 0 sets y(i) to +0 without
/// reading it; alpha == 0 never calls @p a_at or reads x.
///
/// @tparam AAt Callable j -> op(A)(i, j) as a T; called only for j in [j0, j1)
/// @param yi    This thread's y(i); read unless beta == 0, always written
/// @param x     Logical first element of x (the far end of a negative stride)
/// @param safe1 The nudge, (n + 1) times the underflow threshold
template<typename T, typename R, typename AAt>
__device__ __forceinline__ void la_amv_row(R *const yi, const R alpha, const R beta, const R safe1,
                                           const T *const x, const std::ptrdiff_t incx,
                                           const std::size_t j0, const std::size_t j1,
                                           const AAt &a_at) {
  // SYMB_ZERO starts true when beta * |y(i)| is exactly zero by construction.
  bool symb_zero = true;
  R acc = R{0};
  if (beta != R{0}) {
    acc = *yi;
    if (acc != R{0}) {
      symb_zero = false;
      acc = mul_rn(beta, wwr::fabs(acc));
    }
  }
  if (alpha != R{0}) {
    for (std::size_t j = j0; j < j1; ++j) {
      const R temp = abs1<T, R>(a_at(j));
      const R xa = abs1<T, R>(x[static_cast<std::ptrdiff_t>(j) * incx]);
      symb_zero = symb_zero && (xa == R{0} || temp == R{0});
      acc = add_rn(acc, mul_rn(mul_rn(alpha, xa), temp));
    }
  }
  if (!symb_zero) {
    acc = add_rn(acc, wwr::copysign(safe1, acc));
  }
  *yi = acc;
}

} // namespace calaman::device
