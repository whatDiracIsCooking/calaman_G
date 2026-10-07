/**
 * @file lapy3.cuh
 * @brief Device port of LAPACK's ?lapy3 -- sqrt(x^2 + y^2 + z^2) without
 *        spurious overflow
 *
 * Header-only `__device__` helper, ported verbatim from LAPACK 3.12.0 so a
 * kernel that calls it per-thread reproduces the reference's branch choices.
 * Real types only (float, double).
 *
 * Reached root-relative off the src/ root as "lapack/lapy3/lapy3.cuh"; link the
 * INTERFACE target calaman::lapy3::header, which carries that root and
 * wwr.device (for wrappers/math/math.cuh). Device-only: include it from a .cu,
 * never from a module interface. The batched oracle wrapper is calaman.lapy3.
 *
 * Usage:
 *   #include "lapack/lapy3/lapy3.cuh"
 *
 *   const T r = calaman::device::lapy3_scalar(x, y, z);
 */

#pragma once

#include <wrappers/math/math.cuh>

#include <cfloat>
#include <type_traits>

namespace calaman::device {

/// @brief ?lapy3: sqrt(x^2 + y^2 + z^2) without spurious overflow
///
/// A NaN argument yields NaN and an infinite one +inf, both through the
/// reference's |x| + |y| + |z| fallback. The overflow threshold is
/// DLAMCH('O'), spelled FLT_MAX / DBL_MAX.
template<typename T>
__device__ __forceinline__ T lapy3_scalar(const T x, const T y, const T z) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "lapy3_scalar is real-only (float, double)");
  T huge_val;
  if constexpr (std::is_same_v<T, float>) {
    huge_val = FLT_MAX;
  } else {
    huge_val = DBL_MAX;
  }
  const T xabs = wwr::fabs(x);
  const T yabs = wwr::fabs(y);
  const T zabs = wwr::fabs(z);
  // fmax drops a NaN operand, as gfortran's MAX does: w == 0 then catches
  // max(0, NaN, 0), and the sum keeps the NaN alive.
  const T w = wwr::fmax(wwr::fmax(xabs, yabs), zabs);
  if (w == T{0} || w > huge_val) {
    return xabs + yabs + zabs;
  }
  const T xs = xabs / w;
  const T ys = yabs / w;
  const T zs = zabs / w;
  return w * wwr::sqrt(xs * xs + ys * ys + zs * zs);
}

} // namespace calaman::device
