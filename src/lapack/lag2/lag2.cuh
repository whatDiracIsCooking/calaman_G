/**
 * @file lag2.cuh
 * @brief lag2_convert: one element of LAPACK's ?lag2? / ?lat2?, with the
 *        narrowing overflow check
 *
 * Converts one element between precisions exactly as dlag2s/zlag2c and
 * slag2d/clag2z do. Narrowing (double to single) overflows when any real or
 * imaginary part lies outside +-SLAMCH('O') = FLT_MAX, by the reference's own
 * comparisons (so NaN converts and does not overflow); an overflowing element
 * is not stored. Widening cannot overflow.
 *
 * calaman.lag2's kernel and calaman.lat2's are both built on this. Reach it as
 * "lapack/lag2/lag2.cuh" by linking the INTERFACE target calaman::lag2::header.
 * Device-only: include it from a .cu.
 *
 * Usage:
 *   #include "lapack/lag2/lag2.cuh"
 *   if (!calaman::device::lag2_convert(a[k], sa[k2])) { *info = 1; }
 */

#pragma once

#include "common/elem_ops.cuh"
#include <complex.h>

#include <cfloat>

namespace calaman::device {

/// @brief True when @p x lies outside +-SLAMCH('O'), by ?lag2s's comparisons
__device__ __forceinline__ bool lag2_overflows(const double x) {
  constexpr double rmax = FLT_MAX;
  return x < -rmax || x > rmax;
}

/// @brief sa <- a (narrowing); false, and sa untouched, if a overflows single
__device__ __forceinline__ bool lag2_convert(const double a, float &sa) {
  if (lag2_overflows(a)) {
    return false;
  }
  sa = static_cast<float>(a);
  return true;
}

/// @brief sa <- a (narrowing); false, and sa untouched, if either part overflows
__device__ __forceinline__ bool lag2_convert(const wwr::wwrDoubleComplex a,
                                             wwr::wwrFloatComplex &sa) {
  const double re = elem_ops<wwr::wwrDoubleComplex>::real_part(a);
  const double im = elem_ops<wwr::wwrDoubleComplex>::imag_part(a);
  if (lag2_overflows(re) || lag2_overflows(im)) {
    return false;
  }
  sa = make_complex(static_cast<float>(re), static_cast<float>(im));
  return true;
}

/// @brief a <- sa (widening, exact); always true
__device__ __forceinline__ bool lag2_convert(const float sa, double &a) {
  a = static_cast<double>(sa);
  return true;
}

/// @brief a <- sa (widening, exact); always true
__device__ __forceinline__ bool lag2_convert(const wwr::wwrFloatComplex sa,
                                             wwr::wwrDoubleComplex &a) {
  a = make_complex(static_cast<double>(elem_ops<wwr::wwrFloatComplex>::real_part(sa)),
                   static_cast<double>(elem_ops<wwr::wwrFloatComplex>::imag_part(sa)));
  return true;
}

} // namespace calaman::device
