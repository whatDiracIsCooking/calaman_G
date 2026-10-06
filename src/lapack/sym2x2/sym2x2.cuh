/**
 * @file sym2x2.cuh
 * @brief Device ports of LAPACK's ?lae2, ?laev2 and ?lapy2 -- the scalar
 *        helpers of the symmetric tridiagonal QL/QR chase (sterf, steqr)
 *
 * Header-only `__device__` functions, ported verbatim from LAPACK 3.12.0 so a
 * kernel that calls them per-thread reproduces the reference's branch choices:
 * lae2_scalar (eigenvalues of [a b; b c]), laev2_scalar (plus the eigenvector
 * rotation) and lapy2_scalar (overflow-safe sqrt(x^2 + y^2), NaN-propagating).
 * Real types only (float, double).
 *
 * Reached root-relative off the src/ root as "lapack/sym2x2/sym2x2.cuh"; link
 * the INTERFACE target calaman::sym2x2, which carries that root and wwr.device
 * (for wrappers/math/math.cuh). Device-only: include it from a .cu, never from a
 * module interface. The batched oracle wrappers are calaman.lae2, calaman.laev2
 * and calaman.lapy2.
 *
 * Usage:
 *   #include "lapack/sym2x2/sym2x2.cuh"
 *
 *   T rt1, rt2, cs1, sn1;
 *   calaman::device::laev2_scalar(a, b, c, &rt1, &rt2, &cs1, &sn1);
 */

#pragma once

#include <wrappers/math/math.cuh>

#include <cfloat>
#include <type_traits>

namespace calaman::device {

/// @brief ?lae2: eigenvalues of the symmetric 2x2 [a b; b c]
///
/// rt1 is the eigenvalue of larger absolute value, rt2 the other. The smaller
/// one is formed as det/rt1 to stay accurate, exactly as the reference does.
template<typename T>
__device__ __forceinline__ void lae2_scalar(const T a, const T b, const T c, T *rt1, T *rt2) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "lae2_scalar is real-only (float, double)");
  const T sm = a + c;
  const T df = a - c;
  const T adf = wwr::fabs(df);
  const T tb = b + b;
  const T ab = wwr::fabs(tb);
  T acmx;
  T acmn;
  if (wwr::fabs(a) > wwr::fabs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }
  T rt;
  if (adf > ab) {
    rt = adf * wwr::sqrt(T{1} + (ab / adf) * (ab / adf));
  } else if (adf < ab) {
    rt = ab * wwr::sqrt(T{1} + (adf / ab) * (adf / ab));
  } else {
    rt = ab * wwr::sqrt(T{2}); // includes ab == adf == 0
  }
  // The reference's evaluation order for rt2 is what keeps it accurate; do not
  // fold it into one expression.
  if (sm < T{0}) {
    *rt1 = T{0.5} * (sm - rt);
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else if (sm > T{0}) {
    *rt1 = T{0.5} * (sm + rt);
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else {
    *rt1 = T{0.5} * rt; // includes rt1 == rt2 == 0
    *rt2 = T{-0.5} * rt;
  }
}

/// @brief ?laev2: ?lae2's eigenvalues plus the unit eigenvector (cs1, sn1) of rt1
///
/// [cs1 sn1; -sn1 cs1] * [a b; b c] * [cs1 -sn1; sn1 cs1] = diag(rt1, rt2).
template<typename T>
__device__ __forceinline__ void laev2_scalar(const T a, const T b, const T c, T *rt1, T *rt2,
                                             T *cs1, T *sn1) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "laev2_scalar is real-only (float, double)");
  const T sm = a + c;
  const T df = a - c;
  const T adf = wwr::fabs(df);
  const T tb = b + b;
  const T ab = wwr::fabs(tb);
  T acmx;
  T acmn;
  if (wwr::fabs(a) > wwr::fabs(c)) {
    acmx = a;
    acmn = c;
  } else {
    acmx = c;
    acmn = a;
  }
  T rt;
  if (adf > ab) {
    rt = adf * wwr::sqrt(T{1} + (ab / adf) * (ab / adf));
  } else if (adf < ab) {
    rt = ab * wwr::sqrt(T{1} + (adf / ab) * (adf / ab));
  } else {
    rt = ab * wwr::sqrt(T{2});
  }
  int sgn1;
  if (sm < T{0}) {
    *rt1 = T{0.5} * (sm - rt);
    sgn1 = -1;
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else if (sm > T{0}) {
    *rt1 = T{0.5} * (sm + rt);
    sgn1 = 1;
    *rt2 = (acmx / *rt1) * acmn - (b / *rt1) * b;
  } else {
    *rt1 = T{0.5} * rt;
    *rt2 = T{-0.5} * rt;
    sgn1 = 1;
  }
  int sgn2;
  T cs;
  if (df >= T{0}) {
    cs = df + rt;
    sgn2 = 1;
  } else {
    cs = df - rt;
    sgn2 = -1;
  }
  T c1;
  T s1;
  if (wwr::fabs(cs) > ab) {
    const T ct = -tb / cs;
    s1 = T{1} / wwr::sqrt(T{1} + ct * ct);
    c1 = ct * s1;
  } else if (ab == T{0}) {
    c1 = T{1};
    s1 = T{0};
  } else {
    const T tn = -cs / tb;
    c1 = T{1} / wwr::sqrt(T{1} + tn * tn);
    s1 = tn * c1;
  }
  if (sgn1 == sgn2) {
    const T tn = c1;
    c1 = -s1;
    s1 = tn;
  }
  *cs1 = c1;
  *sn1 = s1;
}

/// @brief ?lapy2: sqrt(x^2 + y^2) without spurious overflow
///
/// A NaN argument is returned as is (y's when both are NaN, as the reference
/// does); an infinite one yields +inf. The overflow threshold is DLAMCH('O'),
/// spelled FLT_MAX / DBL_MAX.
template<typename T>
__device__ __forceinline__ T lapy2_scalar(const T x, const T y) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "lapy2_scalar is real-only (float, double)");
  const bool x_is_nan = wwr::isnan(x);
  const bool y_is_nan = wwr::isnan(y);
  if (y_is_nan) {
    return y;
  }
  if (x_is_nan) {
    return x;
  }
  T huge_val;
  if constexpr (std::is_same_v<T, float>) {
    huge_val = FLT_MAX;
  } else {
    huge_val = DBL_MAX;
  }
  const T xabs = wwr::fabs(x);
  const T yabs = wwr::fabs(y);
  const T w = xabs > yabs ? xabs : yabs;
  const T z = xabs < yabs ? xabs : yabs;
  if (z == T{0} || w > huge_val) {
    return w;
  }
  return w * wwr::sqrt(T{1} + (z / w) * (z / w));
}

} // namespace calaman::device
