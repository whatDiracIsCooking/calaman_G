/**
 * @file lartg.cuh
 * @brief lartg_scalar: one ?lartg plane rotation, per thread, on the device
 *
 * The scalar core of reference ?lartg (LAPACK 3.12.0's safe-scaling algorithm),
 * shared by calaman.lartg's batched functor and every kernel that generates a
 * rotation inside its own launch (?steqr's chase). The four thresholds are
 * precision constants; build them once on the host with lartg_thresholds<T>()
 * and hand them to the kernel.
 *
 * Reached root-relative as "lapack/lartg/lartg.cuh" by linking the INTERFACE
 * target calaman::lartg::header (the src/ root plus wwr.device for
 * wrappers/math/math.cuh). Device-only: include it from a .cu.
 *
 * Usage:
 *   #include "lapack/lartg/lartg.cuh"
 *   const auto th = calaman::device::lartg_thresholds<T>(); // host side
 *   T c, s, r;
 *   calaman::device::lartg_scalar(f, g, th, &c, &s, &r);    // device side
 */

#pragma once

#include <wrappers/math/math.cuh>

#include <cmath>
#include <limits>

namespace calaman::device {

/// @brief ?lartg's la_constants: safmin (smallest normal), safmax = 1/safmin,
///        rtmin = sqrt(safmin), rtmax = sqrt(safmax/2)
template<typename T>
struct LartgThresholds {
  T safmin;
  T safmax;
  T rtmin;
  T rtmax;
};

/// @brief The thresholds in precision T, computed on the host
template<typename T>
LartgThresholds<T> lartg_thresholds() {
  const T safmin = std::numeric_limits<T>::min();
  const T safmax = T{1} / safmin;
  return {safmin, safmax, std::sqrt(safmin), std::sqrt(safmax / T{2})};
}

/// @brief ?lartg: (c, s, r) with [c s; -s c] [f; g] = [r; 0], c >= 0
///
/// The g == 0 / f == 0 special cases, the unscaled fast path inside
/// [rtmin, rtmax], and the scaled fallback, exactly as the reference. SIGN(a, b)
/// is copysign, reached only where b is non-zero.
template<typename T>
__device__ __forceinline__ void lartg_scalar(const T f, const T g, const LartgThresholds<T> th,
                                             T *const c, T *const s, T *const r) {
  const T f1 = wwr::fabs(f);
  const T g1 = wwr::fabs(g);
  if (g == T{0}) {
    *c = T{1};
    *s = T{0};
    *r = f;
  } else if (f == T{0}) {
    *c = T{0};
    *s = wwr::copysign(T{1}, g);
    *r = g1;
  } else if (f1 > th.rtmin && f1 < th.rtmax && g1 > th.rtmin && g1 < th.rtmax) {
    const T d = wwr::sqrt(f * f + g * g);
    *c = f1 / d;
    *r = wwr::copysign(d, f);
    *s = g / *r;
  } else {
    const T u = wwr::fmin(th.safmax, wwr::fmax(wwr::fmax(th.safmin, f1), g1));
    const T fs = f / u;
    const T gs = g / u;
    const T d = wwr::sqrt(fs * fs + gs * gs);
    *c = wwr::fabs(fs) / d;
    const T rr = wwr::copysign(d, f);
    *s = gs / rr;
    *r = rr * u;
  }
}

} // namespace calaman::device
