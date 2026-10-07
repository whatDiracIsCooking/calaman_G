/**
 * @file larmm.cuh
 * @brief Device port of LAPACK's ?larmm -- the scale factor that keeps the
 *        update C - A*B from overflowing (latrs3's per-block safeguard)
 *
 * Header-only `__device__` helper, ported verbatim from LAPACK 3.12.0 so a
 * kernel that calls it per-thread reproduces the reference's choice of scale.
 * Real types only (float, double).
 *
 * Reached root-relative off the src/ root as "lapack/larmm/larmm.cuh"; link the
 * INTERFACE target calaman::larmm::header, which carries that root and
 * wwr.device (for runtime.h). Device-only:
 * include it from a .cu, never from a module interface. The batched oracle
 * wrapper is calaman.larmm.
 *
 * Usage:
 *   #include "lapack/larmm/larmm.cuh"
 *
 *   const T s = calaman::device::larmm_scalar(anorm, bnorm, cnorm);
 */

#pragma once

#include <runtime.h> // __forceinline__ is a HIP runtime macro, not a keyword

#include <cfloat>
#include <type_traits>

namespace calaman::device {

/// @brief ?larmm: scale s in (0, 1] so that s * (cnorm + anorm * bnorm) stays
///        below the overflow threshold
///
/// The norms are non-negative. Returns 1, 1/2 or 1/(2*bnorm), exactly as the
/// reference does. smlnum is DLAMCH('S') / DLAMCH('P'), spelled
/// FLT_MIN / FLT_EPSILON (DBL_ for double).
template<typename T>
__device__ __forceinline__ T larmm_scalar(const T anorm, const T bnorm, const T cnorm) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "larmm_scalar is real-only (float, double)");
  T smlnum;
  if constexpr (std::is_same_v<T, float>) {
    smlnum = FLT_MIN / FLT_EPSILON;
  } else {
    smlnum = DBL_MIN / DBL_EPSILON;
  }
  const T bignum = (T{1} / smlnum) / T{4};
  if (bnorm <= T{1}) {
    if (anorm * bnorm > bignum - cnorm) {
      return T{0.5};
    }
  } else if (anorm > (bignum - cnorm) / bnorm) {
    return T{0.5} / bnorm;
  }
  return T{1};
}

} // namespace calaman::device
