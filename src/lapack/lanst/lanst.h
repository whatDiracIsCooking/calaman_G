/**
 * @file lanst.h
 * @brief lanst_max_abs: ?lanst('M') of a symmetric tridiagonal, one thread
 *
 * The largest |entry| of the tridiagonal (d, e) over a caller-chosen sub-range,
 * sequentially, by whichever single thread calls it -- the per-block
 * `?lanst('M', ...)` that ?sterf and ?steqr take inside their own kernels.
 * `CLM_HOST_DEVICE`, so the identical body also runs on the host.
 *
 * Downstream .cu files reach it root-relative as "lapack/lanst/lanst.h" by
 * linking the INTERFACE target calaman::lanst::header (the src/ root; nothing
 * else -- the body uses only compare and negate). Not module-exported: a kernel
 * reaches it by #include, not import.
 *
 * Usage (DSTERF's `ANORM = DLANST('M', LEND-L+1, D(L), E(L))`, 0-based):
 *   #include "lapack/lanst/lanst.h"
 *   const T anorm = calaman::lanst_max_abs(lend - l + 1, d + l, e + l);
 */

#pragma once

#include "common/host_device.h"

#include <cstddef>

namespace calaman {

/// @brief max(|d[0..n)|, |e[0..n-1)|), with a NaN anywhere winning (DLANST 'M')
///
/// Reads exactly @p n entries of @p d and n-1 of @p e (so @p e may be null when
/// n <= 1); returns 0 when n is 0, as DLANST does.
///
/// @tparam T Real element type (float, double)
/// @param n  Order of the (sub-)tridiagonal
/// @param d  Its diagonal, length n
/// @param e  Its off-diagonal, length n-1
template<typename T>
CLM_HOST_DEVICE T lanst_max_abs(const std::size_t n, const T *const d, const T *const e) {
  T anorm = T{0};
  // `x != x` is isnan without <cmath>; a NaN, once in, never leaves (DLANST's
  // `ANORM.LT.SUM .OR. DISNAN(SUM)`), so the walk stops at the first one.
  const std::size_t count = n == 0 ? 0 : 2 * n - 1;
  for (std::size_t k = 0; k < count && anorm == anorm; ++k) {
    // Interleaved d[0], e[0], d[1], ..., d[n-1]: even k is d, odd k is e.
    const T x = (k % 2 == 0) ? d[k / 2] : e[k / 2];
    const T a = x < T{0} ? -x : x;
    if (anorm < a || a != a) {
      anorm = a;
    }
  }
  return anorm;
}

} // namespace calaman
