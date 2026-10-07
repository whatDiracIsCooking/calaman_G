/**
 * @file aberth.cuh
 * @brief aberth_warp: every root of one polynomial, by Aberth-Ehrlich, on one warp
 *
 * calaman::device::aberth_warp runs the simultaneous Aberth-Ehrlich iteration for
 * p(z) = a_0 + a_1 z + ... + a_n z^n on a kWarpSize-wide thread_block_tile:
 * root i belongs to lane i % kWarpSize, and the estimates live in a caller-owned
 * shared buffer that every lane reads. Updates are Jacobi-style (all roots move
 * at once, ping-ponging between the buffer's two halves), so the result does not
 * depend on lane scheduling. Coefficients are real or complex; roots are complex.
 *
 * Constraints, one line each:
 *   - a_n != 0; the degree n is a runtime value, any n >= 1 fits in 2n slots.
 *   - |z| > 1 is evaluated through the reversed polynomial in 1/z: no |z|^n overflow.
 *   - Cost is O(n^2 / kWarpSize) per lane per iteration (the Aberth sum).
 *
 * #include'd into a .cu (CUDA) or -x hip device-compiled (HIP) TU: no module
 * face, reached root-relative as "polynomial_roots/aberth.cuh". Link calaman.common for
 * the src/ root and, through it, wwr.device.
 *
 * Usage:
 *   #include "polynomial_roots/aberth.cuh"
 *
 *   __shared__ wwrDoubleComplex roots[2 * kMaxDegree];
 *   auto tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
 *   const int iters = aberth_warp(tile, d_coeffs, n, roots, 1e-14, 100);
 */
#pragma once

#include "common/block_params.h"
// MaxNanOp: the convergence fold, where a NaN correction must not read as done.
#include "common/block_reduce.cuh"
#include "common/constants.h"
#include "common/elem_ops.cuh"
#include "common/fp_types.h"
#include "common/warp_reduce.cuh"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <cooperative_groups.h>
#include <wrappers/math/math.cuh>

// FLT/DBL_EPSILON: nvcc rejects std::numeric_limits in device code.
#include <cfloat>
#include <concepts>

namespace calaman::device {

namespace cg = cooperative_groups;

/// @brief 1 / d, pre-scaled by |re|+|im| so |d|^2 cannot overflow or underflow
///
/// cuCdiv's guard with the numerator fixed at 1; hipCdiv has no guard at all.
/// d == 0 gives NaN, as a general complex division would.
template<complex_fp CT>
__device__ __forceinline__ CT aberth_recip(const CT d) {
  using R = ComplexToRealType<CT>;
  const R re = elem_ops<CT>::real_part(d);
  const R im = elem_ops<CT>::imag_part(d);
  const R rs = kOne<R> / (wwr::fabs(re) + wwr::fabs(im));
  const R br = re * rs;
  const R bi = im * rs;
  const R u = rs / (br * br + bi * bi);
  return make_complex(br * u, -bi * u);
}

/// @brief Every root of a_0 + a_1 z + ... + a_n z^n, iterated across @p warp_tile
///
/// Every lane must call, with identical arguments. Returns the iteration count
/// at which every root settled -- relative correction |w| <= @p tol |z|, or
/// |p(z)| inside Horner's rounding bound -- else -1 (NaN never settles). The
/// final estimates are in shared_roots[0, n) on return, either way.
///
/// @tparam ParentT     Deduced, so tiled_partition's result binds as-is
/// @tparam CT          Deduced from @p shared_roots: the complex type of T's precision
/// @param coeffs       a_0 .. a_n, ascending, any address space; a_n != 0
/// @param n            The degree, n >= 1
/// @param shared_roots Shared scratch of 2n slots; no lane may touch it on entry
///                     (warp_tile.sync() between reuses)
template<typename ParentT, usual_fp T, complex_fp CT>
  requires std::same_as<ComplexToRealType<CT>, ComplexToRealType<T>>
__device__ int aberth_warp(cg::thread_block_tile<kWarpSize, ParentT> &warp_tile,
                           const T *const coeffs, const int n, CT *const shared_roots,
                           const ComplexToRealType<T> tol, const int max_iter) {
  using R = ComplexToRealType<T>;

  const int lane = static_cast<int>(warp_tile.thread_rank());
  const auto coeff = [coeffs](const int k) {
    return make_complex(elem_ops<T>::real_part(coeffs[k]), elem_ops<T>::imag_part(coeffs[k]));
  };

  // Initial estimates: n points on the circle of radius |a_0 / a_n|^(1/n) -- the
  // geometric mean of the root moduli -- turned off the real axis by a fixed
  // angle so a real polynomial's conjugate pairs are not started symmetric.
  {
    R radius =
        wwr::pow(elem_ops<T>::modulus(coeffs[0]) / elem_ops<T>::modulus(coeffs[n]), kOne<R> / R(n));
    if (!(radius > kZero<R>) || !(radius - radius == kZero<R>)) {
      radius = kOne<R>; // a_0 == 0, or the ratio overflowed: any distinct start works
    }
    const R step = kTwo<R> * kPi<R> / R(n);
    for (int i = lane; i < n; i += kWarpSize) {
      const R theta = step * R(i) + R(0.4);
      shared_roots[i] = make_complex(radius * wwr::cos(theta), radius * wwr::sin(theta));
    }
  }
  warp_tile.sync();

  // gamma_2n ~ 2n unit roundoffs: |p(z)| below gamma * sum |a_k| |z|^k is
  // rounding noise, and further corrections cannot improve z.
  const R gamma = R(n) * (sizeof(R) == sizeof(float) ? R(FLT_EPSILON) : R(DBL_EPSILON));
  int cur = 0; // the half of shared_roots holding the current estimates
  int iters = -1;
  for (int iter = 1; iter <= max_iter; ++iter) {
    const CT *const z = shared_roots + cur * n;
    CT *const z_next = shared_roots + (1 - cur) * n;
    R worst = kZero<R>;
    for (int i = lane; i < n; i += kWarpSize) {
      const CT zi = z[i];
      // Newton's ratio as num / den = p / p'. Inside the unit disc, p and p' by
      // Horner; outside it, the reversal q(y) = y^n p(1/y) at y = 1/zi, where
      // p'/p = y (n q - y q') / q -- so no power of |zi| > 1 is ever formed.
      // bound is sum |a_k| |x|^k for the polynomial actually evaluated.
      const R zmod = elem_ops<CT>::modulus(zi);
      const bool inside = zmod <= kOne<R>;
      const CT x = inside ? zi : elem_ops<CT>::div(kOne<CT>, zi);
      const R xmod = elem_ops<CT>::modulus(x);
      const int k0 = inside ? n : 0; // the leading coefficient, then step toward the other end
      const int step = inside ? -1 : 1;
      CT num = coeff(k0);
      CT dnum = kZero<CT>;
      R bound = elem_ops<T>::modulus(coeffs[k0]);
      for (int k = k0 + step; k >= 0 && k <= n; k += step) {
        dnum = elem_ops<CT>::add(elem_ops<CT>::mul(dnum, x), num);
        num = elem_ops<CT>::add(elem_ops<CT>::mul(num, x), coeff(k));
        bound = bound * xmod + elem_ops<T>::modulus(coeffs[k]);
      }
      const CT den = inside ? dnum
                            : elem_ops<CT>::mul(x, elem_ops<CT>::sub(elem_ops<CT>::scale(num, R(n)),
                                                                     elem_ops<CT>::mul(x, dnum)));
      CT s = kZero<CT>;
      for (int j = 0; j < n; ++j) {
        if (j != i) {
          s = elem_ops<CT>::add(s, aberth_recip(elem_ops<CT>::sub(zi, z[j])));
        }
      }
      // w = N / (1 - N s) with N = num / den, rearranged so den == 0 is no special case.
      const CT w = elem_ops<CT>::div(num, elem_ops<CT>::sub(den, elem_ops<CT>::mul(num, s)));
      z_next[i] = elem_ops<CT>::sub(zi, w);
      const R wmod = elem_ops<CT>::modulus(w);
      // wmod == 0 is settled even at zi == 0, where the ratio would be 0/0. A NaN
      // num fails the <= and keeps its NaN ratio, so it still reaches the fold.
      const bool noise = elem_ops<CT>::modulus(num) <= gamma * bound;
      worst = max_nan(worst, (noise || wmod == kZero<R>) ? kZero<R> : wmod / zmod);
    }
    worst = warp_reduce(warp_tile, worst, MaxNanOp{});
    // Orders this iteration's z_next writes before the next one reads them, and
    // its z reads before the next one overwrites that half.
    warp_tile.sync();
    cur = 1 - cur;
    if (worst <= tol) {
      iters = iter;
      break;
    }
  }

  if (cur == 1) {
    for (int i = lane; i < n; i += kWarpSize) {
      shared_roots[i] = shared_roots[n + i];
    }
    warp_tile.sync();
  }
  return iters;
}

} // namespace calaman::device
