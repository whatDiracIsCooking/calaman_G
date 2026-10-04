/**
 * @file ladiv.h
 * @brief The scalar kernel of calaman.ladiv -- real-arithmetic complex division
 *        by Smith's algorithm, callable from host AND device code
 *
 * calaman::ladiv_scalar(a, b, c, d, &p, &q) writes (p + i*q) = (a + i*b) /
 * (c + i*d) in pure real arithmetic. Smith's algorithm divides through by the
 * LARGER of |c|, |d| first (t = d/c, den = c + d*t, or the c/d symmetric branch)
 * so the intermediate den never overflows when the two parts differ wildly in
 * magnitude -- the whole reason ?ladiv exists over the naive (ac+bd)/(c^2+d^2).
 *
 * It is `CLM_HOST_DEVICE` so laln2's kernel can call it per-thread: in a device
 * compile (__CUDACC__/__HIP__) the macro is `__host__ __device__`, so the body
 * gets device linkage; in a host compile it is empty, so the identical body
 * compiles for the host (the interface unit re-exports it, the oracle test and
 * the batched launcher below call it). A plain inline template, no attributes,
 * would NOT be callable from a __global__ kernel -- hence the macro, not the
 * attribute-free shape calaman.lanv2 gets away with (lanv2 runs host-only).
 *
 * Shared by ladiv.cu (the batched device launcher the oracle exercises), which
 * includes it bare as a same-directory header, and later by laln2's own .cu,
 * which will reach it root-relative ("lapack/ladiv/ladiv.h") off the src/ root -- the
 * two-consumer split constants.h uses. The interface unit does NOT include it:
 * a kernel reaches this by #include, not by import, so it is not module-exported.
 */

#pragma once

// __host__ __device__ in a device pass; empty on a host compile so the same
// template body serves the interface unit and the oracle. The device macros name
// the vendor's device-compile passes (CUDA nvcc, HIP clang) without pulling a
// vendor header -- the attribute tokens are keywords in those passes.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#define CLM_HOST_DEVICE __host__ __device__
#else
#define CLM_HOST_DEVICE
#endif

namespace calaman {

/// @brief Real-arithmetic complex division (a+ib)/(c+id) by Smith's algorithm
///
/// Writes the quotient's real part to @p p and imaginary part to @p q, choosing
/// the branch off whichever of |c|, |d| is larger so the denominator stays in
/// range -- LAPACK ?ladiv's safeguard against spurious overflow when |c| and |d|
/// are far apart. Uses only compare, multiply, divide: no library call, so the
/// body is identical host and device (@c CLM_HOST_DEVICE makes it callable from a
/// kernel). The divisor c + i*d must be non-zero.
///
/// @tparam T Real element type; one of the instantiated types (float, double)
/// @param a Real part of the numerator
/// @param b Imaginary part of the numerator
/// @param c Real part of the denominator
/// @param d Imaginary part of the denominator
/// @param p Out: real part of the quotient
/// @param q Out: imaginary part of the quotient
template<typename T>
CLM_HOST_DEVICE void ladiv_scalar(const T a, const T b, const T c, const T d, T *p, T *q) {
  // abs() inlined as a ternary so the body needs no <cmath> / wwr::fabs: it must
  // compile unchanged in a host TU and a device TU, and a bare conditional is
  // portable to both where a library call is not.
  const T abs_c = c < T{0} ? -c : c;
  const T abs_d = d < T{0} ? -d : d;
  if (abs_d < abs_c) {
    // |d| < |c|: divide through by c.
    const T t = d / c;
    const T den = c + d * t;
    *p = (a + b * t) / den;
    *q = (b - a * t) / den;
  } else {
    // |c| <= |d|: the symmetric branch, divide through by d.
    const T t = c / d;
    const T den = d + c * t;
    *p = (b + a * t) / den;
    *q = (-a + b * t) / den;
  }
}

} // namespace calaman
