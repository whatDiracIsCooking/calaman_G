// la_wwaddw.cu
//
// The device half of calaman.la_wwaddw: one thread per element of the
// ?LA_WWADDW loop, in 1-D blocks of 4*WWR_WARP_SIZE.
//
// The body is an error-free transformation, so it is only right under strict
// left-to-right IEEE evaluation: each add and subtract below is rounded on its
// own, in the reference's order, and nothing is contracted or re-associated.
// CUDA spells that with the _rn intrinsics, which nvcc never fuses or reorders;
// HIP spells plain operators under `#pragma clang fp contract(off)`, because
// HIP's _rn functions are plain operators that its default fp-contract=fast
// fuses (la_geamv/README.md, decision 2). No fast-math reaches this TU.
#include "la_wwaddw_bridge.h"

#include "common/align_up.h"
#include "common/elem_ops.cuh"
#include <runtime.h>

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

/// @brief a + b, rounded once
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

/// @brief a - b, rounded once
template<typename R>
__device__ __forceinline__ R sub_rn(const R a, const R b) {
#if defined(__HIP__)
#pragma clang fp contract(off)
  return a - b;
#else
  if constexpr (std::is_same_v<R, float>) {
    return __fsub_rn(a, b);
  } else {
    return __dsub_rn(a, b);
  }
#endif
}

/// @brief One real component of the ?LA_WWADDW body, statement for statement
template<typename R>
__device__ __forceinline__ void wwaddw(R &x, R &y, const R w) {
  // S = X + W: the rounded high word.
  R s = add_rn(x, w);
  // S = (S + S) - S: the reference's forced round-trip (an identity in IEEE
  // binary arithmetic, short of S + S overflowing; kept, not folded away).
  s = sub_rn(add_rn(s, s), s);
  // Y = ((X - S) + W) + Y: (X - S) + W is the error X + W lost to rounding,
  // exact only in this grouping; it is then folded into the old low word.
  y = add_rn(add_rn(sub_rn(x, s), w), y);
  x = s;
}

/// @brief [kernel] (x(i), y(i)) <-- (x(i), y(i)) + w(i), per component
template<typename T>
__global__ void la_wwaddw_kernel(T *const x, T *const y, const T *const w, const std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  using ops = elem_ops<T>;
  using R = typename ops::real_type;
  if constexpr (std::is_same_v<T, R>) {
    wwaddw(x[i], y[i], w[i]);
  } else {
    // Fortran's COMPLEX + and - act per component, so the body runs twice.
    R xr = ops::real_part(x[i]);
    R xi = ops::imag_part(x[i]);
    R yr = ops::real_part(y[i]);
    R yi = ops::imag_part(y[i]);
    wwaddw(xr, yr, ops::real_part(w[i]));
    wwaddw(xi, yi, ops::imag_part(w[i]));
    x[i] = make_complex(xr, xi);
    y[i] = make_complex(yr, yi);
  }
}

} // namespace

template<typename T>
void la_wwaddw(const wwr::wwrStream_t stream, const std::size_t n, T *const x, T *const y,
               const T *const w) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const dim3 grid(idivup<std::size_t>(n, kBlockSize));
  const dim3 block(kBlockSize);
  la_wwaddw_kernel<T><<<grid, block, 0, stream>>>(x, y, w, n);
}

// One per supported type, matching interface.cppm's `extern template` list and
// instantiations.cpp's.
template void la_wwaddw<float>(wwr::wwrStream_t, std::size_t, float *, float *, const float *);
template void la_wwaddw<double>(wwr::wwrStream_t, std::size_t, double *, double *, const double *);
template void la_wwaddw<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t, wwr::wwrFloatComplex *,
                                              wwr::wwrFloatComplex *, const wwr::wwrFloatComplex *);
template void la_wwaddw<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t,
                                               wwr::wwrDoubleComplex *, wwr::wwrDoubleComplex *,
                                               const wwr::wwrDoubleComplex *);

} // namespace calaman::device
