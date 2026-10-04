/**
 * @file inv_sqrt.cu
 * @brief The inverse_sqrt device kernel -- the one genuinely per-element step of
 *        the symmetric inverse square root
 *
 * Launches a single parallel_for functor that maps the eigenvalue vector Lambda
 * to Lambda^{-1/2} in place, flooring near-null / non-positive modes to 0. The
 * floor is checked before the square root, so wwr::sqrt never sees a
 * zero-or-negative argument and the result is never NaN/Inf. The column scaling
 * and recombination that surround this (dgmm, gemm) are stock BLAS, so they stay
 * on the host side in interface.cppm -- only the flooring rsqrt needs a kernel.
 */

#include "inv_sqrt_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "wrappers/math/math.cuh"

#include <cstdint>

namespace calaman::device {
namespace {

/// @brief data[i] <- (data[i] <= floor) ? 0 : 1/sqrt(data[i]). The pointer and
/// scalar members are const but non-class, so the functor stays trivially
/// copyable -- what parallel_for's device_functor requires.
template<typename T>
struct inverse_sqrt_functor {
  T *__restrict__ const data_;
  const T floor_;

  __device__ __forceinline__ void operator()(const std::int32_t i) const {
    const T value = data_[i];
    data_[i] = value <= floor_ ? T{0} : T{1} / wwr::sqrt(value);
  }
};

} // namespace

template<typename T>
void inverse_sqrt(const wwr::wwrStream_t stream, const std::int32_t count, T *data, const T floor) {
  if (count < 1) {
    return;
  }
  const inverse_sqrt_functor<T> functor{data, floor};
  wwr::extension::parallel_for<std::int32_t>(stream, count, functor);
}

template void inverse_sqrt<float>(wwr::wwrStream_t, std::int32_t, float *, float);
template void inverse_sqrt<double>(wwr::wwrStream_t, std::int32_t, double *, double);

} // namespace calaman::device
