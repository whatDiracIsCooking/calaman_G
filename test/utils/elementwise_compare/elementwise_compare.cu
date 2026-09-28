// elementwise_compare.cu
//
// The device-kernel half of calaman.test.elementwise_compare: the two stages
// count_mismatches runs on the GPU. Shared unchanged between both backends --
// under CUDA the .cu extension is all CMake needs, under HIP this directory's
// CMakeLists.txt forces LANGUAGE CXX back on so clang compiles it with -x hip.
//
// Stage 1 (write_mismatch_flags) is an index-per-thread map, so it reuses
// wwr.extension.parallel_for rather than a hand-written launcher. Stage 2
// (reduce_count) is the custom one-block reduction: no such primitive exists in
// the gpu* layer, and one block keeps the result a single scalar with no second
// pass.
#include "elementwise_compare_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::test::device {

namespace {

/// @brief Writes the per-element mismatch flag, one index per thread
///
/// Members are const so the functor is not copy-assignable, which is what
/// parallel_for's device_functor concept checks for immutability.
template<typename T>
struct mismatch_functor {
  const T *const a_;
  const T *const b_;
  unsigned int *const diffs_;

  __device__ void operator()(const std::size_t i) const {
    diffs_[i] = (a_[i] != b_[i]) ? 1u : 0u;
  }
};

/// @brief Writes the per-element tolerance flag, one index per thread
///
/// abs() is spelled by hand (no <cmath> in a kernel functor, and it keeps the
/// functor type-agnostic): a signed difference negated when below zero.
template<typename T>
struct tolerance_functor {
  const T *const a_;
  const T *const b_;
  unsigned int *const flags_;
  const T atol_;
  const T rtol_;

  __device__ void operator()(const std::size_t i) const {
    const T d = a_[i] - b_[i];
    const T abs_d = d < T(0) ? -d : d;
    const T abs_b = b_[i] < T(0) ? -b_[i] : b_[i];
    flags_[i] = (abs_d > atol_ + rtol_ * abs_b) ? 1u : 0u;
  }
};

/// @brief Writes the per-element absolute difference, one index per thread
template<typename T>
struct abs_diff_functor {
  const T *const a_;
  const T *const b_;
  T *const out_;

  __device__ void operator()(const std::size_t i) const {
    const T d = a_[i] - b_[i];
    out_[i] = d < T(0) ? -d : d;
  }
};

/// One block, so the reduction lands in a single scalar with no inter-block
/// step. A power of two makes the tree halving below exact.
constexpr unsigned int kReduceBlock = 256;

/// @brief [kernel] Sum diffs[0, n) into *result with one block
///
/// Each thread strides over the array (grid-stride within the single block)
/// into a register, then the block tree-reduces through shared memory.
__global__ void reduce_count_kernel(const unsigned int *diffs, unsigned int *result,
                                    const std::size_t n) {
  __shared__ unsigned int scratch[kReduceBlock];

  unsigned int local = 0;
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
    local += diffs[i];
  }
  scratch[threadIdx.x] = local;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = scratch[0];
  }
}

/// @brief [kernel] Maximum of in[0, n) into *result with one block
///
/// Same shape as reduce_count_kernel with max in place of +. The identity is
/// 0, valid because every input here is an absolute value, hence non-negative;
/// threads past `n` contribute 0 and cannot beat a real element.
template<typename T>
__global__ void reduce_max_kernel(const T *in, T *result, const std::size_t n) {
  __shared__ T scratch[kReduceBlock];

  T local = T(0);
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
    const T v = in[i];
    local = v > local ? v : local;
  }
  scratch[threadIdx.x] = local;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const T other = scratch[threadIdx.x + stride];
      if (other > scratch[threadIdx.x]) {
        scratch[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *result = scratch[0];
  }
}

} // namespace

template<typename T>
void write_mismatch_flags(const wwr::wwrStream_t stream, const T *a, const T *b,
                          unsigned int *diffs, const std::size_t n) {
  if (n < 1) {
    return;
  }
  const mismatch_functor<T> functor{a, b, diffs};
  wwr::extension::parallel_for(stream, n, functor);
}

template<typename T>
void write_tolerance_flags(const wwr::wwrStream_t stream, const T *a, const T *b,
                           unsigned int *flags, const std::size_t n, const T atol, const T rtol) {
  if (n < 1) {
    return;
  }
  const tolerance_functor<T> functor{a, b, flags, atol, rtol};
  wwr::extension::parallel_for(stream, n, functor);
}

template<typename T>
void write_abs_diff(const wwr::wwrStream_t stream, const T *a, const T *b, T *out,
                    const std::size_t n) {
  if (n < 1) {
    return;
  }
  const abs_diff_functor<T> functor{a, b, out};
  wwr::extension::parallel_for(stream, n, functor);
}

void reduce_count(const wwr::wwrStream_t stream, const unsigned int *diffs, unsigned int *result,
                  const std::size_t n) {
  reduce_count_kernel<<<1, kReduceBlock, 0, stream>>>(diffs, result, n);
}

template<typename T>
void reduce_max(const wwr::wwrStream_t stream, const T *in, T *result, const std::size_t n) {
  reduce_max_kernel<T><<<1, kReduceBlock, 0, stream>>>(in, result, n);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's -- all these lists cover the same types.
template void write_mismatch_flags<float>(wwr::wwrStream_t, const float *, const float *,
                                          unsigned int *, std::size_t);
template void write_mismatch_flags<double>(wwr::wwrStream_t, const double *, const double *,
                                           unsigned int *, std::size_t);
template void write_tolerance_flags<float>(wwr::wwrStream_t, const float *, const float *,
                                           unsigned int *, std::size_t, float, float);
template void write_tolerance_flags<double>(wwr::wwrStream_t, const double *, const double *,
                                            unsigned int *, std::size_t, double, double);
template void write_abs_diff<float>(wwr::wwrStream_t, const float *, const float *, float *,
                                    std::size_t);
template void write_abs_diff<double>(wwr::wwrStream_t, const double *, const double *, double *,
                                     std::size_t);
template void reduce_max<float>(wwr::wwrStream_t, const float *, float *, std::size_t);
template void reduce_max<double>(wwr::wwrStream_t, const double *, double *, std::size_t);

} // namespace calaman::test::device
