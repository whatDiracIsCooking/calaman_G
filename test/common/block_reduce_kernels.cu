/**
 * @file block_reduce_kernels.cu
 * @brief Device half of the block_reduce suite: one single-block kernel per
 *        (overload, block size, op), behind block_reduce_bridge.h's launchers
 *
 * Written once for both backends; the device pass is selected by how the TU
 * is compiled (nvcc under CUDA, clang -x hip under HIP), not by an #if here.
 */
#include "block_reduce_bridge.h"

#include "mat2.h"

#include "common/block_reduce.cuh"

#include <runtime.h>

#include <cstddef>

namespace cg = cooperative_groups;

namespace calaman::test {

namespace {

/// @brief mat2_mul as a fold op
struct Mat2Op {
  __device__ unsigned long long operator()(const unsigned long long x,
                                           const unsigned long long y) const {
    return mat2_mul(x, y);
  }
};

template<unsigned int kBlock, BlockReduceShape kShape, typename Op, typename T>
__global__ void block_reduce_kernel(const T *in, const unsigned int nactive, const bool chained,
                                    T *out) {
  const unsigned int t = threadIdx.x;
  T r{};
  // `chained` is block-uniform, so every thread still reaches the barriers.
  if constexpr (kShape == BlockReduceShape::kTree) {
    r = device::block_reduce<kBlock>(in[t], Op{}, nactive);
    if (chained) {
      r = device::block_reduce<kBlock>(static_cast<T>(in[t] + r), Op{}, nactive);
    }
  } else {
    // Alternating buffers is the barrier-free reuse: the second call's writes
    // land in s1, so they cannot race a slow warp still reading s0.
    __shared__ T s0[kBlock / kWarpSize];
    __shared__ T s1[kBlock / kWarpSize];
    const cg::thread_block block = cg::this_thread_block();
    r = device::block_reduce(block, s0, in[t], Op{}, nactive);
    if (chained) {
      r = device::block_reduce(block, s1, static_cast<T>(in[t] + r), Op{}, nactive);
    }
  }
  out[t] = r;
}

template<unsigned int kBlock, typename Op, typename T>
void launch(const BlockReduceShape shape, const T *d_in, const unsigned int nactive,
            const bool chained, T *d_out) {
  switch (shape) {
  case BlockReduceShape::kTree:
    block_reduce_kernel<kBlock, BlockReduceShape::kTree, Op><<<1, kBlock>>>(d_in, nactive,
                                                                            chained, d_out);
    break;
  case BlockReduceShape::kBlock:
    block_reduce_kernel<kBlock, BlockReduceShape::kBlock, Op><<<1, kBlock>>>(d_in, nactive,
                                                                             chained, d_out);
    break;
  }
}

/// Allocate, copy in, launch one block, copy out, free; first failure wins.
template<typename Op, typename T>
int run(const BlockReduceShape shape, const unsigned int warps, const T *host_in,
        const unsigned int nactive, const bool chained, T *host_out) {
  if (warps != 1 && warps != 4) {
    return -1;
  }
  const std::size_t bytes = block_threads(warps) * sizeof(T);
  T *d_in = nullptr;
  T *d_out = nullptr;
  wwr::wwrError_t err = wwr::wwrMalloc(reinterpret_cast<void **>(&d_in), bytes);
  if (err == wwr::wwrSuccess) {
    err = wwr::wwrMalloc(reinterpret_cast<void **>(&d_out), bytes);
  }
  if (err == wwr::wwrSuccess) {
    err = wwr::wwrMemcpy(d_in, host_in, bytes, wwr::wwrMemcpyHostToDevice);
  }
  if (err == wwr::wwrSuccess) {
    if (warps == 1) {
      launch<kWarpSize, Op>(shape, d_in, nactive, chained, d_out);
    } else {
      launch<4 * kWarpSize, Op>(shape, d_in, nactive, chained, d_out);
    }
    err = wwr::wwrGetLastError();
  }
  if (err == wwr::wwrSuccess) {
    err = wwr::wwrMemcpy(host_out, d_out, bytes, wwr::wwrMemcpyDeviceToHost); // synchronizes
  }
  const wwr::wwrError_t free_in = wwr::wwrFree(d_in);
  const wwr::wwrError_t free_out = wwr::wwrFree(d_out);
  if (err == wwr::wwrSuccess) {
    err = (free_in != wwr::wwrSuccess) ? free_in : free_out;
  }
  return static_cast<int>(err);
}

} // namespace

unsigned int block_threads(const unsigned int warps) { return warps * kWarpSize; }

int block_reduce_sum(const BlockReduceShape shape, const unsigned int warps, const float *host_in,
                     const unsigned int nactive, float *host_out) {
  return run<device::AddOp>(shape, warps, host_in, nactive, false, host_out);
}

int block_reduce_chained_sum(const BlockReduceShape shape, const unsigned int warps,
                             const float *host_in, const unsigned int nactive,
                             float *host_out) {
  return run<device::AddOp>(shape, warps, host_in, nactive, true, host_out);
}

int block_reduce_max_nan(const BlockReduceShape shape, const unsigned int warps,
                         const float *host_in, const unsigned int nactive, float *host_out) {
  return run<device::MaxNanOp>(shape, warps, host_in, nactive, false, host_out);
}

int block_reduce_mat2(const BlockReduceShape shape, const unsigned int warps,
                      const unsigned long long *host_in, const unsigned int nactive,
                      unsigned long long *host_out) {
  return run<Mat2Op>(shape, warps, host_in, nactive, false, host_out);
}

} // namespace calaman::test
