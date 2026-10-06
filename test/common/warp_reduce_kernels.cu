/**
 * @file warp_reduce_kernels.cu
 * @brief Device half of the warp_reduce suite: one single-tile kernel per op,
 *        behind the plain-type launchers warp_reduce_bridge.h declares
 *
 * Written once for both backends; the device pass is selected by how the TU
 * is compiled (nvcc under CUDA, clang -x hip under HIP), not by an #if here.
 */
#include "warp_reduce_bridge.h"

#include "mat2.h"

#include "common/block_reduce.cuh" // AddOp, MaxNanOp
#include "common/warp_reduce.cuh"

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

template<typename T, typename Op>
__global__ void warp_reduce_kernel(const T *in, const unsigned int nactive, T *out) {
  cg::thread_block_tile<kWarpSize, cg::thread_block> tile =
      cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const unsigned int lane = tile.thread_rank();
  out[lane] = device::warp_reduce(tile, in[lane], Op{}, nactive);
}

/// Allocate, copy in, launch one tile, copy out, free; first failure wins.
template<typename T, typename Op>
int run(const T *host_in, const unsigned int nactive, T *host_out) {
  const std::size_t bytes = kWarpSize * sizeof(T);
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
    warp_reduce_kernel<T, Op><<<1, kWarpSize>>>(d_in, nactive, d_out);
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

unsigned int warp_size() { return kWarpSize; }

int warp_reduce_sum(const float *host_in, const unsigned int nactive, float *host_out) {
  return run<float, device::AddOp>(host_in, nactive, host_out);
}

int warp_reduce_max_nan(const float *host_in, const unsigned int nactive, float *host_out) {
  return run<float, device::MaxNanOp>(host_in, nactive, host_out);
}

int warp_reduce_mat2(const unsigned long long *host_in, const unsigned int nactive,
                     unsigned long long *host_out) {
  return run<unsigned long long, Mat2Op>(host_in, nactive, host_out);
}

} // namespace calaman::test
