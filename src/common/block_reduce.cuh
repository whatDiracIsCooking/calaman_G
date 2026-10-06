/**
 * @file block_reduce.cuh
 * @brief Block-wide reductions, with the stock fold ops
 *
 * calaman::device::block_reduce folds one value per thread across the block
 * under an associative binary op and returns the fold in every thread. The
 * arguments pick the mechanism:
 *
 * | Call                                   | Values             | Mechanism                      |
 * |----------------------------------------|--------------------|--------------------------------|
 * | block_reduce<kBlock>(v, op)            | trivially copyable | shared-memory tree             |
 * | block_reduce(block, shared_arr, v, op) | arithmetic         | warp_reduce, then across warps |
 *
 * The tree carries class types (complex, argmax pairs) that HIP's tile
 * shuffles cannot (warp_reduce.cuh). Every fold is identity-free: only the
 * first @p nactive threads' values enter (reduce_columns.cuh's activity-bound
 * design), so the op needs associativity only. The tree owns one static shared
 * buffer per instantiation, made reusable by its leading barrier; the warp
 * overload takes the caller's buffer and leaves reuse to the caller.
 *
 * #include'd into a .cu (CUDA) or -x hip device-compiled (HIP) TU: no module
 * face, reached root-relative as "common/block_reduce.cuh". Link calaman.common
 * for the src/ root and, through it, wwr.device (cooperative_groups.h,
 * device_guard.h, runtime.h).
 *
 * Usage:
 *   #include "common/block_reduce.cuh"
 *
 *   acc = block_reduce<kBlock>(acc, AddOp{});                            // tree
 *
 *   __shared__ float partials[kNumWarps];
 *   acc = block_reduce(cg::this_thread_block(), partials, acc, AddOp{}); // shuffles
 */
#pragma once

#include <type_traits>

#include "align_up.h"
#include "block_params.h"
// device_functor, the constraint on Op below.
#include "device_functor.h"
#include "warp_reduce.cuh"

// The device-pass gate: #errors outside a CUDA or HIP device compile, so this
// header carries no guard of its own, like reduce_columns.cuh.
#include <device_guard.h>

// __forceinline__, __syncthreads, threadIdx: nvcc pre-includes the runtime,
// HIP does not.
#include <runtime.h>

namespace calaman::device {

/// The larger of two, with NaN winning: a NaN must not be dropped by a maximum.
template<typename R>
__device__ __forceinline__ R max_nan(const R a, const R b) {
  return (a != a || a > b) ? a : b;
}

/// @brief Associative fold: addition
struct AddOp {
  template<typename R>
  __device__ R operator()(const R a, const R b) const {
    return a + b;
  }
};

/// @brief Associative fold: maximum with NaN winning (max_nan)
struct MaxNanOp {
  template<typename R>
  __device__ R operator()(const R a, const R b) const {
    return max_nan(a, b);
  }
};

/// @brief Fold @p v across the block under @p op, result valid in every thread
///
/// Threads with threadIdx.x >= @p nactive contribute nothing (their @p v is
/// ignored, so no identity is needed) but must still call: the barriers are
/// block-wide collectives.
///
/// @tparam kBlock  blockDim.x of the calling kernel; a power of two
/// @param nactive  Threads whose @p v enters the fold; 1 <= nactive <= kBlock
template<unsigned int kBlock, typename R, device_functor Op>
__device__ __forceinline__ R block_reduce(const R v, const Op op,
                                          const unsigned int nactive = kBlock) {
  static_assert(kBlock > 0 && (kBlock & (kBlock - 1)) == 0,
                "the pairwise tree-reduce requires a power-of-two block");
  static_assert(kBlock <= 1024,
                "block size exceeds the 1024 threads/block both backends cap at");
  static_assert(std::is_trivially_copyable_v<R>, "shared-memory staging copies R bytewise");

  __shared__ R s[kBlock];
  const unsigned int t = threadIdx.x;
  __syncthreads(); // a previous call's reads of s must finish before this one writes
  if (t < nactive) {
    s[t] = v;
  }
  __syncthreads();
  // Offsets grow, as in warp_reduce: s[t] for t a multiple of 2 * offset holds
  // the fold of the contiguous [t, t + offset) and appends [t + offset, ...) on
  // the right. A halving stride would fold s[0] with s[kBlock / 2] first and
  // silently need commutativity.
  for (unsigned int offset = 1; offset < kBlock; offset <<= 1) {
    if (t % (2 * offset) == 0 && t + offset < nactive) {
      s[t] = op(s[t], s[t + offset]);
    }
    __syncthreads();
  }
  const R r = s[0];
  __syncthreads(); // every thread reads s[0] before a later call may overwrite it
  return r;
}

/// @brief Fold arithmetic @p v across the block under @p op, through warp_reduce
///
/// Each warp folds its lanes, then every warp folds the per-warp partials in
/// warp order, so the order is thread order and associativity suffices. Every
/// thread must call, as for the tree overload. One barrier per call: the
/// caller orders any earlier use of @p shared_arr before entry.
///
/// @tparam kNumWarps   Warps in the calling kernel's block: blockDim.x / kWarpSize
/// @param block        The calling 1-D thread block, partitioned into warp tiles here
/// @param shared_arr   Shared scratch for the per-warp partials, unused when
///                     kNumWarps == 1. No thread may still read or write it on
///                     entry: block.sync() between reuses, or alternate buffers.
/// @param nactive      Threads whose @p v enters the fold;
///                     1 <= nactive <= kNumWarps * kWarpSize
template<unsigned int kNumWarps, typename T, device_functor Op>
  requires std::is_arithmetic_v<T> && num_warps<kNumWarps>
__device__ __forceinline__ T block_reduce(const cg::thread_block &block,
                                          T (&shared_arr)[kNumWarps], const T v, const Op op,
                                          const unsigned int nactive = kNumWarps * kWarpSize) {
  auto warp_tile = cg::tiled_partition<kWarpSize>(block);
  // nactive <= kNumWarps * kWarpSize is unchecked in both branches: past it,
  // this one folds out-of-tile shuffles (a lane's own value, twice) and the
  // other reads shared_arr out of bounds.
  if constexpr (kNumWarps == 1) {
    return warp_reduce(warp_tile, v, op, nactive);
  } else {
    // num_warps caps kNumWarps at 1024 / kWarpSize <= kWarpSize, so the
    // second level is one warp_reduce.
    const unsigned int warpIdx = warp_tile.meta_group_rank();
    const unsigned int first = warpIdx * kWarpSize;
    const unsigned int num_warps_needed = idivup(nactive, kWarpSize);
    // Warp-uniform branch: a warp past nactive skips its shuffles as a whole.
    if (first < nactive) {
      const unsigned int n = nactive - first < kWarpSize ? nactive - first : kWarpSize;
      const T partial = warp_reduce(warp_tile, v, op, n);
      if (warp_tile.thread_rank() == 0) {
        shared_arr[warpIdx] = partial;
      }
    }
    block.sync();
    // Every warp folds the partials itself: no broadcast, so no third barrier.
    // Lanes >= num_warps_needed read a valid slot that warp_reduce then ignores.
    const unsigned int lane = warp_tile.thread_rank();
    return warp_reduce(warp_tile, shared_arr[lane < num_warps_needed ? lane : 0], op,
                       num_warps_needed);
  }
}

} // namespace calaman::device
