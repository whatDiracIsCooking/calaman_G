/**
 * @file block_reduce.cuh
 * @brief Block-wide shared-memory tree reduction, with the stock fold ops
 *
 * calaman::device::block_reduce folds one value per thread across the block
 * under an associative binary op and returns the fold in every thread. A
 * shared-memory tree, not a warp-shuffle ladder: HIP's tile shfl_down resolves
 * only for vendor scalar types, so a shuffle cannot carry the class-type
 * values (argmax pairs) shared consumers need, and every caller is a small
 * control kernel where the tree's extra barriers cost nothing measurable.
 *
 * The fold is identity-free: only the first @p nactive threads' values enter
 * (reduce_columns.cuh's activity-bound design), so the op needs only
 * associativity -- no identity, no commutativity. Each (kBlock, R, Op)
 * instantiation owns one static kBlock-sized shared buffer per kernel; the
 * leading and trailing barriers make repeated calls over that buffer safe.
 *
 * #include'd into a .cu (CUDA) or -x hip device-compiled (HIP) TU, like
 * elem_ops.cuh: no module face, reached root-relative as
 * "common/block_reduce.cuh". Link calaman.common for the src/ root and,
 * through it, wwr.device (device_guard.h, the device-pass gate).
 *
 * Usage:
 *   #include "common/block_reduce.cuh"
 *
 *   R acc = ...;                              // this thread's partial
 *   acc = block_reduce<kBlock>(acc, AddOp{}); // every thread: the block sum
 */
#pragma once

#include <type_traits>

// device_functor, the constraint on Op below.
#include "device_functor.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile, so this
// header carries no guard of its own, like reduce_columns.cuh.
#include <device_guard.h>

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
                "the halving tree-reduce requires a power-of-two block");
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
  for (unsigned int stride = kBlock / 2; stride > 0; stride >>= 1) {
    if (t < stride && t + stride < nactive) {
      s[t] = op(s[t], s[t + stride]);
    }
    __syncthreads();
  }
  const R r = s[0];
  __syncthreads(); // every thread reads s[0] before a later call may overwrite it
  return r;
}

} // namespace calaman::device
