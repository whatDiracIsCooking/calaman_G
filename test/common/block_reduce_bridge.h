/**
 * @file block_reduce_bridge.h
 * @brief Plain-type launchers between block_reduce_tests.cpp (host, GoogleTest)
 *        and block_reduce_kernels.cu (device pass)
 *
 * Builtin types only cross it, as in warp_reduce_bridge.h. Each launcher runs
 * ONE block of block_threads(warps) threads, reducing through the overload
 * @p shape names, and copies every thread's result back into
 * host_out[0, block_threads(warps)), synchronously. @p warps is 1 (the
 * single-warp case) or 4. Returns the runtime's status code, 0 on success, or
 * -1 for an unsupported @p warps.
 */
#pragma once

namespace calaman::test {

/// @brief Which block_reduce overload a launcher calls
enum class BlockReduceShape : int {
  kTree,  ///< block_reduce<kBlock>(v, op): the shared-memory tree
  kBlock, ///< block_reduce(this_thread_block(), shared_arr, v, op): warp shuffles
};

/// @brief Threads in a block of @p warps warps: the length of host_in / host_out
unsigned int block_threads(unsigned int warps);

/// @brief block_reduce under AddOp
int block_reduce_sum(BlockReduceShape shape, unsigned int warps, const float *host_in,
                     unsigned int nactive, float *host_out);

/// @brief Two back-to-back AddOp calls, the second over in[t] + the first's sum
int block_reduce_chained_sum(BlockReduceShape shape, unsigned int warps, const float *host_in,
                             unsigned int nactive, float *host_out);

/// @brief block_reduce under MaxNanOp
int block_reduce_max_nan(BlockReduceShape shape, unsigned int warps, const float *host_in,
                         unsigned int nactive, float *host_out);

/// @brief block_reduce under mat2_mul (mat2.h), the order probe
int block_reduce_mat2(BlockReduceShape shape, unsigned int warps,
                      const unsigned long long *host_in, unsigned int nactive,
                      unsigned long long *host_out);

} // namespace calaman::test
