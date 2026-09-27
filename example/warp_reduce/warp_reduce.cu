// warp_reduce.cu
//
// The device-kernel half of the warp-reduction example: one block of four
// warps, every thread cascading over the whole array, then a tile.shfl_down
// ladder in registers. Written once for both backends -- cooperative_groups.cuh
// resolves the only thing that differs (the vendor header), so there is no
// per-backend #if here.
//
// Shared unchanged between backends the way src/extension/rand's kernels are:
// under CUDA the .cu extension is all CMake needs, under HIP this directory's
// CMakeLists.txt forces LANGUAGE CXX back on and links hip::device PRIVATE so
// clang compiles it with `-x hip`.
#include "warp_reduce_bridge.h"

#include "cooperative_groups.cuh"

#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

namespace wwr::example {

namespace {

// Four warps, as parallel_for.cuh uses, and computed the same way:
// WWR_WARP_SIZE is a configure-time value, so this is 128 by default and 256
// on a CDNA build that sets 64.
constexpr std::uint32_t kWarp = static_cast<std::uint32_t>(WWR_WARP_SIZE);
constexpr std::uint32_t kBlockSize = 4 * kWarp;
constexpr std::uint32_t kNumTiles = kBlockSize / kWarp;

static_assert(kBlockSize <= 1024,
              "block size exceeds the 1024 threads/block both backends cap at -- "
              "WWR_WARP_SIZE is too large");

/// @brief [kernel] Single-block sum of `count` elements into `output[0]`
__global__ void warp_reduce_sum_kernel(const float *input, std::size_t count, float *output) {
  __shared__ float partials[kNumTiles];

  const cg::thread_block block = cg::this_thread_block();
  const cg::thread_block_tile<WWR_WARP_SIZE> tile = cg::tiled_partition<WWR_WARP_SIZE>(block);
  const std::uint32_t rank = static_cast<std::uint32_t>(block.thread_rank());

  // Cascade: every thread walks the whole array with stride kBlockSize, so
  // count is unbounded by the launch geometry. Starting from 0 is what makes
  // count < kBlockSize need no guard and count == 0 fall out.
  float value = 0.0F;
  for (std::size_t i = rank; i < count; i += kBlockSize) {
    value += input[i];
  }

  // Sequential addressing in registers. No __syncthreads and no volatile: a
  // shuffle is a warp-level collective, so the lanes stay in step by
  // construction. BlockSize is a multiple of the warp size, so every tile is
  // full and no ladder lane is inactive.
  for (std::uint32_t offset = kWarp / 2; offset > 0; offset /= 2) {
    value += tile.shfl_down(value, offset);
  }

  if (tile.thread_rank() == 0) {
    partials[rank / kWarp] = value;
  }
  block.sync();

  // Cross-warp combine by warp 0 alone. `rank < kWarp` is uniform across its
  // lanes, which is what keeps the second ladder convergent; lanes past
  // kNumTiles feed in 0.
  if (rank < kWarp) {
    value = (rank < kNumTiles) ? partials[rank] : 0.0F;
    for (std::uint32_t offset = kWarp / 2; offset > 0; offset /= 2) {
      value += tile.shfl_down(value, offset);
    }
    if (rank == 0) {
      output[0] = value;
    }
  }
}

} // namespace

void warp_reduce_sum(const gpuStream_t stream, const std::size_t count, const float *input,
                     float *output) {
  warp_reduce_sum_kernel<<<1, kBlockSize, 0, stream>>>(input, count, output);
}

} // namespace wwr::example
