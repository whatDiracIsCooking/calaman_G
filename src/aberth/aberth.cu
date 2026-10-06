/**
 * @file aberth.cu
 * @brief calaman.aberth's device side: the batched launch over aberth_warp
 *
 * One warp per polynomial, up to kMaxWarpsPerBlock per block as the 48 KiB
 * dynamic-shared budget allows; each warp owns a 2n slice of that buffer.
 * Instantiated here for the four type pairs aberth_bridge.h's callers use.
 */

#include "aberth_bridge.h"

#include "aberth/aberth.cuh"
#include "common/align_up.h"
#include "common/block_params.h"

#include <complex.h>
#include <cooperative_groups.h>
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

namespace {

/// Warps (polynomials) per block, when the shared budget allows that many.
constexpr int kMaxWarpsPerBlock = 4;

// One warp per polynomial: warp w of block blk takes polynomial
// blk * warps_per_block + w and the w-th 2n slice of the dynamic shared buffer.
template<typename T, typename CT>
__global__ void aberth_kernel(const int n, const int batch, const int warps_per_block,
                              const T *const coeffs, CT *const roots, int *const iters,
                              const ComplexToRealType<T> tol, const int max_iter) {
  // One untemplated declaration for every instantiation (a templated extern
  // __shared__ array conflicts across them); 16-byte elements align any CT.
  extern __shared__ wwr::wwrDoubleComplex aberth_smem[];
  auto warp_tile = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const int w = static_cast<int>(warp_tile.meta_group_rank());
  const int b = static_cast<int>(blockIdx.x) * warps_per_block + w;
  // Warp-uniform exit: aberth's collectives are tile-wide, never block-wide.
  if (b >= batch) {
    return;
  }
  CT *const shared_roots =
      reinterpret_cast<CT *>(aberth_smem) + static_cast<std::size_t>(w) * 2 * n;
  const std::size_t base = static_cast<std::size_t>(b);
  const int it = aberth_warp(warp_tile, coeffs + base * (n + 1), n, shared_roots, tol, max_iter);
  for (int i = static_cast<int>(warp_tile.thread_rank()); i < n; i += kWarpSize) {
    roots[base * n + i] = shared_roots[i];
  }
  if (warp_tile.thread_rank() == 0) {
    iters[b] = it;
  }
}

} // namespace

template<typename T, typename CT, typename R>
void aberth(const wwr::wwrStream_t stream, const int n, const int batch, const T *const coeffs,
            CT *const roots, int *const iters, const R tol, const int max_iter) {
  const std::size_t per_warp = 2 * static_cast<std::size_t>(n) * sizeof(CT);
  const std::size_t fit = kAberthMaxSharedBytes / per_warp;
  const int warps = fit < kMaxWarpsPerBlock ? static_cast<int>(fit) : kMaxWarpsPerBlock;
  const auto grid = static_cast<unsigned int>(idivup(batch, warps));
  aberth_kernel<T, CT>
      <<<grid, static_cast<unsigned int>(warps) * kWarpSize, warps * per_warp, stream>>>(
          n, batch, warps, coeffs, roots, iters, tol, max_iter);
}

// One per supported pair, matching interface.cppm's extern-template list.
template void aberth<float, wwr::wwrFloatComplex, float>(wwr::wwrStream_t, int, int, const float *,
                                                         wwr::wwrFloatComplex *, int *, float, int);
template void aberth<double, wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, int, int,
                                                            const double *, wwr::wwrDoubleComplex *,
                                                            int *, double, int);
template void aberth<wwr::wwrFloatComplex, wwr::wwrFloatComplex, float>(
    wwr::wwrStream_t, int, int, const wwr::wwrFloatComplex *, wwr::wwrFloatComplex *, int *, float,
    int);
template void aberth<wwr::wwrDoubleComplex, wwr::wwrDoubleComplex, double>(
    wwr::wwrStream_t, int, int, const wwr::wwrDoubleComplex *, wwr::wwrDoubleComplex *, int *,
    double, int);

} // namespace calaman::device
