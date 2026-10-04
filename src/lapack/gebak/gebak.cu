// gebak.cu
//
// The device-kernel half of calaman.gebak: the two kernels of the backward
// transformation (scaling and permutation), plus one launcher per kernel declared
// in gebak_bridge.h. The host driver lives in interface.cppm -- but unlike gebal
// it reads nothing back, so it is thin control flow, not a data-driven loop.
// Shared unchanged between both backends, like gebal.cu: a .cu is compiled by the
// backend compiler, so the raw <<<>>> launch syntax and the device intrinsics
// (__syncthreads) are available directly, and the complex types/accessors arrive
// through complex.h (pulled in by elem_ops.cuh via wwr.device).
//
// Scaling (full-grid): rows ilo..ihi are multiplied by the real factor scale(i)
// (right eigenvectors) or 1/scale(i) (left). These rows sit inside the balanced
// window and the permutation touches only rows outside it, so the two stages are
// disjoint and scaling parallelises freely over the window and the columns.
//
// Permutation (one cooperating block): LAPACK's interchange sequence is
// data-dependent (each target is read from scale(i)) and ORDER-dependent
// (consecutive swaps can share a row), so it cannot parallelise across the grid.
// One block replays the sequence with a __syncthreads() between swaps, cooperating
// on the O(m) column work of each -- the same single-block shape gebal's sweep
// uses, and what keeps the host driver free of any device read-back.
#include "gebak_bridge.h"

#include "complex.h"
#include "common/elem_ops.cuh"

#include <cstddef>

namespace calaman::device {

// complex.h puts the neutral complex types in namespace wwr; pull the two type
// names in so the explicit instantiations below can spell them bare.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

/// Column-major offset of V(i, j).
__device__ __forceinline__ std::size_t idx(const int i, const int j, const int ldv) {
  return static_cast<std::size_t>(j) * static_cast<std::size_t>(ldv) + static_cast<std::size_t>(i);
}

inline int grid_for(const int count, const int block) { return (count + block - 1) / block; }

/// @brief [kernel] Scale rows ilo0..ihi0 of V by scale[i] (or 1/scale[i]).
template<typename T, typename R>
__global__ void scale_rows_kernel(T *__restrict__ V, const int ldv, const int m, const int ilo0,
                                  const int ihi0, const R *__restrict__ scale, const bool invert) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = ilo0 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (c >= m || i > ihi0) {
    return;
  }
  const R s = invert ? (R(1) / scale[i]) : scale[i];
  const std::size_t o = idx(i, c, ldv);
  V[o] = elem_ops<T>::scale(V[o], s);
}

/// @brief [kernel] Apply ?gebak's backward row permutation in a single block.
///
/// All threads evaluate the identical ii-loop control flow, so the branches and
/// the barrier below stay block-uniform; only the per-column swap work is split
/// across the block. ilo/ihi are 1-based, matching LAPACK's indices.
template<typename T, typename R, int BLOCK>
__global__ void permute_kernel(T *V, const int ldv, const int m, const int n, const int ilo,
                               const int ihi, const R *__restrict__ scale) {
  const int t = static_cast<int>(threadIdx.x);
  for (int ii = 1; ii <= n; ++ii) {
    int i = ii;
    if (i >= ilo && i <= ihi) {
      continue; // inside the balanced window: never permuted
    }
    if (i < ilo) {
      i = ilo - ii; // the low block is walked in reverse of the forward phase
    }
    const int k = static_cast<int>(scale[i - 1]);
    if (k == i) {
      continue; // self-map: no interchange recorded
    }
    for (int c = t; c < m; c += BLOCK) {
      const std::size_t oi = idx(i - 1, c, ldv);
      const std::size_t ok = idx(k - 1, c, ldv);
      const T tmp = V[oi];
      V[oi] = V[ok];
      V[ok] = tmp;
    }
    __syncthreads(); // this swap must finish before the next reads a shared row
  }
}

} // namespace

// ── launchers ────────────────────────────────────────────────────────────────

template<typename T, typename R>
void gebak_scale_rows(const wwr::wwrStream_t stream, T *const V, const int ldv, const int m,
                      const int ilo0, const int ihi0, const R *const scale, const bool invert) {
  const int rows = ihi0 - ilo0 + 1;
  if (rows < 1 || m < 1) {
    return;
  }
  const dim3 block(32, 8);
  const dim3 grid(grid_for(m, static_cast<int>(block.x)), grid_for(rows, static_cast<int>(block.y)));
  scale_rows_kernel<T, R><<<grid, block, 0, stream>>>(V, ldv, m, ilo0, ihi0, scale, invert);
}

template<typename T, typename R>
void gebak_permute(const wwr::wwrStream_t stream, T *const V, const int ldv, const int m,
                   const int n, const int ilo, const int ihi, const R *const scale) {
  if (n < 1 || m < 1) {
    return;
  }
  permute_kernel<T, R, kGebakPermuteBlock>
      <<<1, kGebakPermuteBlock, 0, stream>>>(V, ldv, m, n, ilo, ihi, scale);
}

// One instantiation per supported type, matching gebak_bridge.h's declarations and
// interface.cppm's extern template list -- float, double, and the two complex
// types. The real scale type R is ComplexToRealType<T>: float for
// float/wwrFloatComplex, double for double/wwrDoubleComplex.
template void gebak_scale_rows<float, float>(wwr::wwrStream_t, float *, int, int, int, int,
                                             const float *, bool);
template void gebak_scale_rows<double, double>(wwr::wwrStream_t, double *, int, int, int, int,
                                               const double *, bool);
template void gebak_scale_rows<wwrFloatComplex, float>(wwr::wwrStream_t, wwrFloatComplex *, int, int,
                                                       int, int, const float *, bool);
template void gebak_scale_rows<wwrDoubleComplex, double>(wwr::wwrStream_t, wwrDoubleComplex *, int,
                                                         int, int, int, const double *, bool);

template void gebak_permute<float, float>(wwr::wwrStream_t, float *, int, int, int, int, int,
                                          const float *);
template void gebak_permute<double, double>(wwr::wwrStream_t, double *, int, int, int, int, int,
                                            const double *);
template void gebak_permute<wwrFloatComplex, float>(wwr::wwrStream_t, wwrFloatComplex *, int, int,
                                                    int, int, int, const float *);
template void gebak_permute<wwrDoubleComplex, double>(wwr::wwrStream_t, wwrDoubleComplex *, int, int,
                                                      int, int, int, const double *);

} // namespace calaman::device
