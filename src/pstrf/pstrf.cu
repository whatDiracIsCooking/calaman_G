// pstrf.cu
//
// The device-kernel half of calaman.pstrf: the two stages of pivoted Cholesky
// that no BLAS call expresses. Everything numeric in pstrf() is a host
// composition of wrapped BLAS (syrk/gemv/scal/swap) over the calaman.pstf2
// fallback; this file owns only
//
//   * the fused diagonal-maintenance-and-pivot reduction (pstrf_pivot): a
//     single-block argmax over the running Schur-complement diagonals of the
//     trailing range [j, n);
//   * zeroing the trailing factor after a rank-revealing stop
//     (pstrf_zero_trailing): elementwise over the trailing triangle, so it
//     rides wwr.extension.parallel_for.
//
// Structurally these mirror calaman.pstf2's kernels; they stay here, not shared,
// because the blocked driver owns its panel bookkeeping (issue #65) and the
// module boundary keeps each routine's device archive its own. Shared unchanged
// between both backends, like lacpy.cu; the .cu extension is all CMake needs
// (under HIP the CMakeLists forces -x hip), so no __CUDACC__.
#include "pstrf_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

constexpr int kPstrfBlock = 256;

/// @brief [kernel] Fuse one step's diagonal update with the pivot argmax
///
/// Single block, grid-stride over i in [j, n). Each thread folds the previous
/// factor entry into dots[i] (when add_prev), forms the Schur diagonal
/// A(i,i) - dots[i], and tracks the per-thread max, its index, and whether any
/// value was NaN; a shared-memory tree-reduce combines them. out[0]/out[1]/out[2]
/// receive the max diagonal, its index (as T), and the NaN flag (0 or 1).
template<typename T, int BLOCK>
__global__ void pstrf_pivot_kernel(const int j, const int n, const bool add_prev,
                                   const T *__restrict__ prev, const std::size_t prev_stride,
                                   const T *__restrict__ A, const std::size_t lda,
                                   T *__restrict__ dots, T *__restrict__ out) {
  __shared__ T sval[BLOCK];
  __shared__ int spos[BLOCK];
  __shared__ int snan[BLOCK];
  const int tid = static_cast<int>(threadIdx.x);

  T best = T{0};
  int best_pos = -1;
  int nan = 0;
  for (int i = j + tid; i < n; i += BLOCK) {
    if (add_prev) {
      const T p = prev[static_cast<std::size_t>(i) * prev_stride];
      dots[i] += p * p;
    }
    const T val = A[static_cast<std::size_t>(i) * lda + static_cast<std::size_t>(i)] - dots[i];
    if (val != val) { // NaN compares unequal to itself
      nan = 1;
    }
    if (best_pos < 0 || val > best) {
      best = val;
      best_pos = i;
    }
  }
  sval[tid] = best;
  spos[tid] = best_pos;
  snan[tid] = nan;
  __syncthreads();

  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) {
      snan[tid] |= snan[tid + s];
      const int other = spos[tid + s];
      if (other >= 0 && (spos[tid] < 0 || sval[tid + s] > sval[tid])) {
        sval[tid] = sval[tid + s];
        spos[tid] = other;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[0] = sval[0];
    out[1] = static_cast<T>(spos[0]);
    out[2] = snan[0] ? T{1} : T{0};
  }
}

/// @brief Zero one trailing-block entry if it lies in the stored triangle
///
/// Trivially copyable with only const scalar members, as device_functor
/// requires. idx enumerates the (n - rank) x (n - rank) trailing block in
/// column-major order; the entry is zeroed when it sits in the referenced
/// triangle (on/above the diagonal for upper, on/below for lower).
template<typename T>
struct ZeroTrailingFunctor {
  T *const A_;
  const std::size_t lda_;
  const int rank_;
  const int width_;
  const bool upper_;

  __device__ void operator()(const std::size_t idx) const {
    const int a = static_cast<int>(idx % static_cast<std::size_t>(width_)); // row offset
    const int b = static_cast<int>(idx / static_cast<std::size_t>(width_)); // col offset
    const int r = rank_ + a;
    const int c = rank_ + b;
    const bool in_triangle = upper_ ? (r <= c) : (r >= c);
    if (in_triangle) {
      A_[static_cast<std::size_t>(c) * lda_ + static_cast<std::size_t>(r)] = T{0};
    }
  }
};

} // namespace

template<typename T>
void pstrf_pivot(const wwr::wwrStream_t stream, const int j, const int n, const bool add_prev,
                 const T *const prev, const std::size_t prev_stride, const T *const A,
                 const std::size_t lda, T *const dots, T *const out) {
  if (j >= n) {
    return;
  }
  pstrf_pivot_kernel<T, kPstrfBlock>
      <<<1, kPstrfBlock, 0, stream>>>(j, n, add_prev, prev, prev_stride, A, lda, dots, out);
}

template<typename T>
void pstrf_zero_trailing(const wwr::wwrStream_t stream, const bool upper, const int rank,
                         const int n, T *const A, const std::size_t lda) {
  const int width = n - rank;
  if (width <= 0) {
    return;
  }
  const ZeroTrailingFunctor<T> functor{A, lda, rank, width, upper};
  wwr::extension::parallel_for<std::size_t>(
      stream, static_cast<std::size_t>(width) * static_cast<std::size_t>(width), functor);
}

// One per supported type (float, double), matching pstrf_bridge.h and the
// module's use sites.
template void pstrf_pivot<float>(wwr::wwrStream_t, int, int, bool, const float *, std::size_t,
                                 const float *, std::size_t, float *, float *);
template void pstrf_pivot<double>(wwr::wwrStream_t, int, int, bool, const double *, std::size_t,
                                  const double *, std::size_t, double *, double *);

template void pstrf_zero_trailing<float>(wwr::wwrStream_t, bool, int, int, float *, std::size_t);
template void pstrf_zero_trailing<double>(wwr::wwrStream_t, bool, int, int, double *, std::size_t);

} // namespace calaman::device
