// lanst.cu
//
// The device half of calaman.lanst: the ?lanst norm of a symmetric tridiagonal
// (d, e) in ONE single-block launch. Each thread takes a contiguous chunk of
// rows and folds its share of the norm; block_reduce (common/block_reduce.cuh)
// then folds the per-thread partials, and thread 0 writes the result.
//
//   max_abs  ('M'): lanst_max_abs (lanst.h) over the chunk, then max
//   one/inf  ('1'/'I', equal by symmetry): per-row |e(i-1)|+|d(i)|+|e(i)|, max
//   frobenius('F'): sum d(i)^2 + 2 e(i)^2, then sqrt -- a plain sum, not ?lassq
//
// One block is deliberate: a tridiagonal of order n holds 2n-1 numbers, so a
// block of threads covers any n this library's O(n^2)-and-up callers reach. The
// max folds propagate a NaN from either operand, matching DLANST's DISNAN guard.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lanst_bridge.h"

#include "lanst.h"

#include "common/block_reduce.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

template<typename T>
__device__ T abs_of(const T x) {
  return x < T{0} ? -x : x;
}

/// @brief [kernel] One block folds the @p which norm of (d, e) into *d_result
template<typename T>
__global__ void lanst_kernel(const MatrixNorm which, const std::size_t n, const T *const d,
                             const T *const e, T *const d_result) {
  const std::size_t chunk = (n + kBlock - 1) / kBlock;
  const unsigned int nactive = static_cast<unsigned int>((n + chunk - 1) / chunk);
  const std::size_t lo = threadIdx.x * chunk;
  const std::size_t hi = lo + chunk < n ? lo + chunk : n;

  T acc = T{0};
  if (lo < n) {
    switch (which) {
    case MatrixNorm::max_abs:
      // The sequential helper over this chunk's d[lo, hi) and e[lo, hi-1); the
      // e(hi-1) linking it to the next chunk is folded here, by its left owner.
      acc = lanst_max_abs(hi - lo, d + lo, e + lo);
      if (hi < n) {
        acc = max_nan(acc, abs_of(e[hi - 1]));
      }
      break;
    case MatrixNorm::one:
    case MatrixNorm::inf:
      for (std::size_t i = lo; i < hi; ++i) {
        T row = abs_of(d[i]);
        if (i > 0) {
          row += abs_of(e[i - 1]);
        }
        if (i + 1 < n) {
          row += abs_of(e[i]);
        }
        acc = (i == lo) ? row : max_nan(acc, row);
      }
      break;
    case MatrixNorm::frobenius:
      for (std::size_t i = lo; i < hi; ++i) {
        acc += d[i] * d[i];
        if (i + 1 < n) {
          acc += T{2} * e[i] * e[i];
        }
      }
      break;
    }
  }

  const T total = (which == MatrixNorm::frobenius) ? block_reduce<kBlock>(acc, AddOp{}, nactive)
                                                   : block_reduce<kBlock>(acc, MaxNanOp{}, nactive);
  if (threadIdx.x == 0) {
    if (which == MatrixNorm::frobenius) {
      if constexpr (std::is_same_v<T, float>) {
        *d_result = sqrtf(total);
      } else {
        *d_result = sqrt(total);
      }
    } else {
      *d_result = total;
    }
  }
}

} // namespace

template<typename T>
void lanst(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
           const T *const d_d, const T *const d_e, T *const d_result) {
  lanst_kernel<T><<<1, kBlock, 0, stream>>>(which, n, d_d, d_e, d_result);
}

// One per supported type, matching lanst_bridge.h and interface.cppm's
// `extern template` list -- float and double.
template void lanst<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                           const float *, float *);
template void lanst<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                            const double *, double *);

} // namespace calaman::device
