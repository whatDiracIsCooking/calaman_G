// langb.cu
//
// The device half of calaman.langb: the ?langb norm of an n-by-n band matrix
// (kl sub-, ku super-diagonals) in LAPACK band storage, A(i,j) at
// AB(ku+i-j, j), in two launches. Stage one is one block per index j, folding
// its share into scratch[j] through common/block_reduce.cuh:
//
//   max_abs  ('M'): max |A(i,j)| over the band of column j
//   one      ('1'): sum |A(i,j)| over the band of column j
//   inf      ('I'): sum |A(j,k)| over the band of ROW j
//   frobenius('F'): sum |A(i,j)|^2 over the band of column j
//
// Stage two is one block folding scratch (max, or sum then sqrt for 'F'). Only
// in-matrix band entries are addressed, so the unused corners of AB and the
// ldab padding may hold anything. The max folds start from 0 and propagate a
// NaN, as DLANGB's DISNAN guard does; 'F' is a plain sum of squares, not ?lassq
// (the calaman.lange precedent). A complex T has a real norm R.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "langb_bridge.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

/// @brief |x|^2 as re^2 + im^2 (for real T, im is 0)
template<typename T, typename R>
__device__ R entry_sq(const T x) {
  const R re = elem_ops<T>::real_part(x);
  const R im = elem_ops<T>::imag_part(x);
  return re * re + im * im;
}

/// @brief [kernel] Block j writes column j's (row j's, for inf) partial
template<typename T, typename R>
__global__ void langb_partials_kernel(const MatrixNorm which, const std::size_t n,
                                      const std::size_t kl, const std::size_t ku, const T *const AB,
                                      const std::size_t ldab, R *const scratch) {
  const std::size_t j = blockIdx.x;
  R acc = R{0};
  if (which == MatrixNorm::inf) {
    // Row j's band: columns k in [max(0, j-kl), min(n, j+ku+1)).
    const std::size_t k_lo = j > kl ? j - kl : 0;
    const std::size_t k_hi = j + ku + 1 < n ? j + ku + 1 : n;
    for (std::size_t k = k_lo + threadIdx.x; k < k_hi; k += kBlock) {
      acc += elem_ops<T>::modulus(AB[ku + j - k + k * ldab]);
    }
  } else {
    // Column j's band: rows i in [max(0, j-ku), min(n, j+kl+1)).
    const std::size_t i_lo = j > ku ? j - ku : 0;
    const std::size_t i_hi = j + kl + 1 < n ? j + kl + 1 : n;
    const T *const col = AB + ku + j * ldab - j; // col[i] == A(i, j)
    for (std::size_t i = i_lo + threadIdx.x; i < i_hi; i += kBlock) {
      switch (which) {
      case MatrixNorm::max_abs:
        acc = max_nan(acc, elem_ops<T>::modulus(col[i]));
        break;
      case MatrixNorm::frobenius:
        acc += entry_sq<T, R>(col[i]);
        break;
      default: // MatrixNorm::one
        acc += elem_ops<T>::modulus(col[i]);
        break;
      }
    }
  }

  const R total = (which == MatrixNorm::max_abs) ? block_reduce<kBlock>(acc, MaxNanOp{})
                                                 : block_reduce<kBlock>(acc, AddOp{});
  if (threadIdx.x == 0) {
    scratch[j] = total;
  }
}

/// @brief [kernel] One block folds the n partials into *d_result
template<typename R>
__global__ void langb_fold_kernel(const MatrixNorm which, const std::size_t n,
                                  const R *const scratch, R *const d_result) {
  const bool sum = which == MatrixNorm::frobenius;
  R acc = R{0};
  for (std::size_t j = threadIdx.x; j < n; j += kBlock) {
    acc = sum ? acc + scratch[j] : max_nan(acc, scratch[j]);
  }
  const R total = sum ? block_reduce<kBlock>(acc, AddOp{}) : block_reduce<kBlock>(acc, MaxNanOp{});
  if (threadIdx.x == 0) {
    if (!sum) {
      *d_result = total;
    } else if constexpr (std::is_same_v<R, float>) {
      *d_result = sqrtf(total);
    } else {
      *d_result = sqrt(total);
    }
  }
}

} // namespace

template<typename T, typename R>
void langb(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
           const std::size_t kl, const std::size_t ku, const T *const d_AB, const std::size_t ldab,
           R *const d_result, R *const d_scratch) {
  langb_partials_kernel<T, R><<<static_cast<unsigned int>(n), kBlock, 0, stream>>>(
      which, n, kl, ku, d_AB, ldab, d_scratch);
  langb_fold_kernel<R><<<1, kBlock, 0, stream>>>(which, n, d_scratch, d_result);
}

// One per supported type, matching langb_bridge.h and interface.cppm's `extern
// template` list.
template void langb<float, float>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                  std::size_t, const float *, std::size_t, float *, float *);
template void langb<double, double>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                    std::size_t, const double *, std::size_t, double *, double *);
template void langb<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                 std::size_t, std::size_t,
                                                 const wwr::wwrFloatComplex *, std::size_t, float *,
                                                 float *);
template void langb<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   std::size_t, std::size_t,
                                                   const wwr::wwrDoubleComplex *, std::size_t,
                                                   double *, double *);

} // namespace calaman::device
