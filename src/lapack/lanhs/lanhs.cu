// lanhs.cu
//
// The device half of calaman.lanhs: the ?lanhs norm of an n-by-n upper
// Hessenberg matrix, in two launches. Stage one is one block per index j,
// folding its share into scratch[j] through common/block_reduce.cuh:
//
//   max_abs  ('M'): max |A(i,j)| over rows [0, min(n, j+2)) of column j
//   one      ('1'): sum |A(i,j)| over the same rows (the column sum)
//   inf      ('I'): sum |A(j,k)| over columns [max(0, j-1), n) of ROW j
//   frobenius('F'): sum |A(i,j)|^2 over the column's rows
//
// Stage two is one block folding scratch (max or sum, then sqrt for 'F').
// Entries below the subdiagonal are never addressed, so they may hold
// anything. The max folds start from 0 and propagate a NaN, as DLANHS's DISNAN
// guard does; 'F' is a plain sum of squares, not ?lassq (the calaman.lange
// precedent). A complex T has a real norm R.
//
// Not calaman.reduce_columns: its pre-transform sees a value, not a row index,
// so it cannot mask the unreferenced triangle, and the infinity norm runs along
// rows. The shape is calaman.lansy's.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lanhs_bridge.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

/// @brief [kernel] Block j writes column j's (row j's, for inf) partial norm
template<typename T, typename R>
__global__ void lanhs_partials_kernel(const MatrixNorm which, const std::size_t n,
                                      const T *const A, const std::size_t lda, R *const scratch) {
  const std::size_t j = blockIdx.x;
  R acc = R{0};
  if (which == MatrixNorm::inf) {
    // Row j of an upper Hessenberg matrix: columns [max(0, j-1), n).
    const std::size_t k_lo = j == 0 ? 0 : j - 1;
    for (std::size_t k = k_lo + threadIdx.x; k < n; k += kBlock) {
      acc += elem_ops<T>::modulus(A[j + k * lda]);
    }
  } else {
    // Column j: rows [0, min(n, j+2)).
    const std::size_t i_hi = (j + 2 < n) ? j + 2 : n;
    const T *const col = A + j * lda;
    for (std::size_t i = threadIdx.x; i < i_hi; i += kBlock) {
      switch (which) {
      case MatrixNorm::max_abs:
        acc = max_nan(acc, elem_ops<T>::modulus(col[i]));
        break;
      case MatrixNorm::frobenius: {
        const R re = elem_ops<T>::real_part(col[i]);
        const R im = elem_ops<T>::imag_part(col[i]);
        acc += re * re + im * im;
        break;
      }
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
__global__ void lanhs_fold_kernel(const MatrixNorm which, const std::size_t n,
                                  const R *const scratch, R *const d_result) {
  const bool sum = which == MatrixNorm::frobenius;
  R acc = R{0};
  for (std::size_t j = threadIdx.x; j < n; j += kBlock) {
    acc = sum ? acc + scratch[j] : max_nan(acc, scratch[j]);
  }
  const R total =
      sum ? block_reduce<kBlock>(acc, AddOp{}) : block_reduce<kBlock>(acc, MaxNanOp{});
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
void lanhs(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
           const T *const d_A, const std::size_t lda, R *const d_result, R *const d_scratch) {
  lanhs_partials_kernel<T, R>
      <<<static_cast<unsigned int>(n), kBlock, 0, stream>>>(which, n, d_A, lda, d_scratch);
  lanhs_fold_kernel<R><<<1, kBlock, 0, stream>>>(which, n, d_scratch, d_result);
}

// One per supported type, matching lanhs_bridge.h and interface.cppm's `extern
// template` list.
template void lanhs<float, float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                                  std::size_t, float *, float *);
template void lanhs<double, double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                                    std::size_t, double *, double *);
template void lanhs<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                 const wwr::wwrFloatComplex *, std::size_t,
                                                 float *, float *);
template void lanhs<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   const wwr::wwrDoubleComplex *, std::size_t,
                                                   double *, double *);

} // namespace calaman::device
