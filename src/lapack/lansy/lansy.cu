// lansy.cu
//
// The device half of calaman.lansy AND calaman.lanhe: the ?lansy/?lanhe norm of
// a symmetric (or Hermitian) matrix from the one triangle `uplo` names, in two
// launches. Stage one is one block per column j, folding its share into
// scratch[j] through common/block_reduce.cuh:
//
//   max_abs  ('M'): max |A(i,j)| over the stored part of column j
//   one/inf  ('1'/'I', equal by symmetry): the full row sum of row j -- the
//                   stored part of column j plus the stored part of row j
//   frobenius('F'): |A(j,j)|^2 + 2 * sum of the stored off-diagonal |.|^2
//
// Stage two is one block folding scratch (max or sum, then sqrt for 'F'). The
// other triangle is never addressed, so it may hold anything. The max folds
// start from 0 and propagate a NaN, as DLANSY's DISNAN guard does; 'F' is a
// plain sum of squares, not ?lassq (the calaman.lange precedent). A complex T
// has a real norm R. The `hermitian` flag is the whole ?lanhe/?lansy difference:
// a Hermitian diagonal entry contributes only its real part.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lansy_bridge.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

/// @brief |x|, or |Re x| alone when @p real_only (a Hermitian diagonal entry)
template<typename T, typename R>
__device__ R entry_abs(const T x, const bool real_only) {
  return real_only ? elem_ops<R>::modulus(elem_ops<T>::real_part(x)) : elem_ops<T>::modulus(x);
}

/// @brief |x|^2 as re^2 + im^2, or re^2 alone when @p real_only
template<typename T, typename R>
__device__ R entry_sq(const T x, const bool real_only) {
  const R re = elem_ops<T>::real_part(x);
  const R im = real_only ? R{0} : elem_ops<T>::imag_part(x);
  return re * re + im * im;
}

/// @brief [kernel] Block j writes column j's partial of the @p which norm
template<typename T, typename R>
__global__ void lansy_columns_kernel(const MatrixNorm which, const Uplo uplo, const bool hermitian,
                                     const std::size_t n, const T *const A, const std::size_t lda,
                                     R *const scratch) {
  const std::size_t j = blockIdx.x;
  const bool upper = uplo == Uplo::U;
  // The stored part of column j: rows [0, j] (upper) or [j, n) (lower).
  const std::size_t c_lo = upper ? 0 : j;
  const std::size_t c_hi = upper ? j + 1 : n;
  const T *const col = A + j * lda;

  R acc = R{0};
  for (std::size_t i = c_lo + threadIdx.x; i < c_hi; i += kBlock) {
    const bool real_only = hermitian && i == j;
    switch (which) {
    case MatrixNorm::max_abs:
      acc = max_nan(acc, entry_abs<T, R>(col[i], real_only));
      break;
    case MatrixNorm::one:
    case MatrixNorm::inf:
      acc += entry_abs<T, R>(col[i], real_only);
      break;
    case MatrixNorm::frobenius:
      acc += (i == j ? R{1} : R{2}) * entry_sq<T, R>(col[i], real_only);
      break;
    }
  }
  if (which == MatrixNorm::one || which == MatrixNorm::inf) {
    // The stored part of row j, off the diagonal: columns (j, n) (upper) or
    // [0, j) (lower) -- A(k,j) of the unstored triangle, read as A(j,k).
    const std::size_t r_lo = upper ? j + 1 : 0;
    const std::size_t r_hi = upper ? n : j;
    for (std::size_t k = r_lo + threadIdx.x; k < r_hi; k += kBlock) {
      acc += elem_ops<T>::modulus(A[j + k * lda]);
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
__global__ void lansy_fold_kernel(const MatrixNorm which, const std::size_t n,
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
void lansy(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
           const bool hermitian, const std::size_t n, const T *const d_A, const std::size_t lda,
           R *const d_result, R *const d_scratch) {
  lansy_columns_kernel<T, R><<<static_cast<unsigned int>(n), kBlock, 0, stream>>>(
      which, uplo, hermitian, n, d_A, lda, d_scratch);
  lansy_fold_kernel<R><<<1, kBlock, 0, stream>>>(which, n, d_scratch, d_result);
}

// One per supported type, matching lansy_bridge.h and the `extern template`
// lists of calaman.lansy (all four) and calaman.lanhe (the complex two).
template void lansy<float, float>(wwr::wwrStream_t, MatrixNorm, Uplo, bool, std::size_t,
                                  const float *, std::size_t, float *, float *);
template void lansy<double, double>(wwr::wwrStream_t, MatrixNorm, Uplo, bool, std::size_t,
                                    const double *, std::size_t, double *, double *);
template void lansy<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, Uplo, bool,
                                                 std::size_t, const wwr::wwrFloatComplex *,
                                                 std::size_t, float *, float *);
template void lansy<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, Uplo, bool,
                                                   std::size_t, const wwr::wwrDoubleComplex *,
                                                   std::size_t, double *, double *);

} // namespace calaman::device
