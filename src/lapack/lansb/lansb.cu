// lansb.cu
//
// The device half of calaman.lansb AND calaman.lanhb: the ?lansb/?lanhb norm of
// an n-by-n symmetric (or Hermitian) band matrix with k super- (Uplo::U) or
// sub-diagonals (Uplo::L) in LAPACK band storage -- A(i,j) at AB(k+i-j, j) for
// the upper triangle, AB(i-j, j) for the lower -- in two launches, lansp.cu's
// shape. Stage one is one block per column j of the full matrix, folding the
// band of column j (rows max(0, j-k) .. min(n-1, j+k)) into scratch[j]:
//
//   max_abs  ('M'): max |A(i,j)| over the band of the column
//   one/inf  ('1'/'I', equal by symmetry): sum |A(i,j)| over it
//   frobenius('F'): sum |A(i,j)|^2 over it
//
// A(i,j) outside the stored triangle is read as A(j,i); only |.| is taken, so
// symmetric and Hermitian mirrors read alike. Only in-matrix band entries of
// the stored triangle are addressed, so the unused corner of AB and the ldab
// padding may hold anything. Stage two is one block folding scratch (max or
// sum, then sqrt for 'F'). The max folds start from 0 and propagate a NaN, as
// DLANSB's DISNAN guard does; 'F' is a plain sum of squares, not ?lassq (the
// calaman.lange precedent). The `hermitian` flag is the whole ?lanhb/?lansb
// difference: a Hermitian diagonal entry contributes only its real part.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lansb_bridge.h"

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
template<typename T, typename R, bool Upper>
__global__ void lansb_columns_kernel(const MatrixNorm which, const bool hermitian,
                                     const std::size_t n, const std::size_t k, const T *const AB,
                                     const std::size_t ldab, R *const scratch) {
  const std::size_t j = blockIdx.x;
  const std::size_t i_lo = j > k ? j - k : 0;
  const std::size_t i_hi = j + k + 1 < n ? j + k + 1 : n;
  R acc = R{0};
  for (std::size_t i = i_lo + threadIdx.x; i < i_hi; i += kBlock) {
    // The stored entry (r, c) holding A(i,j): itself, or its mirror A(j,i).
    const std::size_t r = (Upper == (i <= j)) ? i : j;
    const std::size_t c = (Upper == (i <= j)) ? j : i;
    const T x = AB[(Upper ? k + r - c : r - c) + c * ldab];
    const bool real_only = hermitian && i == j;
    switch (which) {
    case MatrixNorm::max_abs:
      acc = max_nan(acc, entry_abs<T, R>(x, real_only));
      break;
    case MatrixNorm::one:
    case MatrixNorm::inf:
      acc += entry_abs<T, R>(x, real_only);
      break;
    case MatrixNorm::frobenius:
      acc += entry_sq<T, R>(x, real_only);
      break;
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
__global__ void lansb_fold_kernel(const MatrixNorm which, const std::size_t n,
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
void lansb(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
           const bool hermitian, const std::size_t n, const std::size_t k, const T *const d_AB,
           const std::size_t ldab, R *const d_result, R *const d_scratch) {
  const auto blocks = static_cast<unsigned int>(n);
  if (uplo == Uplo::U) {
    lansb_columns_kernel<T, R, true>
        <<<blocks, kBlock, 0, stream>>>(which, hermitian, n, k, d_AB, ldab, d_scratch);
  } else {
    lansb_columns_kernel<T, R, false>
        <<<blocks, kBlock, 0, stream>>>(which, hermitian, n, k, d_AB, ldab, d_scratch);
  }
  lansb_fold_kernel<R><<<1, kBlock, 0, stream>>>(which, n, d_scratch, d_result);
}

// One per supported type, matching lansb_bridge.h and the `extern template`
// lists of calaman.lansb (all four) and calaman.lanhb (the complex two).
template void lansb<float, float>(wwr::wwrStream_t, MatrixNorm, Uplo, bool, std::size_t,
                                  std::size_t, const float *, std::size_t, float *, float *);
template void lansb<double, double>(wwr::wwrStream_t, MatrixNorm, Uplo, bool, std::size_t,
                                    std::size_t, const double *, std::size_t, double *, double *);
template void lansb<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, Uplo, bool,
                                                 std::size_t, std::size_t,
                                                 const wwr::wwrFloatComplex *, std::size_t, float *,
                                                 float *);
template void lansb<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, Uplo, bool,
                                                   std::size_t, std::size_t,
                                                   const wwr::wwrDoubleComplex *, std::size_t,
                                                   double *, double *);

} // namespace calaman::device
