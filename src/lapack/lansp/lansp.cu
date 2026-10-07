// lansp.cu
//
// The device half of calaman.lansp AND calaman.lanhp: the ?lansp/?lanhp norm of
// a symmetric (or Hermitian) matrix held in packed storage, in two launches --
// lansf.cu's shape. Stage one is one block per column j of the full n-by-n
// matrix, folding column j's share into scratch[j]:
//
//   max_abs  ('M'): max |A(i,j)| over the column
//   one/inf  ('1'/'I', equal by symmetry): sum |A(i,j)| over the column
//   frobenius('F'): sum |A(i,j)|^2 over the column
//
// A(i,j) outside the stored triangle is read as A(j,i), and every stored
// element's AP slot comes from calaman::tri_index's packed_index; only |.| is
// taken, so symmetric and Hermitian mirrors read alike. Stage two is one block
// folding scratch (max or sum, then sqrt for 'F'). The max folds start from 0
// and propagate a NaN, as DLANSP's DISNAN guard does; 'F' is a plain sum of
// squares, not ?lassq (the calaman.lange precedent). The `hermitian` flag is
// the whole ?lanhp/?lansp difference: a Hermitian diagonal entry contributes
// only its real part.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lansp_bridge.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"
#include "lapack/tri_index/tri_index.cuh"

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
__global__ void lansp_columns_kernel(const MatrixNorm which, const bool hermitian,
                                     const std::size_t n, const T *const ap, R *const scratch) {
  const std::size_t j = blockIdx.x;
  R acc = R{0};
  for (std::size_t i = threadIdx.x; i < n; i += kBlock) {
    // A(i,j) of the unstored triangle is read as its mirror A(j,i).
    const bool in = in_triangle<Upper>(i, j);
    const T x = ap[packed_index<Upper>(n, in ? i : j, in ? j : i)];
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
__global__ void lansp_fold_kernel(const MatrixNorm which, const std::size_t n,
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
void lansp(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
           const bool hermitian, const std::size_t n, const T *const d_ap, R *const d_result,
           R *const d_scratch) {
  const auto blocks = static_cast<unsigned int>(n);
  if (uplo == Uplo::U) {
    lansp_columns_kernel<T, R, true>
        <<<blocks, kBlock, 0, stream>>>(which, hermitian, n, d_ap, d_scratch);
  } else {
    lansp_columns_kernel<T, R, false>
        <<<blocks, kBlock, 0, stream>>>(which, hermitian, n, d_ap, d_scratch);
  }
  lansp_fold_kernel<R><<<1, kBlock, 0, stream>>>(which, n, d_scratch, d_result);
}

// One per supported type, matching lansp_bridge.h and the `extern template`
// lists of calaman.lansp (all four) and calaman.lanhp (the complex two).
template void lansp<float, float>(wwr::wwrStream_t, MatrixNorm, Uplo, bool, std::size_t,
                                  const float *, float *, float *);
template void lansp<double, double>(wwr::wwrStream_t, MatrixNorm, Uplo, bool, std::size_t,
                                    const double *, double *, double *);
template void lansp<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, Uplo, bool,
                                                 std::size_t, const wwr::wwrFloatComplex *, float *,
                                                 float *);
template void lansp<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, Uplo, bool,
                                                   std::size_t, const wwr::wwrDoubleComplex *,
                                                   double *, double *);

} // namespace calaman::device
