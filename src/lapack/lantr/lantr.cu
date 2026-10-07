// lantr.cu
//
// The device half of calaman.lantr, calaman.lantp AND calaman.lantb: the
// ?lantr/?lantp/?lantb norm of an upper or lower triangular (trapezoidal, for
// full storage) matrix, in two launches -- lanhs.cu's shape. Stage one is one
// block per line, folding its share into scratch[line]:
//
//   max_abs  ('M'): max |A(i,j)| over the triangle's part of column j
//   one      ('1'): sum |A(i,j)| over the same entries (the column sum)
//   inf      ('I'): sum |A(i,j)| over the triangle's part of ROW i
//   frobenius('F'): sum |A(i,j)|^2 over column j's entries
//
// so there are m lines for 'I' and n otherwise. A line visits only the in-matrix
// entries of the triangle (and, for band storage, of the band), so every other
// slot -- the opposite triangle, the unused band corner, any ld padding -- may
// hold anything. With Diag::U each diagonal entry counts as 1 and is NOT read.
// The storage layout is a template parameter: TriStorage in lantr_bridge.h, the
// packed map from calaman::tri_index, the band index inline (calaman.langb's
// precedent). Stage two is one block folding scratch (max or sum, then sqrt
// for 'F'). The max folds start from 0 and propagate a NaN, as DLANTR's DISNAN
// guard does; 'F' is a plain sum of squares, not ?lassq (calaman.lange).
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lantr_bridge.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"
#include "lapack/tri_index/tri_index.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

/// @brief Offset of the stored A(i,j), (i,j) in the triangle (and band)
template<TriStorage S, bool Upper>
__device__ std::size_t stored_index(const std::size_t n, const std::size_t k, const std::size_t ld,
                                    const std::size_t i, const std::size_t j) {
  if constexpr (S == TriStorage::full) {
    return i + j * ld;
  } else if constexpr (S == TriStorage::packed) {
    return packed_index<Upper>(n, i, j);
  } else {
    return (Upper ? k + i - j : i - j) + j * ld;
  }
}

/// @brief [kernel] Block l writes column l's (row l's, for inf) partial norm
template<typename T, typename R, TriStorage S, bool Upper>
__global__ void lantr_partials_kernel(const MatrixNorm which, const bool unit, const std::size_t m,
                                      const std::size_t n, const std::size_t k, const T *const A,
                                      const std::size_t ld, R *const scratch) {
  constexpr bool kBand = S == TriStorage::band;
  const std::size_t line = blockIdx.x;
  const bool by_row = which == MatrixNorm::inf;
  // The line's in-triangle index range [lo, hi): row indices of column `line`,
  // or column indices of row `line` for inf. Upper holds i <= j, lower i >= j.
  std::size_t lo;
  std::size_t hi;
  const std::size_t extent = by_row ? n : m;
  if (by_row != Upper) {
    // Column `line` of an upper / row `line` of a lower: indices up to the diagonal.
    lo = (kBand && line > k) ? line - k : 0;
    hi = line + 1 < extent ? line + 1 : extent;
  } else {
    // Column `line` of a lower / row `line` of an upper: from the diagonal on.
    lo = line;
    hi = (kBand && line + k + 1 < extent) ? line + k + 1 : extent;
  }

  R acc = R{0};
  for (std::size_t t = lo + threadIdx.x; t < hi; t += kBlock) {
    const std::size_t i = by_row ? line : t;
    const std::size_t j = by_row ? t : line;
    R a;  // |A(i,j)|, or |A(i,j)|^2 for frobenius
    if (unit && i == j) {
      a = R{1};
    } else {
      const T x = A[stored_index<S, Upper>(n, k, ld, i, j)];
      if (which == MatrixNorm::frobenius) {
        const R re = elem_ops<T>::real_part(x);
        const R im = elem_ops<T>::imag_part(x);
        a = re * re + im * im;
      } else {
        a = elem_ops<T>::modulus(x);
      }
    }
    acc = (which == MatrixNorm::max_abs) ? max_nan(acc, a) : acc + a;
  }

  const R total = (which == MatrixNorm::max_abs) ? block_reduce<kBlock>(acc, MaxNanOp{})
                                                 : block_reduce<kBlock>(acc, AddOp{});
  if (threadIdx.x == 0) {
    scratch[line] = total;
  }
}

/// @brief [kernel] One block folds the @p lines partials into *d_result
template<typename R>
__global__ void lantr_fold_kernel(const MatrixNorm which, const std::size_t lines,
                                  const R *const scratch, R *const d_result) {
  const bool sum = which == MatrixNorm::frobenius;
  R acc = R{0};
  for (std::size_t l = threadIdx.x; l < lines; l += kBlock) {
    acc = sum ? acc + scratch[l] : max_nan(acc, scratch[l]);
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

template<typename T, typename R, TriStorage S>
void launch(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
            const bool unit, const std::size_t m, const std::size_t n, const std::size_t k,
            const T *const d_A, const std::size_t ld, R *const d_scratch, const unsigned int lines) {
  if (uplo == Uplo::U) {
    lantr_partials_kernel<T, R, S, true>
        <<<lines, kBlock, 0, stream>>>(which, unit, m, n, k, d_A, ld, d_scratch);
  } else {
    lantr_partials_kernel<T, R, S, false>
        <<<lines, kBlock, 0, stream>>>(which, unit, m, n, k, d_A, ld, d_scratch);
  }
}

} // namespace

template<typename T, typename R>
void lantr(const wwr::wwrStream_t stream, const MatrixNorm which, const Uplo uplo,
           const Diag diag, const TriStorage storage, const std::size_t m, const std::size_t n,
           const std::size_t k, const T *const d_A, const std::size_t ld, R *const d_result,
           R *const d_scratch) {
  const std::size_t lines = which == MatrixNorm::inf ? m : n;
  const auto blocks = static_cast<unsigned int>(lines);
  const bool unit = diag == Diag::U;
  switch (storage) {
  case TriStorage::full:
    launch<T, R, TriStorage::full>(stream, which, uplo, unit, m, n, k, d_A, ld, d_scratch, blocks);
    break;
  case TriStorage::packed:
    launch<T, R, TriStorage::packed>(stream, which, uplo, unit, m, n, k, d_A, ld, d_scratch,
                                     blocks);
    break;
  case TriStorage::band:
    launch<T, R, TriStorage::band>(stream, which, uplo, unit, m, n, k, d_A, ld, d_scratch, blocks);
    break;
  }
  lantr_fold_kernel<R><<<1, kBlock, 0, stream>>>(which, lines, d_scratch, d_result);
}

// One per supported type, matching lantr_bridge.h and the `extern template`
// lists of calaman.lantr, calaman.lantp and calaman.lantb.
template void lantr<float, float>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, TriStorage,
                                  std::size_t, std::size_t, std::size_t, const float *,
                                  std::size_t, float *, float *);
template void lantr<double, double>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag, TriStorage,
                                    std::size_t, std::size_t, std::size_t, const double *,
                                    std::size_t, double *, double *);
template void lantr<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                                 TriStorage, std::size_t, std::size_t, std::size_t,
                                                 const wwr::wwrFloatComplex *, std::size_t,
                                                 float *, float *);
template void lantr<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, Uplo, Diag,
                                                   TriStorage, std::size_t, std::size_t,
                                                   std::size_t, const wwr::wwrDoubleComplex *,
                                                   std::size_t, double *, double *);

} // namespace calaman::device
