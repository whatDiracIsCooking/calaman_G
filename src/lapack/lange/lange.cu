// lange.cu
//
// The device half of calaman.lange: the ?lange matrix norm of a column-major
// matrix, computed as a two-stage reduction so it owns almost no kernel of its
// own. Stage one reduces A to a per-column (or, for the infinity norm, per-row)
// intermediate; stage two folds that intermediate to a single scalar. Both
// stages defer to calaman.reduce_columns (reduce_columns/reduce_columns.cuh, a
// device-code header): stage two treats the length-k intermediate as a one-column
// matrix (rows = k, ncols = 1), so a column reduction collapses it to one value.
//
//   max_abs ('M'): per-column max|.|,  then max over columns
//   one     ('1'): per-column sum|.|,  then max over columns
//   inf     ('I'): per-row   sum|.|,   then max over rows
//   frobenius('F'): per-column sum(.^2), then sum over columns, then sqrt
//
// Only the per-ROW abs-sum is a hand-written kernel here: reduce_columns folds
// down a column (the contiguous direction), and no existing module reduces along
// rows of a column-major matrix. The frobenius sqrt is one element, taken through
// wwr.extension.parallel_for as columnwise_ell2 does. The folds propagate a NaN
// from either operand, matching DLANGE's `VALUE.LT.temp .OR. disnan(temp)`.
//
// Complex A reduces to a real norm: the pre-transforms map T -> R (R the real
// component type) through common/elem_ops.cuh -- |z| for the sum/max norms, and
// re^2 + im^2 for 'F' (the two components CLANGE's CLASSQ sums), so every fold,
// the scratch and the result are R. For real T, R == T.
//
// Shared unchanged between both backends -- under CUDA the .cu extension is all
// CMake needs, under HIP this directory's CMakeLists.txt forces LANGUAGE CXX back
// on so clang compiles it with -x hip.
#include "lange_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>
#include "common/elem_ops.cuh"
#include "reduce_columns/reduce_columns.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

/// @brief Per-element pre-transform T -> R: |x|, the magnitude the sum/max norms fold
template<typename T, typename R>
struct AbsFunctor {
  __device__ R operator()(const T x) const { return elem_ops<T>::modulus(x); }
};

/// @brief Per-element pre-transform T -> R: re^2 + im^2, what the Frobenius norm sums
template<typename T, typename R>
struct SquareFunctor {
  __device__ R operator()(const T x) const {
    const R re = elem_ops<T>::real_part(x);
    const R im = elem_ops<T>::imag_part(x);
    return re * re + im * im;
  }
};

/// @brief Associative binary fold a + b, the sum the 1/inf/Frobenius norms accumulate
template<typename T>
struct PlusFunctor {
  __device__ T operator()(const T a, const T b) const { return a + b; }
};

/// @brief NaN-propagating max fold, matching DLANGE's disnan-guarded comparison
///
/// A plain `a < b ? b : a` drops a NaN second operand (every comparison with NaN
/// is false); returning the NaN when EITHER operand is one reproduces DLANGE's
/// `VALUE.LT.temp .OR. disnan(temp)`. `x != x` is the identityless isnan.
template<typename T>
struct MaxFunctor {
  __device__ T operator()(const T a, const T b) const {
    if (a != a) {
      return a;
    }
    if (b != b) {
      return b;
    }
    return a < b ? b : a;
  }
};

/// @brief In-place sqrt over the single Frobenius accumulator, the norm's last pass
///
/// A device_functor for wwr.extension.parallel_for launched over one element, as
/// columnwise_ell2's does. `sqrtf`/`sqrt` are the CUDA/HIP device builtins,
/// selected per type so float stays in single precision.
template<typename T>
struct SqrtFunctor {
  T *const d_result_;
  __device__ void operator()(const std::size_t i) const {
    if constexpr (std::is_same_v<T, float>) {
      d_result_[i] = sqrtf(d_result_[i]);
    } else {
      d_result_[i] = sqrt(d_result_[i]);
    }
  }
};

/// @brief [kernel] Per-row abs-sum: `d_rowsum[i] = sum_j |A(i, j)|`
///
/// One thread per row, grid-stride over rows. Each thread walks its row across
/// the @p n columns -- strided by @p lda in the column-major storage -- so the
/// access is uncoalesced, the price of reducing along the non-contiguous
/// direction reduce_columns cannot. The seed is 0 and + propagates a NaN, so a
/// non-finite row carries through to the max fold.
template<typename T, typename R>
__global__ void rowwise_abssum_kernel(const T *const d_A, R *const d_rowsum, const std::size_t m,
                                      const std::size_t n, const std::size_t lda) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < m;
       i += stride) {
    R acc = R{0};
    for (std::size_t j = 0; j < n; ++j) {
      acc += elem_ops<T>::modulus(d_A[i + j * lda]);
    }
    d_rowsum[i] = acc;
  }
}

} // namespace

template<typename T, typename R>
void lange(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t m,
           const std::size_t n, const T *const d_A, const std::size_t lda, R *const d_result,
           R *const d_scratch) {
  switch (which) {
  case MatrixNorm::max_abs:
    // Stage 1: per-column max|.| into d_scratch[0..n). Stage 2: max over the n.
    reduce_columns_transform<T, R>(stream, d_A, d_scratch, m, n, lda, AbsFunctor<T, R>{},
                                   MaxFunctor<R>{});
    reduce_columns<R>(stream, d_scratch, d_result, n, 1, n, MaxFunctor<R>{});
    break;
  case MatrixNorm::one:
    // Stage 1: per-column sum|.| (the column sums). Stage 2: max over them.
    reduce_columns_transform<T, R>(stream, d_A, d_scratch, m, n, lda, AbsFunctor<T, R>{},
                                   PlusFunctor<R>{});
    reduce_columns<R>(stream, d_scratch, d_result, n, 1, n, MaxFunctor<R>{});
    break;
  case MatrixNorm::inf: {
    // Stage 1: per-row sum|.| (the m row sums). Stage 2: max over them.
    constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
    const unsigned int blocks = static_cast<unsigned int>((m + kBlockSize - 1) / kBlockSize);
    rowwise_abssum_kernel<T, R><<<blocks, kBlockSize, 0, stream>>>(d_A, d_scratch, m, n, lda);
    reduce_columns<R>(stream, d_scratch, d_result, m, 1, m, MaxFunctor<R>{});
    break;
  }
  case MatrixNorm::frobenius:
    // Stage 1: per-column sum of squares. Stage 2: sum them. Stage 3: sqrt the one.
    reduce_columns_transform<T, R>(stream, d_A, d_scratch, m, n, lda, SquareFunctor<T, R>{},
                                   PlusFunctor<R>{});
    reduce_columns<R>(stream, d_scratch, d_result, n, 1, n, PlusFunctor<R>{});
    wwr::extension::parallel_for<std::size_t>(stream, 1, SqrtFunctor<R>{d_result});
    break;
  }
}

// One per supported type, matching lange_bridge.h and interface.cppm's `extern
// template` list -- float, double and their complex counterparts (real R).
template void lange<float, float>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                  const float *, std::size_t, float *, float *);
template void lange<double, double>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                    const double *, std::size_t, double *, double *);
template void lange<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                 std::size_t, const wwr::wwrFloatComplex *,
                                                 std::size_t, float *, float *);
template void lange<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   std::size_t, const wwr::wwrDoubleComplex *,
                                                   std::size_t, double *, double *);

} // namespace calaman::device
