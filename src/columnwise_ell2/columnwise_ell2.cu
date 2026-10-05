// columnwise_ell2.cu
//
// The device half of calaman.columnwise_ell2: the per-column L2 (Euclidean)
// norm out[j] = sqrt(sum_i A(i, j)^2). It owns no reduction kernel of its own --
// the sum of squares IS a column reduction, so pass one defers to
// calaman.reduce_columns (reduce_columns/reduce_columns.cuh, a device-code
// header) with (.)^2 as the pre-transform and + as the fold. Pass two then takes
// the elementwise sqrt of the cols results in place, via
// wwr.extension.parallel_for (extension/parallel_for/parallel_for.cuh) -- the
// same hand-written index-per-thread launch gebal seeds its scale[] with.
//
// The reference this was ported from factored that second pass into its own
// `sqrt_each_impl` static library; here it is a two-line functor launched
// inline, as gebal's FillOnesFunctor is -- there is no separate sqrt_each module
// in calaman. Both passes run on the same stream and are asynchronous; the sqrt
// pass reads what the reduce pass wrote, and in-stream ordering serialises them.
//
// Shared unchanged between both backends -- under CUDA the .cu extension is all
// CMake needs, under HIP this directory's CMakeLists.txt forces LANGUAGE CXX
// back on so clang compiles it with -x hip.
//
// Float and double only, matching the rest of calaman. The complex L2 norm is
// real-valued (a sum of squared moduli under a sqrt, a T -> real reduction), a
// deliberate later extension -- see interface.cppm.
#include "columnwise_ell2_bridge.h"

#include "extension/parallel_for/parallel_for.cuh"
#include "reduce_columns/reduce_columns.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-element pre-transform x^2, the square summed by the L2 norm
///
/// A trivially-copyable class with a `__device__` call, as reduce_columns'
/// device_functor requires. `x * x` needs no <cmath> in the device pass.
template<typename T>
struct SquareFunctor {
  __device__ T operator()(const T x) const { return x * x; }
};

/// @brief Associative binary fold a + b -- the sum the L2 norm accumulates
template<typename T>
struct PlusFunctor {
  __device__ T operator()(const T a, const T b) const { return a + b; }
};

/// @brief In-place elementwise sqrt over the per-column sums, the L2 norm's
///        second pass
///
/// A device_functor for wwr.extension.parallel_for: index-per-thread over the
/// cols sums, overwriting each with its root. `sqrtf`/`sqrt` are the CUDA/HIP
/// device builtins (as gebal's `fabsf`/`fabs` are), selected per type so float
/// stays in single precision.
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

} // namespace

template<typename T>
void columnwise_ell2(const wwr::wwrStream_t stream, const std::size_t rows, const std::size_t cols,
                     const T *const d_A, const std::size_t lda, T *const d_result) {
  if (rows == 0 || cols == 0) {
    return;
  }
  // Pass 1: d_result[j] = sum_i A(i, j)^2.
  reduce_columns_transform<T, T>(stream, d_A, d_result, rows, cols, lda, SquareFunctor<T>{},
                                 PlusFunctor<T>{});
  // Pass 2: d_result[j] = sqrt(d_result[j]), in place over the cols sums.
  wwr::extension::parallel_for<std::size_t>(stream, cols, SqrtFunctor<T>{d_result});
}

// One per supported type, matching columnwise_ell2_bridge.h's declarations and
// the module's use sites -- float and double.
template void columnwise_ell2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                     std::size_t, float *);
template void columnwise_ell2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                      std::size_t, double *);

} // namespace calaman::device
