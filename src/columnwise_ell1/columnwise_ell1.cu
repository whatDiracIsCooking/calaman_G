// columnwise_ell1.cu
//
// The device half of calaman.columnwise_ell1: the per-column L1 norm
// out[j] = sum_i |A(i, j)|. There is no per-element device work to launch
// directly -- the whole computation IS a column reduction, so this defers to
// calaman.reduce_columns (reduce_columns/reduce_columns.cuh, a device-code
// header) with |.| as the pre-transform and + as the fold. The transform path
// is the correct one here: it applies |.| to EVERY element including the fold's
// seed, so a single-row column reports |a| rather than a.
//
// Shared unchanged between both backends -- under CUDA the .cu extension is all
// CMake needs, under HIP this directory's CMakeLists.txt forces LANGUAGE CXX
// back on so clang compiles it with -x hip.
//
// Float and double only, matching the rest of calaman. The complex L1 norm is
// real-valued (sum of moduli, a T -> real reduction), a deliberate later
// extension -- see interface.cppm.
#include "columnwise_ell1_bridge.h"

#include "reduce_columns/reduce_columns.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Per-element pre-transform |x|, the magnitude summed by the L1 norm
///
/// A trivially-copyable class with a `__device__` call, as reduce_columns'
/// unary_transform_functor requires. `x < 0 ? -x : x` matches the host spelling
/// calaman.diff_norm uses and needs no <cmath> in the device pass.
template<typename T>
struct AbsFunctor {
  __device__ T operator()(const T x) const { return x < T{0} ? -x : x; }
};

/// @brief Associative binary fold a + b -- the sum the L1 norm accumulates
template<typename T>
struct PlusFunctor {
  __device__ T operator()(const T a, const T b) const { return a + b; }
};

} // namespace

template<typename T>
void columnwise_ell1(const wwr::wwrStream_t stream, const std::size_t rows, const std::size_t cols,
                     const T *const d_A, const std::size_t lda, T *const d_result) {
  if (rows == 0 || cols == 0) {
    return;
  }
  reduce_columns_transform<T, T>(stream, d_A, d_result, rows, cols, lda, AbsFunctor<T>{},
                                 PlusFunctor<T>{});
}

// One per supported type, matching columnwise_ell1_bridge.h's declarations and
// the module's use sites -- float and double.
template void columnwise_ell1<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                     std::size_t, float *);
template void columnwise_ell1<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                      std::size_t, double *);

} // namespace calaman::device
