// ilalc.cu
//
// The device half of calaman.ilalc: the count of leading columns of a
// column-major matrix up to its last non-zero column, as a two-stage
// max-reduction. Stage one is this file's kernel, one block per column,
// writing j + 1 when column j holds a non-zero and 0 otherwise; stage two folds
// those n values with calaman.reduce_columns' max. Only rows [0, m) of each
// column are read, never the lda - m padding.
//
// "Non-zero" is the reference's `A(i,j).NE.ZERO`: a NaN counts, -0 does not,
// and a complex entry counts when either component does.
#include "ilalc_bridge.h"

#include "common/elem_ops.cuh"
#include "reduce_columns/reduce_columns.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Associative max fold over the per-column values
struct MaxInt {
  __device__ int operator()(const int a, const int b) const { return a < b ? b : a; }
};

/// @brief [kernel] `d_count[j]` = j + 1 when column j holds a non-zero, else 0
///
/// Block j owns column j; each thread strides over the rows and the block
/// combines the flags with __syncthreads_or. Every thread reaches the barrier.
template<typename T>
__global__ void column_nonzero_kernel(const T *const d_A, int *const d_count, const int m,
                                      const int lda) {
  const T *const column = d_A + static_cast<std::size_t>(blockIdx.x) * lda;
  int found = 0;
  for (int i = static_cast<int>(threadIdx.x); i < m && found == 0;
       i += static_cast<int>(blockDim.x)) {
    const T a = column[i];
    found = (elem_ops<T>::real_part(a) != 0 || elem_ops<T>::imag_part(a) != 0) ? 1 : 0;
  }
  const int any = __syncthreads_or(found);
  if (threadIdx.x == 0) {
    d_count[blockIdx.x] = any != 0 ? static_cast<int>(blockIdx.x) + 1 : 0;
  }
}

} // namespace

template<typename T>
void ilalc(const wwr::wwrStream_t stream, const int m, const int n, const T *const d_A,
           const int lda, int *const d_result, int *const d_scratch) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  column_nonzero_kernel<T>
      <<<static_cast<unsigned int>(n), kBlockSize, 0, stream>>>(d_A, d_scratch, m, lda);
  const auto len = static_cast<std::size_t>(n);
  reduce_columns<int>(stream, d_scratch, d_result, len, 1, len, MaxInt{});
}

// One per supported type, matching interface.cppm's `extern template` list.
template void ilalc<float>(wwr::wwrStream_t, int, int, const float *, int, int *, int *);
template void ilalc<double>(wwr::wwrStream_t, int, int, const double *, int, int *, int *);
template void ilalc<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                          const wwr::wwrFloatComplex *, int, int *, int *);
template void ilalc<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                           const wwr::wwrDoubleComplex *, int, int *, int *);

} // namespace calaman::device
