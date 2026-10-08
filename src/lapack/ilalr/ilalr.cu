// ilalr.cu
//
// The device half of calaman.ilalr: the count of leading rows of a
// column-major matrix that hold a non-zero, as a two-stage max-reduction.
// Stage one is this file's kernel, one block per column, writing the 1-based
// index of that column's last non-zero (0 for an all-zero column); stage two
// folds those n counts with calaman.reduce_columns' max. Only rows [0, m) of
// each column are read, never the lda - m padding.
//
// "Non-zero" is the reference's `A(i,j).NE.ZERO`: a NaN counts, -0 does not,
// and a complex entry counts when either component does.
#include "ilalr_bridge.h"

#include "common/elem_ops.cuh"
#include "reduce_columns/reduce_columns.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief Associative max fold over the per-column counts
struct MaxInt {
  __device__ int operator()(const int a, const int b) const { return a < b ? b : a; }
};

/// @brief [kernel] `d_last[j]` = 1 + the row of column j's last non-zero, or 0
///
/// Block j owns column j. Each thread strides over the rows, keeping the
/// largest non-zero row it saw; a shared atomicMax combines the block.
template<typename T>
__global__ void column_last_row_kernel(const T *const d_A, int *const d_last, const int m,
                                       const int lda) {
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    s_last = 0;
  }
  __syncthreads();

  const T *const column = d_A + static_cast<std::size_t>(blockIdx.x) * lda;
  int last = 0;
  for (int i = static_cast<int>(threadIdx.x); i < m; i += static_cast<int>(blockDim.x)) {
    const T a = column[i];
    if (elem_ops<T>::real_part(a) != 0 || elem_ops<T>::imag_part(a) != 0) {
      last = i + 1;
    }
  }
  if (last != 0) {
    atomicMax(&s_last, last);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    d_last[blockIdx.x] = s_last;
  }
}

} // namespace

template<typename T>
void ilalr(const wwr::wwrStream_t stream, const int m, const int n, const T *const d_A,
           const int lda, int *const d_result, int *const d_scratch) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  column_last_row_kernel<T>
      <<<static_cast<unsigned int>(n), kBlockSize, 0, stream>>>(d_A, d_scratch, m, lda);
  const auto len = static_cast<std::size_t>(n);
  reduce_columns<int>(stream, d_scratch, d_result, len, 1, len, MaxInt{});
}

// One per supported type, matching interface.cppm's `extern template` list.
template void ilalr<float>(wwr::wwrStream_t, int, int, const float *, int, int *, int *);
template void ilalr<double>(wwr::wwrStream_t, int, int, const double *, int, int *, int *);
template void ilalr<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                          const wwr::wwrFloatComplex *, int, int *, int *);
template void ilalr<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                           const wwr::wwrDoubleComplex *, int, int *, int *);

} // namespace calaman::device
