// trttp.cu
//
// The device half of calaman.trttp: one thread per element of the n-by-n
// square, launched through wwr.extension.parallel_for. Thread k owns A(i, j)
// with i = k % n, j = k / n -- consecutive threads walk a column, so both the
// read from A and the write to AP are contiguous. Threads outside the triangle
// return; A's opposite triangle is never read.
#include "trttp_bridge.h"

#include "lapack/tri_index/tri_index.cuh"
#include <complex.h>
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// Members are const scalars, as device_functor requires (trivially copyable).
template<typename T, bool Upper>
struct TrttpFunctor {
  const T *const a_;
  const std::size_t lda_;
  T *const ap_;
  const std::size_t n_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    if (in_triangle<Upper>(i, j)) {
      ap_[packed_index<Upper>(n_, i, j)] = a_[i + j * lda_];
    }
  }
};

} // namespace

template<typename T>
void trttp(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n, const T *const a,
           const std::size_t lda, T *const ap) {
  if (n == 0) {
    return;
  }
  if (uplo == Uplo::U) {
    wwr::extension::parallel_for<std::size_t>(stream, n * n, TrttpFunctor<T, true>{a, lda, ap, n});
  } else {
    wwr::extension::parallel_for<std::size_t>(stream, n * n, TrttpFunctor<T, false>{a, lda, ap, n});
  }
}

// Matches interface.cppm's extern template list and instantiations.cpp.
template void trttp<float>(wwr::wwrStream_t, Uplo, std::size_t, const float *, std::size_t,
                           float *);
template void trttp<double>(wwr::wwrStream_t, Uplo, std::size_t, const double *, std::size_t,
                            double *);
template void trttp<wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t, const wwrFloatComplex *,
                                     std::size_t, wwrFloatComplex *);
template void trttp<wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                      const wwrDoubleComplex *, std::size_t, wwrDoubleComplex *);

} // namespace calaman::device
