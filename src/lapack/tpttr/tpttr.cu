// tpttr.cu
//
// The device half of calaman.tpttr: one thread per element of the n-by-n
// square, launched through wwr.extension.parallel_for. Thread k owns A(i, j)
// with i = k % n, j = k / n -- consecutive threads walk a column, so both the
// write to A and the read from AP are contiguous. Threads outside the triangle
// return, which is what leaves the opposite triangle of A untouched.
#include "tpttr_bridge.h"

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
struct TpttrFunctor {
  const T *const ap_;
  T *const a_;
  const std::size_t n_;
  const std::size_t lda_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    if (in_triangle<Upper>(i, j)) {
      a_[i + j * lda_] = ap_[packed_index<Upper>(n_, i, j)];
    }
  }
};

} // namespace

template<typename T>
void tpttr(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n, const T *const ap,
           T *const a, const std::size_t lda) {
  if (n == 0) {
    return;
  }
  if (uplo == Uplo::U) {
    wwr::extension::parallel_for<std::size_t>(stream, n * n, TpttrFunctor<T, true>{ap, a, n, lda});
  } else {
    wwr::extension::parallel_for<std::size_t>(stream, n * n, TpttrFunctor<T, false>{ap, a, n, lda});
  }
}

// Matches interface.cppm's extern template list and instantiations.cpp.
template void tpttr<float>(wwr::wwrStream_t, Uplo, std::size_t, const float *, float *,
                           std::size_t);
template void tpttr<double>(wwr::wwrStream_t, Uplo, std::size_t, const double *, double *,
                            std::size_t);
template void tpttr<wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t, const wwrFloatComplex *,
                                     wwrFloatComplex *, std::size_t);
template void tpttr<wwrDoubleComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                      const wwrDoubleComplex *, wwrDoubleComplex *, std::size_t);

} // namespace calaman::device
