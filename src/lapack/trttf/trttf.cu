// trttf.cu
//
// The device half of calaman.trttf: one thread per element of the n-by-n
// square, launched through wwr.extension.parallel_for. Thread k owns A(i, j)
// with i = k % n, j = k / n, so the read from A is contiguous; the RFP slot and
// whether it is stored conjugated come from calaman::tri_index. Threads
// outside the triangle return; A's opposite triangle is never read.
#include "trttf_bridge.h"

#include "common/elem_ops.cuh"
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
struct TrttfFunctor {
  const T *const a_;
  const std::size_t lda_;
  T *const arf_;
  const std::size_t n_;
  const bool transr_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    if (in_triangle<Upper>(i, j)) {
      const RfpSlot slot = rfp_index<Upper>(transr_, n_, i, j);
      const T v = a_[i + j * lda_];
      arf_[slot.index] = slot.conj ? elem_ops<T>::conj(v) : v;
    }
  }
};

} // namespace

template<typename T>
void trttf(const wwr::wwrStream_t stream, const bool transr, const Uplo uplo, const std::size_t n,
           const T *const a, const std::size_t lda, T *const arf) {
  if (n == 0) {
    return;
  }
  if (uplo == Uplo::U) {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TrttfFunctor<T, true>{a, lda, arf, n, transr});
  } else {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TrttfFunctor<T, false>{a, lda, arf, n, transr});
  }
}

// Matches interface.cppm's extern template list and instantiations.cpp.
template void trttf<float>(wwr::wwrStream_t, bool, Uplo, std::size_t, const float *, std::size_t,
                           float *);
template void trttf<double>(wwr::wwrStream_t, bool, Uplo, std::size_t, const double *,
                            std::size_t, double *);
template void trttf<wwrFloatComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                     const wwrFloatComplex *, std::size_t, wwrFloatComplex *);
template void trttf<wwrDoubleComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                      const wwrDoubleComplex *, std::size_t, wwrDoubleComplex *);

} // namespace calaman::device
