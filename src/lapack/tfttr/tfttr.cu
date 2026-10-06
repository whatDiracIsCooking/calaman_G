// tfttr.cu
//
// The device half of calaman.tfttr: one thread per element of the n-by-n
// square, launched through wwr.extension.parallel_for. Thread k owns A(i, j)
// with i = k % n, j = k / n, so the write to A is contiguous; the RFP slot and
// whether it is stored conjugated come from calaman::tri_index. Threads
// outside the triangle return, which leaves A's opposite triangle untouched.
#include "tfttr_bridge.h"

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
struct TfttrFunctor {
  const T *const arf_;
  T *const a_;
  const std::size_t lda_;
  const std::size_t n_;
  const bool transr_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    if (in_triangle<Upper>(i, j)) {
      const RfpSlot slot = rfp_index<Upper>(transr_, n_, i, j);
      const T v = arf_[slot.index];
      a_[i + j * lda_] = slot.conj ? elem_ops<T>::conj(v) : v;
    }
  }
};

} // namespace

template<typename T>
void tfttr(const wwr::wwrStream_t stream, const bool transr, const Uplo uplo, const std::size_t n,
           const T *const arf, T *const a, const std::size_t lda) {
  if (n == 0) {
    return;
  }
  if (uplo == Uplo::U) {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TfttrFunctor<T, true>{arf, a, lda, n, transr});
  } else {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TfttrFunctor<T, false>{arf, a, lda, n, transr});
  }
}

// Matches interface.cppm's extern template list and instantiations.cpp.
template void tfttr<float>(wwr::wwrStream_t, bool, Uplo, std::size_t, const float *, float *,
                           std::size_t);
template void tfttr<double>(wwr::wwrStream_t, bool, Uplo, std::size_t, const double *, double *,
                            std::size_t);
template void tfttr<wwrFloatComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                     const wwrFloatComplex *, wwrFloatComplex *, std::size_t);
template void tfttr<wwrDoubleComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                      const wwrDoubleComplex *, wwrDoubleComplex *, std::size_t);

} // namespace calaman::device
