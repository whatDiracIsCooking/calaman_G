// tfttp.cu
//
// The device half of calaman.tfttp: one thread per element of the n-by-n
// square, launched through wwr.extension.parallel_for. Thread k owns A(i, j)
// with i = k % n, j = k / n; threads outside the triangle return. The packed
// offset, the RFP slot and whether it is stored conjugated all come from
// calaman::tri_index, so no full matrix is ever formed.
#include "tfttp_bridge.h"

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
struct TfttpFunctor {
  const T *const arf_;
  T *const ap_;
  const std::size_t n_;
  const bool transr_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    if (in_triangle<Upper>(i, j)) {
      const RfpSlot slot = rfp_index<Upper>(transr_, n_, i, j);
      const T v = arf_[slot.index];
      ap_[packed_index<Upper>(n_, i, j)] = slot.conj ? elem_ops<T>::conj(v) : v;
    }
  }
};

} // namespace

template<typename T>
void tfttp(const wwr::wwrStream_t stream, const bool transr, const Uplo uplo, const std::size_t n,
           const T *const arf, T *const ap) {
  if (n == 0) {
    return;
  }
  if (uplo == Uplo::U) {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TfttpFunctor<T, true>{arf, ap, n, transr});
  } else {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TfttpFunctor<T, false>{arf, ap, n, transr});
  }
}

// Matches interface.cppm's extern template list and instantiations.cpp.
template void tfttp<float>(wwr::wwrStream_t, bool, Uplo, std::size_t, const float *, float *);
template void tfttp<double>(wwr::wwrStream_t, bool, Uplo, std::size_t, const double *, double *);
template void tfttp<wwrFloatComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                     const wwrFloatComplex *, wwrFloatComplex *);
template void tfttp<wwrDoubleComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                      const wwrDoubleComplex *, wwrDoubleComplex *);

} // namespace calaman::device
