// tpttf.cu
//
// The device half of calaman.tpttf: one thread per element of the n-by-n
// square, launched through wwr.extension.parallel_for. Thread k owns A(i, j)
// with i = k % n, j = k / n; threads outside the triangle return. The packed
// offset, the RFP slot and whether it is stored conjugated all come from
// calaman::tri_index, so no full matrix is ever formed.
#include "tpttf_bridge.h"

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
struct TpttfFunctor {
  const T *const ap_;
  T *const arf_;
  const std::size_t n_;
  const bool transr_;

  __device__ void operator()(const std::size_t k) const {
    const std::size_t i = k % n_;
    const std::size_t j = k / n_;
    if (in_triangle<Upper>(i, j)) {
      const RfpSlot slot = rfp_index<Upper>(transr_, n_, i, j);
      const T v = ap_[packed_index<Upper>(n_, i, j)];
      arf_[slot.index] = slot.conj ? elem_ops<T>::conj(v) : v;
    }
  }
};

} // namespace

template<typename T>
void tpttf(const wwr::wwrStream_t stream, const bool transr, const Uplo uplo, const std::size_t n,
           const T *const ap, T *const arf) {
  if (n == 0) {
    return;
  }
  if (uplo == Uplo::U) {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TpttfFunctor<T, true>{ap, arf, n, transr});
  } else {
    wwr::extension::parallel_for<std::size_t>(stream, n * n,
                                              TpttfFunctor<T, false>{ap, arf, n, transr});
  }
}

// Matches interface.cppm's extern template list and instantiations.cpp.
template void tpttf<float>(wwr::wwrStream_t, bool, Uplo, std::size_t, const float *, float *);
template void tpttf<double>(wwr::wwrStream_t, bool, Uplo, std::size_t, const double *, double *);
template void tpttf<wwrFloatComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                     const wwrFloatComplex *, wwrFloatComplex *);
template void tpttf<wwrDoubleComplex>(wwr::wwrStream_t, bool, Uplo, std::size_t,
                                      const wwrDoubleComplex *, wwrDoubleComplex *);

} // namespace calaman::device
