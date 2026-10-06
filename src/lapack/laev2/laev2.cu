// laev2.cu
//
// The device half of calaman.laev2: one per-element functor that calls the
// shared laev2_scalar from "lapack/sym2x2/sym2x2.cuh" -- the same body steqr
// calls per-thread -- launched through wwr.extension.parallel_for. Shared
// unchanged between both backends.
#include "laev2_bridge.h"
#include "lapack/sym2x2/sym2x2.cuh"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

// Members are const scalars (pointers), as device_functor requires. Inputs are
// read before outputs are written, so an output may alias an input.
template<typename T>
struct Laev2Functor {
  const T *const a_;
  const T *const b_;
  const T *const c_;
  T *const rt1_;
  T *const rt2_;
  T *const cs1_;
  T *const sn1_;

  __device__ void operator()(const std::size_t k) const {
    T rt1;
    T rt2;
    T cs1;
    T sn1;
    laev2_scalar<T>(a_[k], b_[k], c_[k], &rt1, &rt2, &cs1, &sn1);
    rt1_[k] = rt1;
    rt2_[k] = rt2;
    cs1_[k] = cs1;
    sn1_[k] = sn1;
  }
};

} // namespace

template<typename T>
void laev2(const wwr::wwrStream_t stream, const std::size_t n, const T *a, const T *b, const T *c,
           T *rt1, T *rt2, T *cs1, T *sn1) {
  if (n == 0) {
    return;
  }
  const Laev2Functor<T> functor{a, b, c, rt1, rt2, cs1, sn1};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// Must match laev2_bridge.h and interface.cppm's extern-template list.
template void laev2<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                           const float *, float *, float *, float *, float *);
template void laev2<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                            const double *, double *, double *, double *, double *);

} // namespace calaman::device
