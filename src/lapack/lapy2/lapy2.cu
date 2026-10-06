// lapy2.cu
//
// The device half of calaman.lapy2: one per-element functor that calls the
// shared lapy2_scalar from "lapack/sym2x2/sym2x2.cuh", launched through
// wwr.extension.parallel_for. Shared unchanged between both backends.
#include "lapack/sym2x2/sym2x2.cuh"
#include "lapy2_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

// Members are const scalars (pointers), as device_functor requires.
template<typename T>
struct Lapy2Functor {
  const T *const x_;
  const T *const y_;
  T *const r_;

  __device__ void operator()(const std::size_t k) const { r_[k] = lapy2_scalar<T>(x_[k], y_[k]); }
};

} // namespace

template<typename T>
void lapy2(const wwr::wwrStream_t stream, const std::size_t n, const T *x, const T *y, T *r) {
  if (n == 0) {
    return;
  }
  const Lapy2Functor<T> functor{x, y, r};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// Must match lapy2_bridge.h and interface.cppm's extern-template list.
template void lapy2<float>(wwr::wwrStream_t, std::size_t, const float *, const float *, float *);
template void lapy2<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                            double *);

} // namespace calaman::device
