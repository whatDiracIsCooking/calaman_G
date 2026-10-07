// lapy3.cu
//
// The device half of calaman.lapy3: one per-element functor that calls the
// header-only lapy3_scalar from "lapack/lapy3/lapy3.cuh", launched through
// wwr.extension.parallel_for. Shared unchanged between both backends.
#include "lapack/lapy3/lapy3.cuh"
#include "lapy3_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

// Members are const scalars (pointers), as device_functor requires.
template<typename T>
struct Lapy3Functor {
  const T *const x_;
  const T *const y_;
  const T *const z_;
  T *const r_;

  __device__ void operator()(const std::size_t k) const {
    r_[k] = lapy3_scalar<T>(x_[k], y_[k], z_[k]);
  }
};

} // namespace

template<typename T>
void lapy3(const wwr::wwrStream_t stream, const std::size_t n, const T *x, const T *y, const T *z,
           T *r) {
  if (n == 0) {
    return;
  }
  const Lapy3Functor<T> functor{x, y, z, r};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// Must match lapy3_bridge.h and interface.cppm's extern-template list.
template void lapy3<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                           const float *, float *);
template void lapy3<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                            const double *, double *);

} // namespace calaman::device
