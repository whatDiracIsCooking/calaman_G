// larmm.cu
//
// The device half of calaman.larmm: one per-element functor that calls the
// header-only larmm_scalar from "lapack/larmm/larmm.cuh", launched through
// wwr.extension.parallel_for. Shared unchanged between both backends.
#include "lapack/larmm/larmm.cuh"
#include "larmm_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

// Members are const scalars (pointers), as device_functor requires.
template<typename T>
struct LarmmFunctor {
  const T *const anorm_;
  const T *const bnorm_;
  const T *const cnorm_;
  T *const s_;

  __device__ void operator()(const std::size_t k) const {
    s_[k] = larmm_scalar<T>(anorm_[k], bnorm_[k], cnorm_[k]);
  }
};

} // namespace

template<typename T>
void larmm(const wwr::wwrStream_t stream, const std::size_t n, const T *anorm, const T *bnorm,
           const T *cnorm, T *s) {
  if (n == 0) {
    return;
  }
  const LarmmFunctor<T> functor{anorm, bnorm, cnorm, s};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// Must match larmm_bridge.h and interface.cppm's extern-template list.
template void larmm<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                           const float *, float *);
template void larmm<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                            const double *, double *);

} // namespace calaman::device
