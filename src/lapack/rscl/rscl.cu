// rscl.cu
//
// The device-kernel half of calaman.rscl: one per-element functor that
// multiplies a strided vector element by a real factor, launched through
// wwr.extension.parallel_for -- lacgv.cu's shape, and the only device code this
// module has.
//
// The overflow-safe decomposition of 1/a lives entirely in the host wrapper
// (interface.cppm), which calls this launcher once per factor of the chain, in
// the role reference ?rscl gives ?scal. The kernel knows nothing of it, so it
// needs no machine constants.
//
// NEUTRAL COMPLEX, NEVER .x/.y: the multiply goes through
// calaman::device::elem_ops<T>::scale (common/elem_ops.cuh), which takes a REAL
// factor for either family -- a real T's `x * mul`, a complex T's per-component
// scaling. That is exactly the ?scal / ?dscal split the reference's four ?rscl
// variants call, under one name.
#include "rscl_bridge.h"

#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

// complex.h (via elem_ops.cuh) puts the neutral complex types in namespace wwr;
// pull the two names in so the explicit instantiations below can spell them bare.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// The per-element scaling. Members are const scalar (a pointer, an int, a real
// factor), as device_functor requires: trivially copyable, and the const
// deletes the copy-assignment the grid-constant kernel copy would permit.
template<typename T, typename R>
struct RsclFunctor {
  T *const x_;
  const R mul_;
  const int incx_;

  __device__ void operator()(const std::size_t k) const {
    const std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(k) * static_cast<std::ptrdiff_t>(incx_);
    x_[idx] = elem_ops<T>::scale(x_[idx], mul_);
  }
};

} // namespace

template<typename T, typename R>
void rscl(const wwr::wwrStream_t stream, const std::size_t n, const R mul, T *const x,
          const int incx) {
  if (n < 1) {
    return;
  }

  const RsclFunctor<T, R> functor{x, mul, incx};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// One per supported type, matching rscl_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here but not there, or
// the reverse, links against nothing.
template void rscl<float, float>(wwr::wwrStream_t, std::size_t, float, float *, int);
template void rscl<double, double>(wwr::wwrStream_t, std::size_t, double, double *, int);
template void rscl<wwrFloatComplex, float>(wwr::wwrStream_t, std::size_t, float, wwrFloatComplex *,
                                           int);
template void rscl<wwrDoubleComplex, double>(wwr::wwrStream_t, std::size_t, double,
                                             wwrDoubleComplex *, int);

} // namespace calaman::device
