// ladiv.cu
//
// The device-kernel half of calaman.ladiv: one per-element functor that divides
// (a[k]+i*b[k]) by (c[k]+i*d[k]), launched through wwr.extension.parallel_for.
// There is no host driver -- the module function is a thin wrapper that forwards
// here -- so, like lartg.cu, nothing but this kernel lives on the device path.
// Shared unchanged between both backends.
//
// The arithmetic is NOT reimplemented here: each thread calls the header-only
// ladiv_scalar from ladiv.h, the SAME scalar routine laln2 will call per-thread
// and the oracle test calls on the host. Compiled in this .cu's device pass,
// ladiv.h's CLM_HOST_DEVICE expands to __host__ __device__, so the call has
// device linkage -- this file is the proof the scalar helper is device-callable,
// which is the whole point of the module (issue #86). No wwr::math is needed:
// ladiv_scalar uses only compare/multiply/divide, no transcendental.
#include "ladiv.h"
#include "ladiv_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

// The per-element division. Members are const scalar (pointers), as
// device_functor requires: trivially copyable, and the const deletes the
// copy-assignment the grid-constant kernel copy would otherwise permit. The four
// inputs are read before p/q are written, so an output aliasing an input is
// correct, but p and q must be distinct.
template<typename T>
struct LadivFunctor {
  const T *const a_;
  const T *const b_;
  const T *const c_;
  const T *const d_;
  T *const p_;
  T *const q_;

  __device__ void operator()(const std::size_t k) const {
    ladiv_scalar<T>(a_[k], b_[k], c_[k], d_[k], &p_[k], &q_[k]);
  }
};

} // namespace

template<typename T>
void ladiv(const wwr::wwrStream_t stream, const std::size_t n, const T *a, const T *b, const T *c,
           const T *d, T *p, T *q) {
  if (n == 0) {
    return;
  }
  const LadivFunctor<T> functor{a, b, c, d, p, q};
  wwr::extension::parallel_for<std::size_t>(stream, n, functor);
}

// One per supported type, matching ladiv_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void ladiv<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                           const float *, const float *, float *, float *);
template void ladiv<double>(wwr::wwrStream_t, std::size_t, const double *, const double *,
                            const double *, const double *, double *, double *);

} // namespace calaman::device
