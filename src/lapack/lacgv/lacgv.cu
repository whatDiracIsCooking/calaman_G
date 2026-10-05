// lacgv.cu
//
// The device-kernel half of calaman.lacgv: one per-element functor that replaces
// a complex element with its conjugate, launched through
// wwr.extension.parallel_for. There is no host driver -- the module function is a
// thin wrapper that forwards here -- so, like complex_cast.cu, nothing but this
// kernel lives on the device path. Shared unchanged between both backends: a .cu
// is compiled by the backend compiler, so parallel_for's launch machinery and the
// complex accessors arrive through their respective device headers.
//
// NEUTRAL COMPLEX, NEVER .x/.y: the conjugate goes through
// calaman::device::elem_ops<ComplexT>::conj (common/elem_ops.cuh), which spells
// the precision-divergent wwrConj/wwrConjf once -- a vendor complex is an
// operator-less float2 on CUDA but a class on HIP, so raw field access is not
// portable.
//
// STRIDE: thread k touches x[k*incx]. For incx < 0 the host launcher below hands
// the functor a base pointer at the far end of the vector (x[(n-1)*|incx|]) and a
// negative step, so the SAME set of elements is conjugated as for +incx -- which
// is all that is observable, conjugation being per-element and order-free. This
// mirrors reference CLACGV's ioff stepping.
#include "lacgv_bridge.h"

#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

// complex.h (via elem_ops.cuh) puts the neutral complex types in namespace wwr;
// pull the two names in so the explicit instantiations below can spell them bare.
// The conj call stays elem_ops-qualified.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// The per-element conjugation. Members are const scalar (a pointer, an int), as
// device_functor requires: trivially copyable, and the const deletes the
// copy-assignment the grid-constant kernel copy would otherwise permit. One
// thread per element; @c base_ is already offset so thread k's element is at
// base_[k*incx_] (incx_ may be negative -- see the file header).
template<typename ComplexT>
struct LacgvFunctor {
  ComplexT *const base_;
  const int incx_;

  __device__ void operator()(const std::size_t k) const {
    const std::ptrdiff_t idx = static_cast<std::ptrdiff_t>(k) * static_cast<std::ptrdiff_t>(incx_);
    base_[idx] = elem_ops<ComplexT>::conj(base_[idx]);
  }
};

} // namespace

template<typename ComplexT>
void lacgv(const wwr::wwrStream_t stream, const int n, ComplexT *const x, const int incx) {
  if (n < 1) {
    return;
  }

  // For incx < 0, start the walk at the far end so base_[k*incx] steps back
  // through x[(n-1)*|incx|], ..., x[0] -- the same elements +incx walks forward
  // over. For incx >= 0 the base is x itself.
  ComplexT *base = x;
  if (incx < 0) {
    base = x + static_cast<std::ptrdiff_t>(n - 1) * static_cast<std::ptrdiff_t>(-incx);
  }

  const LacgvFunctor<ComplexT> functor{base, incx};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(n), functor);
}

// One per supported precision, matching lacgv_bridge.h's declaration and
// interface.cppm's extern-template list -- a type added here without being added
// there, or vice versa, links against nothing.
template void lacgv<wwrFloatComplex>(wwr::wwrStream_t, int, wwrFloatComplex *, int);
template void lacgv<wwrDoubleComplex>(wwr::wwrStream_t, int, wwrDoubleComplex *, int);

} // namespace calaman::device
