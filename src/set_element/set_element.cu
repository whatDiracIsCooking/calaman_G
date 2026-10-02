// set_element.cu
//
// The device-kernel half of calaman.set_element: pick one element of a device
// array by a device-held 1-based index and write it (or its magnitude) to a
// device scalar. Declared in set_element_bridge.h. There is no host driver --
// the two module functions are thin wrappers that forward to these launchers --
// like complex_cast.cu. Shared unchanged between both backends: a .cu is
// compiled by the backend compiler, so parallel_for's launch machinery and the
// complex accessors arrive through their respective device headers.
//
// ONE KERNEL, A UNARY FUNCTOR. The two operations (gather, magnitude) differ
// only in a per-element transform, so one SetElementFunctor carries the index
// read and defers the transform to a UnaryOp template argument -- the gather is
// IdentityOp, the magnitude is AbsOp over calaman::device::elem_ops<T>::modulus
// (elem_ops/elem_ops.cuh), the one portable spelling of |.| across the four
// element types (never .x/.y, which hipComplex lacks). Adding another transform
// is one more functor plus its launcher, no new kernel.
//
// A single element is set, so the launch is one parallel_for item: the outermost
// WarpWraps layer that does the job (CLAUDE.md), one thread reading *d_idx.
#include "set_element_bridge.h"

#include "elem_ops/elem_ops.cuh"
#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

// complex.h (via elem_ops.cuh) puts the neutral complex types in namespace wwr;
// pull the two names in so the explicit instantiations below can spell them bare.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// The per-element transforms. Empty, stateless classes with a __device__
// operator(); the functor below default-constructs one in the kernel rather
// than storing it (a const class-type member would make the functor itself
// non-trivially-copyable, which parallel_for's device_functor concept rejects).
template<typename T>
struct IdentityOp {
  __device__ T operator()(const T v) const { return v; }
};

template<typename T>
struct AbsOp {
  // Returns elem_ops<T>::real_type -- T for a real element, its real component
  // for a complex one -- which is the RealT the launcher is instantiated with.
  __device__ auto operator()(const T v) const { return elem_ops<T>::modulus(v); }
};

// Reads the 1-based index *idx_ and writes UnaryOp applied to that element. All
// members are const (pointer/scalar), as a parallel_for functor wants: the const
// deletes the copy-assignment the grid-constant copy would otherwise permit,
// while keeping the functor trivially copyable. The transform is a stateless
// type argument invoked as UnaryOp{}, not a stored member (see above). The
// offset widens to ptrdiff_t before scaling so a large strided index cannot
// overflow a 32-bit intermediate.
template<typename T, typename ValT, typename UnaryOp>
struct SetElementFunctor {
  const T *const x_;
  const int incx_;
  const int *const idx_;
  ValT *const result_;

  __device__ void operator()(const std::size_t) const {
    const int one_based = *idx_;
    result_[0] = UnaryOp{}(x_[(static_cast<std::ptrdiff_t>(one_based) - 1) * incx_]);
  }
};

// One launch helper the two public launchers share: build the functor and
// enqueue a single parallel_for item (one thread). The transform UnaryOp and the
// output type ValT are explicit template arguments.
template<typename T, typename ValT, typename UnaryOp>
void launch(const wwr::wwrStream_t stream, const T *const x, const int incx, const int *const idx,
            ValT *const result) {
  const SetElementFunctor<T, ValT, UnaryOp> functor{x, incx, idx, result};
  wwr::extension::parallel_for<std::size_t>(stream, 1, functor);
}

} // namespace

template<typename T>
void set_element(const wwr::wwrStream_t stream, const T *const d_x, const int incx,
                 const int *const d_idx, T *const d_result) {
  launch<T, T, IdentityOp<T>>(stream, d_x, incx, d_idx, d_result);
}

template<typename T, typename RealT>
void set_element_abs(const wwr::wwrStream_t stream, const T *const d_x, const int incx,
                     const int *const d_idx, RealT *const d_result) {
  launch<T, RealT, AbsOp<T>>(stream, d_x, incx, d_idx, d_result);
}

// One instantiation per supported type, matching set_element_bridge.h's
// declarations and interface.cppm's extern-template list -- all three lists must
// stay in step, or a type named in one and missing from another links against
// nothing. Gather is T -> T; magnitude is T -> its real component type.
template void set_element<float>(wwr::wwrStream_t, const float *, int, const int *, float *);
template void set_element<double>(wwr::wwrStream_t, const double *, int, const int *, double *);
template void set_element<wwrFloatComplex>(wwr::wwrStream_t, const wwrFloatComplex *, int,
                                           const int *, wwrFloatComplex *);
template void set_element<wwrDoubleComplex>(wwr::wwrStream_t, const wwrDoubleComplex *, int,
                                            const int *, wwrDoubleComplex *);

template void set_element_abs<float, float>(wwr::wwrStream_t, const float *, int, const int *,
                                            float *);
template void set_element_abs<double, double>(wwr::wwrStream_t, const double *, int, const int *,
                                              double *);
template void set_element_abs<wwrFloatComplex, float>(wwr::wwrStream_t, const wwrFloatComplex *, int,
                                                      const int *, float *);
template void set_element_abs<wwrDoubleComplex, double>(wwr::wwrStream_t, const wwrDoubleComplex *,
                                                        int, const int *, double *);

} // namespace calaman::device
