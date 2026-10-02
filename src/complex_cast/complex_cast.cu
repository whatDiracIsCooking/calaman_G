// complex_cast.cu
//
// The device-kernel half of calaman.complex_cast: one per-element functor and
// one launcher per direction (set/get real/imag), each declared in
// complex_cast_bridge.h. There is no host driver -- the four module functions
// are thin wrappers that just forward to these launchers -- so unlike gebal
// nothing but these kernels lives on the device path. Shared unchanged between
// both backends, like lacpy.cu: a .cu is compiled by the backend compiler, so
// parallel_for's launch machinery and the complex types/accessors arrive through
// their respective device headers.
//
// NEUTRAL COMPLEX, NEVER .x/.y. cuFloatComplex is an operator-less float2 whose
// components are reached as .x/.y, but hipFloatComplex is a class -- so raw
// field access is not portable. Component reads go through
// calaman::device::elem_ops<T>::real_part / imag_part and a new value is built
// with make_complex (elem_ops/elem_ops.cuh), which is where the precision-
// divergent wwrC* accessors and make_wwr*Complex are spelled once. This .cu used
// to carry its own complex_ops struct for exactly that; it is now the shared header.
#include "complex_cast_bridge.h"

#include "complex.h"
#include "elem_ops/elem_ops.cuh"
#include "extension/parallel_for/parallel_for.cuh"

#include <cstddef>

namespace calaman::device {

// complex.h puts the neutral complex types in namespace wwr; pull the two type
// names in so the explicit instantiations below can spell them bare. The
// accessor / constructor calls stay wwr::-qualified.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// The four functors. All members are const scalar (pointer) members, as
// device_functor requires: trivially copyable, and the const deletes the
// copy-assignment the grid-constant kernel copy would otherwise permit. One
// thread per element; the access is contiguous, so consecutive threads touch
// consecutive addresses.

template<typename ComplexT, typename RealT>
struct SetRealPartFunctor {
  ComplexT *const output_;
  const RealT *const input_;

  __device__ void operator()(const std::size_t i) const {
    // Rebuild to keep the imaginary component -- there is no portable "set only
    // the real field" accessor.
    output_[i] = make_complex(input_[i], elem_ops<ComplexT>::imag_part(output_[i]));
  }
};

template<typename ComplexT, typename RealT>
struct SetImagPartFunctor {
  ComplexT *const output_;
  const RealT *const input_;

  __device__ void operator()(const std::size_t i) const {
    output_[i] = make_complex(elem_ops<ComplexT>::real_part(output_[i]), input_[i]);
  }
};

template<typename ComplexT, typename RealT>
struct GetRealPartFunctor {
  RealT *const output_;
  const ComplexT *const input_;

  __device__ void operator()(const std::size_t i) const {
    output_[i] = elem_ops<ComplexT>::real_part(input_[i]);
  }
};

template<typename ComplexT, typename RealT>
struct GetImagPartFunctor {
  RealT *const output_;
  const ComplexT *const input_;

  __device__ void operator()(const std::size_t i) const {
    output_[i] = elem_ops<ComplexT>::imag_part(input_[i]);
  }
};

} // namespace

template<typename ComplexT, typename RealT>
void set_real_part(const wwr::wwrStream_t stream, ComplexT *const output, const RealT *const input,
                   const std::size_t count) {
  if (count < 1) {
    return;
  }
  const SetRealPartFunctor<ComplexT, RealT> functor{output, input};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

template<typename ComplexT, typename RealT>
void set_imag_part(const wwr::wwrStream_t stream, ComplexT *const output, const RealT *const input,
                   const std::size_t count) {
  if (count < 1) {
    return;
  }
  const SetImagPartFunctor<ComplexT, RealT> functor{output, input};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

template<typename ComplexT, typename RealT>
void get_real_part(const wwr::wwrStream_t stream, RealT *const output, const ComplexT *const input,
                   const std::size_t count) {
  if (count < 1) {
    return;
  }
  const GetRealPartFunctor<ComplexT, RealT> functor{output, input};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

template<typename ComplexT, typename RealT>
void get_imag_part(const wwr::wwrStream_t stream, RealT *const output, const ComplexT *const input,
                   const std::size_t count) {
  if (count < 1) {
    return;
  }
  const GetImagPartFunctor<ComplexT, RealT> functor{output, input};
  wwr::extension::parallel_for<std::size_t>(stream, count, functor);
}

// One pair per supported precision, matching complex_cast_bridge.h's declarations
// and the module's use sites. This list and interface.cppm's extern-template list
// must stay in step -- a type added here without being declared there, or vice
// versa, links against nothing.
template void set_real_part<wwrFloatComplex, float>(wwr::wwrStream_t, wwrFloatComplex *,
                                                    const float *, std::size_t);
template void set_real_part<wwrDoubleComplex, double>(wwr::wwrStream_t, wwrDoubleComplex *,
                                                      const double *, std::size_t);

template void set_imag_part<wwrFloatComplex, float>(wwr::wwrStream_t, wwrFloatComplex *,
                                                    const float *, std::size_t);
template void set_imag_part<wwrDoubleComplex, double>(wwr::wwrStream_t, wwrDoubleComplex *,
                                                      const double *, std::size_t);

template void get_real_part<wwrFloatComplex, float>(wwr::wwrStream_t, float *,
                                                    const wwrFloatComplex *, std::size_t);
template void get_real_part<wwrDoubleComplex, double>(wwr::wwrStream_t, double *,
                                                      const wwrDoubleComplex *, std::size_t);

template void get_imag_part<wwrFloatComplex, float>(wwr::wwrStream_t, float *,
                                                    const wwrFloatComplex *, std::size_t);
template void get_imag_part<wwrDoubleComplex, double>(wwr::wwrStream_t, double *,
                                                      const wwrDoubleComplex *, std::size_t);

} // namespace calaman::device
