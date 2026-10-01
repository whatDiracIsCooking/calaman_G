/**
 * @file interface.cppm
 * @brief Primary interface for calaman.complex_cast -- GPU-parallel conversion
 *        between real and complex floating-point device arrays
 *
 * Four elementwise operations over a device array, each one parallel_for pass:
 *
 *     set_real_part:  output[i] <- (input[i], Im(output[i]))   complex <- real
 *     set_imag_part:  output[i] <- (Re(output[i]), input[i])   complex <- real
 *     get_real_part:  output[i] <- Re(input[i])                real <- complex
 *     get_imag_part:  output[i] <- Im(input[i])                real <- complex
 *
 * The two set_* operations touch one component and leave the other intact, so a
 * complex array is assembled from two real arrays by one set_real_part followed
 * by one set_imag_part (or split apart by the two get_*). This is the packing
 * glue a complex solver needs when its inputs or outputs arrive as separate
 * real and imaginary planes.
 *
 * COMPLEX is the whole point here, not the deferred extension it is for larfg /
 * horner: there are no gemm scalars to spell constexpr, only component reads and
 * writes, which complex.cuh's accessors do portably. So the surface is the two
 * complex types ONLY (constrained by wwr::complex_fp) -- a real-to-real cast is
 * the identity and has no place. The real component type is spelled
 * wwr::ComplexToRealType<ComplexT>: float for wwrFloatComplex, double for
 * wwrDoubleComplex.
 *
 * Each function is a thin host wrapper that forwards to a device launcher in
 * complex_cast.cu (declared in complex_cast_bridge.h, included in the GMF), the
 * same module/.cu split calaman.lacpy uses. Everything is enqueued on @p stream
 * and nothing synchronizes; the operations do not alias-check, so in-place reuse
 * is the caller's responsibility.
 *
 * Usage:
 *   import calaman.complex_cast;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   import wwr.complex;       // wwrDoubleComplex
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_z: count complex elements; d_re, d_im: count reals, all device memory
 *   calaman::set_real_part<wwr::wwrDoubleComplex>(stream, d_z, d_re, count);
 *   calaman::set_imag_part<wwr::wwrDoubleComplex>(stream, d_z, d_im, count);
 */

module;

#include "complex_cast_bridge.h"

export module calaman.complex_cast;

import std;
import wwr.runtime_api;      // wwrStream_t
import wwr.complex;          // wwrFloatComplex, wwrDoubleComplex (extern template list)
import wwr.wrappers.common;  // complex_fp, ComplexToRealType

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so each template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Write real values into the real component of complex elements
///
/// output[i] <- (input[i], Im(output[i])) for i in [0, count): the imaginary
/// component of each element is left unchanged. Enqueued on @p stream; returns
/// without synchronizing. A null launch when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream    Stream the launch is enqueued on
/// @param output    Device array of @p count complex elements, updated in place
/// @param input     Device array of @p count real values
/// @param count     Number of elements
export template<wwr::complex_fp ComplexT>
void set_real_part(wwr::wwrStream_t stream, ComplexT *output,
                   const wwr::ComplexToRealType<ComplexT> *input, const std::size_t count) {
  device::set_real_part(stream, output, input, count);
}

/// @brief Write real values into the imaginary component of complex elements
///
/// output[i] <- (Re(output[i]), input[i]) for i in [0, count): the real
/// component of each element is left unchanged. Enqueued on @p stream; returns
/// without synchronizing. A null launch when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream    Stream the launch is enqueued on
/// @param output    Device array of @p count complex elements, updated in place
/// @param input     Device array of @p count real values
/// @param count     Number of elements
export template<wwr::complex_fp ComplexT>
void set_imag_part(wwr::wwrStream_t stream, ComplexT *output,
                   const wwr::ComplexToRealType<ComplexT> *input, const std::size_t count) {
  device::set_imag_part(stream, output, input, count);
}

/// @brief Extract the real component of complex elements into a real array
///
/// output[i] <- Re(input[i]) for i in [0, count). Enqueued on @p stream; returns
/// without synchronizing. A null launch when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream    Stream the launch is enqueued on
/// @param output    Device array of @p count real values, overwritten
/// @param input     Device array of @p count complex elements
/// @param count     Number of elements
export template<wwr::complex_fp ComplexT>
void get_real_part(wwr::wwrStream_t stream, wwr::ComplexToRealType<ComplexT> *output,
                   const ComplexT *input, const std::size_t count) {
  device::get_real_part(stream, output, input, count);
}

/// @brief Extract the imaginary component of complex elements into a real array
///
/// output[i] <- Im(input[i]) for i in [0, count). Enqueued on @p stream; returns
/// without synchronizing. A null launch when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream    Stream the launch is enqueued on
/// @param output    Device array of @p count real values, overwritten
/// @param input     Device array of @p count complex elements
/// @param count     Number of elements
export template<wwr::complex_fp ComplexT>
void get_imag_part(wwr::wwrStream_t stream, wwr::ComplexToRealType<ComplexT> *output,
                   const ComplexT *input, const std::size_t count) {
  device::get_imag_part(stream, output, input, count);
}

extern template void set_real_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                         const float *, std::size_t);
extern template void set_real_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                          const double *, std::size_t);

extern template void set_imag_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                         const float *, std::size_t);
extern template void set_imag_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                          const double *, std::size_t);

extern template void get_real_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, float *,
                                                         const wwr::wwrFloatComplex *, std::size_t);
extern template void get_real_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, double *,
                                                          const wwr::wwrDoubleComplex *,
                                                          std::size_t);

extern template void get_imag_part<wwr::wwrFloatComplex>(wwr::wwrStream_t, float *,
                                                         const wwr::wwrFloatComplex *, std::size_t);
extern template void get_imag_part<wwr::wwrDoubleComplex>(wwr::wwrStream_t, double *,
                                                          const wwr::wwrDoubleComplex *,
                                                          std::size_t);

} // namespace calaman
