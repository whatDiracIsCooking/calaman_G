/**
 * @file interface.cppm
 * @brief Primary interface for calaman.complex_cast -- GPU-parallel conversion
 *        between real and complex floating-point device arrays
 *
 * Elementwise passes (one parallel_for each) between complex and real planes:
 *
 * | Operation     | Effect                                         |
 * |---------------|------------------------------------------------|
 * | set_real_part | output[i] <- (input[i], Im(output[i]))         |
 * | set_imag_part | output[i] <- (Re(output[i]), input[i])         |
 * | get_real_part | output[i] <- Re(input[i])                      |
 * | get_imag_part | output[i] <- Im(input[i])                      |
 * | split_planes  | re(i,j), im(i,j) <- Re, Im of A(i,j), strided  |
 * | merge_planes  | C(i,j) <- (re(i,j), im(i,j)), strided          |
 *
 * The *_part calls are flat over `count` contiguous elements; a set_* leaves
 * the other component intact. The strided pair walks a column-major rows-by-cols
 * block with a leading dimension per side -- the deinterleave ?lacrm / ?larcm
 * run around their real gemms, generic over which operand is split.
 *
 * Complex types only (calaman::complex_fp); the real component type is
 * calaman::ComplexToRealType<ComplexT>. Each function forwards to a launcher in
 * complex_cast.cu (declared in complex_cast_bridge.h, in the GMF). Everything is
 * enqueued on @p stream, nothing synchronizes, nothing alias-checks.
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
import calaman.common;  // complex_fp, ComplexToRealType

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
export template<calaman::complex_fp ComplexT>
void set_real_part(wwr::wwrStream_t stream, ComplexT *output,
                   const calaman::ComplexToRealType<ComplexT> *input, const std::size_t count) {
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
export template<calaman::complex_fp ComplexT>
void set_imag_part(wwr::wwrStream_t stream, ComplexT *output,
                   const calaman::ComplexToRealType<ComplexT> *input, const std::size_t count) {
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
export template<calaman::complex_fp ComplexT>
void get_real_part(wwr::wwrStream_t stream, calaman::ComplexToRealType<ComplexT> *output,
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
export template<calaman::complex_fp ComplexT>
void get_imag_part(wwr::wwrStream_t stream, calaman::ComplexToRealType<ComplexT> *output,
                   const ComplexT *input, const std::size_t count) {
  device::get_imag_part(stream, output, input, count);
}

/// @brief Split a strided complex matrix into two strided real planes
///
/// re(i,j) <- Re(A(i,j)), im(i,j) <- Im(A(i,j)) over the @p rows by @p cols
/// column-major block; padding rows are neither read nor written. The operand
/// is generic: pass whichever matrix the caller splits. Enqueued on @p stream;
/// a null launch when rows or cols is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream    Stream the launch is enqueued on
/// @param rows      Row count of the block
/// @param cols      Column count of the block
/// @param a         Device complex matrix, leading dimension @p lda >= rows
/// @param lda       Leading dimension of @p a
/// @param re        Device real plane, leading dimension @p ldp >= rows; overwritten
/// @param im        Device imaginary plane, leading dimension @p ldp; overwritten
/// @param ldp       Leading dimension of both planes
export template<calaman::complex_fp ComplexT>
void split_planes(wwr::wwrStream_t stream, const std::size_t rows, const std::size_t cols,
                  const ComplexT *a, const std::size_t lda,
                  calaman::ComplexToRealType<ComplexT> *re,
                  calaman::ComplexToRealType<ComplexT> *im, const std::size_t ldp) {
  device::split_planes(stream, rows, cols, a, lda, re, im, ldp);
}

/// @brief Merge two strided real planes into a strided complex matrix
///
/// C(i,j) <- (re(i,j), im(i,j)) over the @p rows by @p cols column-major block;
/// C's padding rows are untouched. split_planes' inverse. Enqueued on
/// @p stream; a null launch when rows or cols is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream    Stream the launch is enqueued on
/// @param rows      Row count of the block
/// @param cols      Column count of the block
/// @param re        Device real plane, leading dimension @p ldp >= rows
/// @param im        Device imaginary plane, leading dimension @p ldp
/// @param ldp       Leading dimension of both planes
/// @param c         Device complex matrix, leading dimension @p ldc >= rows; overwritten
/// @param ldc       Leading dimension of @p c
export template<calaman::complex_fp ComplexT>
void merge_planes(wwr::wwrStream_t stream, const std::size_t rows, const std::size_t cols,
                  const calaman::ComplexToRealType<ComplexT> *re,
                  const calaman::ComplexToRealType<ComplexT> *im, const std::size_t ldp,
                  ComplexT *c, const std::size_t ldc) {
  device::merge_planes(stream, rows, cols, re, im, ldp, c, ldc);
}

extern template void split_planes<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t,
                                                        std::size_t, const wwr::wwrFloatComplex *,
                                                        std::size_t, float *, float *,
                                                        std::size_t);
extern template void split_planes<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t,
                                                         std::size_t,
                                                         const wwr::wwrDoubleComplex *,
                                                         std::size_t, double *, double *,
                                                         std::size_t);
extern template void merge_planes<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t,
                                                        std::size_t, const float *, const float *,
                                                        std::size_t, wwr::wwrFloatComplex *,
                                                        std::size_t);
extern template void merge_planes<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t,
                                                         std::size_t, const double *,
                                                         const double *, std::size_t,
                                                         wwr::wwrDoubleComplex *, std::size_t);

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
