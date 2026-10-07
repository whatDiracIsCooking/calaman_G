/**
 * @file complex_cast_bridge.h
 * @brief Device-launcher declarations shared between calaman.complex_cast's
 *        interface unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by
 * complex_cast.cu directly -- the same split lacpy_bridge.h / gebal_bridge.h
 * use: the declarations live in the GMF, not the module purview, so a purview
 * name's module linkage cannot stop them binding to the definitions compiled in
 * the plain .cu translation unit.
 *
 * COMPLEX TYPES DO NOT APPEAR HERE. This header is parsed in two contexts that
 * have no common complex builder -- wwr.complex is a module (interface.cppm's
 * GMF cannot import) and complex.h's complex constructors are gated to a device
 * pass (the host GMF parse does not get them). So every launcher is generic in
 * the complex element type @c ComplexT and
 * in its real component type @c RealT, which the caller spells as
 * calaman::ComplexToRealType<ComplexT>. The .cu names the concrete wwrFloatComplex /
 * wwrDoubleComplex only in its explicit instantiations, in device context.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * handle's stream straight through. Reading the backend define that
 * header needs is why the module links wwr_backend PRIVATE -- see this
 * directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief output[i] <- (input[i], Im(output[i])) for i in [0, count)
///
/// Overwrites the real component of each complex element with input[i], leaving
/// the imaginary component untouched. Enqueued on @p stream; returns without
/// synchronizing. Launches nothing when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void set_real_part(wwr::wwrStream_t stream, ComplexT *output, const RealT *input,
                   std::size_t count);

/// @brief output[i] <- (Re(output[i]), input[i]) for i in [0, count)
///
/// Overwrites the imaginary component of each complex element with input[i],
/// leaving the real component untouched. Enqueued on @p stream; returns without
/// synchronizing. Launches nothing when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void set_imag_part(wwr::wwrStream_t stream, ComplexT *output, const RealT *input,
                   std::size_t count);

/// @brief output[i] <- Re(input[i]) for i in [0, count)
///
/// Extracts the real component of each complex element. Enqueued on @p stream;
/// returns without synchronizing. Launches nothing when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void get_real_part(wwr::wwrStream_t stream, RealT *output, const ComplexT *input,
                   std::size_t count);

/// @brief output[i] <- Im(input[i]) for i in [0, count)
///
/// Extracts the imaginary component of each complex element. Enqueued on
/// @p stream; returns without synchronizing. Launches nothing when @p count is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void get_imag_part(wwr::wwrStream_t stream, RealT *output, const ComplexT *input,
                   std::size_t count);

/// @brief Split the rows-by-cols complex A (lda) into real planes re, im (ldp)
///
/// re(i,j) <- Re(A(i,j)), im(i,j) <- Im(A(i,j)), all column-major; the planes'
/// padding rows are untouched. Enqueued on @p stream; launches nothing when
/// rows or cols is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void split_planes(wwr::wwrStream_t stream, std::size_t rows, std::size_t cols, const ComplexT *a,
                  std::size_t lda, RealT *re, RealT *im, std::size_t ldp);

/// @brief Merge real planes re, im (ldp) into the rows-by-cols complex C (ldc)
///
/// C(i,j) <- (re(i,j), im(i,j)), all column-major; C's padding rows are
/// untouched. Enqueued on @p stream; launches nothing when rows or cols is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void merge_planes(wwr::wwrStream_t stream, std::size_t rows, std::size_t cols, const RealT *re,
                  const RealT *im, std::size_t ldp, ComplexT *c, std::size_t ldc);

} // namespace calaman::device
