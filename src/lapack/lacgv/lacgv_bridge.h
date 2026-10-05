/**
 * @file lacgv_bridge.h
 * @brief Device-launcher declaration shared between calaman.lacgv's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lacgv.cu
 * directly -- the same split lacpy_bridge.h / complex_cast_bridge.h use: the
 * declaration lives in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop it binding to the definition compiled in the plain
 * .cu translation unit.
 *
 * THE COMPLEX TYPE DOES NOT APPEAR HERE. This header is parsed in a host GMF
 * context that has no complex builder (wwr.complex is a module the GMF cannot
 * import, and complex.h's constructors are gated to a device pass), so the
 * launcher is generic in the complex element type @c ComplexT; the .cu names the
 * concrete wwrFloatComplex / wwrDoubleComplex only in its explicit
 * instantiations, in device context -- exactly as complex_cast_bridge.h does.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * stream straight through. Reading the backend define that header needs is why
 * the module links wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Conjugate the length-@p n strided complex vector @p x in place
///
/// Overwrites x[k*incx] with conj(x[k*incx]) for k in [0, n); for incx < 0 the
/// walk starts at the far end, so the SAME elements are conjugated regardless of
/// sign. Enqueued on @p stream; returns without synchronizing. Launches nothing
/// when @p n < 1. @p incx must be non-zero.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
template<typename ComplexT>
void lacgv(wwr::wwrStream_t stream, int n, ComplexT *x, int incx);

} // namespace calaman::device
