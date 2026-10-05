/**
 * @file set_element_bridge.h
 * @brief Device-launcher declarations shared between calaman.set_element's
 *        interface unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by
 * set_element.cu directly -- the same split complex_cast_bridge.h / laset_bridge.h
 * use: the declarations live in the GMF, not the module purview, so a purview
 * name's module linkage cannot stop them binding to the definitions compiled in
 * the plain .cu translation unit.
 *
 * COMPLEX TYPES DO NOT APPEAR HERE, exactly as in complex_cast_bridge.h: this
 * header is parsed in a host GMF (which cannot import wwr.complex) and in a
 * device pass (where complex.h's builders exist), with no common spelling. So
 * the magnitude launcher is generic in the element type @c T and its real
 * component type @c RealT -- the caller spells RealT as
 * calaman::ComplexToRealType<T>, and the .cu names the concrete types only in its
 * explicit instantiations, in device context.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * handle's stream straight through. Reading the backend define that header needs
 * is why the module links wwr_backend PRIVATE -- see this directory's
 * CMakeLists.txt.
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief d_result[0] <- d_x[(*d_idx - 1) * incx]: gather one element by a
///        device-held index
///
/// Reads the 1-based index @p d_idx holds (the convention iamax writes) and
/// copies that element of @p d_x into @p d_result. All three pointers are device
/// memory; the index is read on the device, so the handle's pointer mode is
/// irrelevant. Enqueued on @p stream; returns without synchronizing.
///
/// @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
template<typename T>
void set_element(wwr::wwrStream_t stream, const T *d_x, int incx, const int *d_idx, T *d_result);

/// @brief d_result[0] <- |d_x[(*d_idx - 1) * incx]|: the indexed element's magnitude
///
/// Like set_element, but writes the element's modulus -- a real value, so the
/// output type @c RealT differs from @c T for a complex element. This is the
/// device-pointer-mode back half of iamax: it turns iamax's device index into
/// the max-magnitude value cuBLAS/hipBLAS expose no call for.
///
/// @tparam T     Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT T's real component type; call with calaman::ComplexToRealType<T>
template<typename T, typename RealT>
void set_element_abs(wwr::wwrStream_t stream, const T *d_x, int incx, const int *d_idx,
                     RealT *d_result);

} // namespace calaman::device
