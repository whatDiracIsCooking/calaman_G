/**
 * @file rscl_bridge.h
 * @brief Device-launcher declaration shared between calaman.rscl's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by rscl.cu
 * directly -- lacgv_bridge.h's split: a purview name gets module linkage and
 * could never bind to the definition compiled in the plain .cu.
 *
 * NEITHER ELEMENT TYPE APPEARS HERE BY NAME. This header is parsed in a host
 * GMF with no complex builder, so the launcher is generic in the element type T
 * and its real component type R (a second parameter rather than
 * ComplexToRealType<T>, so no fp_types.h is needed here); the .cu names the
 * concrete types only in its explicit instantiations, in device context.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF
 * cannot import; reading the backend define behind it is why the module links
 * wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Multiply the length-@p n strided vector @p x by one real factor
///
/// Overwrites x[k*incx] with mul * x[k*incx] for k in [0, n). Enqueued on
/// @p stream; returns without synchronizing. Launches nothing when @p n < 1.
/// @p incx must be positive -- the caller's ?rscl loop calls this once per
/// factor of its decomposition, in the role reference ?rscl gives ?scal.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the factor
template<typename T, typename R>
void rscl(wwr::wwrStream_t stream, std::size_t n, R mul, T *x, int incx);

} // namespace calaman::device
