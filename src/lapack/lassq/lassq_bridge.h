/**
 * @file lassq_bridge.h
 * @brief Device-launcher declaration shared between calaman.lassq's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lassq.cu
 * directly -- lanst_bridge.h's split: a purview name gets module linkage and
 * could never bind to the definition compiled in the plain .cu.
 *
 * R is T's real component type, a second parameter rather than
 * ComplexToRealType<T> so this GMF header needs no fp_types.h (as for lanst).
 * The complex element type never appears by name: this header is parsed in a
 * host GMF with no complex builder, so the .cu names wwrFloatComplex /
 * wwrDoubleComplex only in its explicit instantiations.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF
 * cannot import; reading the backend define behind it is why the module links
 * wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lassq update of a (scale, sumsq) pair over x
///
/// One single-block launch; reads and overwrites both scalars, returning
/// without synchronizing. @p n == 0 applies only ?lassq's normalization of the
/// incoming pair. A negative @p incx walks from the far end, as ?lassq's ix
/// does; either sign visits the same elements.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of both scalars
/// @param stream   Stream the launch is enqueued on; all pointers live on its device
/// @param n        Number of elements of x
/// @param x        Device vector, stride @p incx; read, never written
/// @param incx     Stride between elements of x (non-zero)
/// @param d_scale  Device scalar, in and out: ?lassq's SCALE
/// @param d_sumsq  Device scalar, in and out: ?lassq's SUMSQ
template<typename T, typename R>
void lassq(wwr::wwrStream_t stream, std::size_t n, const T *x, int incx, R *d_scale, R *d_sumsq);

} // namespace calaman::device
