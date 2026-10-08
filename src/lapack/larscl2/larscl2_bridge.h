/**
 * @file larscl2_bridge.h
 * @brief Device-launcher declaration shared between calaman.larscl2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by larscl2.cu
 * directly: a purview declaration would get module linkage and never bind to
 * the plain-TU definition in the .cu. wwrStream_t arrives from runtime.h by
 * #include, since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the reciprocal scaling x(i,j) <-- x(i,j) / d(i) of column-major x
///
/// A true divide per element (per component for complex), never a reciprocal
/// multiply. Returns without synchronizing. Assumes m, n >= 1.
///
/// @tparam T     Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam RealT Its real component type; call with ComplexToRealType<T>
/// @param d Device diagonal vector, length m
/// @param x Device matrix, column-major, leading dimension ldx; scaled in place
template<typename T, typename RealT>
void larscl2(wwr::wwrStream_t stream, std::size_t m, std::size_t n, const RealT *d, T *x,
             std::size_t ldx);

} // namespace calaman::device
