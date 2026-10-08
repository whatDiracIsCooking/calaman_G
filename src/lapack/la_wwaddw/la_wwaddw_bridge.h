/**
 * @file la_wwaddw_bridge.h
 * @brief Device-launcher declaration shared between calaman.la_wwaddw's
 *        interface unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by la_wwaddw.cu
 * directly: a purview declaration would get module linkage and never bind to
 * the plain-TU definition in the .cu. wwrStream_t arrives from runtime.h by
 * #include, since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the double-word accumulate (x, y) <-- (x, y) + w, elementwise
///
/// The ?LA_WWADDW loop body per element (per component for complex), every add
/// and subtract rounded on its own. Returns without synchronizing. Assumes n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @param x Device high words, length n; updated in place
/// @param y Device low words, length n; updated in place
/// @param w Device addends, length n
template<typename T>
void la_wwaddw(wwr::wwrStream_t stream, std::size_t n, T *x, T *y, const T *w);

} // namespace calaman::device
