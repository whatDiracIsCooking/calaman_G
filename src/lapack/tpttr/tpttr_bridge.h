/**
 * @file tpttr_bridge.h
 * @brief Device-launcher declaration shared between calaman.tpttr's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by tpttr.cu
 * directly: a GMF declaration keeps global linkage, so it binds to the
 * definition compiled in the plain .cu TU. The element type stays a template
 * parameter here; the .cu names the concrete types only in its explicit
 * instantiations, in device context (complex.h's builders are device-gated).
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the unpack of the @p uplo triangle of AP into A
///
/// Writes only the triangle of A; launches nothing when @p n is 0.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex,
///         wwrDoubleComplex
template<typename T>
void tpttr(wwr::wwrStream_t stream, Uplo uplo, std::size_t n, const T *ap, T *a, std::size_t lda);

} // namespace calaman::device
