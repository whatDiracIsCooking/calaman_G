/**
 * @file tfttr_bridge.h
 * @brief Device-launcher declaration shared between calaman.tfttr's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by tfttr.cu
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

/// @brief Enqueue the conversion of RFP ARF into the @p uplo triangle of A
///
/// @p transr selects the (conjugate-)transposed RFP layout. Writes only the
/// triangle of A; launches nothing when @p n is 0.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex,
///         wwrDoubleComplex
template<typename T>
void tfttr(wwr::wwrStream_t stream, bool transr, Uplo uplo, std::size_t n, const T *arf, T *a,
           std::size_t lda);

} // namespace calaman::device
