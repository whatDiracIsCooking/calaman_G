/**
 * @file trttf_bridge.h
 * @brief Device-launcher declaration shared between calaman.trttf's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by trttf.cu
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

/// @brief Enqueue the conversion of the @p uplo triangle of A into RFP ARF
///
/// @p transr selects the (conjugate-)transposed RFP layout. Reads only the
/// triangle of A; launches nothing when @p n is 0.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex,
///         wwrDoubleComplex
template<typename T>
void trttf(wwr::wwrStream_t stream, bool transr, Uplo uplo, std::size_t n, const T *a,
           std::size_t lda, T *arf);

} // namespace calaman::device
