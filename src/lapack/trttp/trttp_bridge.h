/**
 * @file trttp_bridge.h
 * @brief Device-launcher declaration shared between calaman.trttp's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by trttp.cu
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

/// @brief Enqueue the pack of the @p uplo triangle of A into AP
///
/// Reads only the triangle of A; launches nothing when @p n is 0.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex,
///         wwrDoubleComplex
template<typename T>
void trttp(wwr::wwrStream_t stream, Uplo uplo, std::size_t n, const T *a, std::size_t lda, T *ap);

} // namespace calaman::device
