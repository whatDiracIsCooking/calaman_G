/**
 * @file lat2_bridge.h
 * @brief Device-launcher declaration shared between calaman.lat2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lat2.cu)
 * and by lat2.cu directly. wwrStream_t and Uplo arrive by #include, since a GMF
 * cannot import.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue SA <- A, narrowed to single, over the @p uplo triangle
///
/// Sets the device int @p info to 1 if any entry of the triangle overflows
/// single precision; the caller has already zeroed it. Returns without
/// synchronizing; launches nothing when n is 0.
///
/// @tparam From, To (double, float) or (wwrDoubleComplex, wwrFloatComplex)
template<typename From, typename To>
void lat2(wwr::wwrStream_t stream, Uplo uplo, std::size_t n, const From *a, std::size_t lda,
          To *sa, std::size_t ldsa, int *info);

} // namespace calaman::device
