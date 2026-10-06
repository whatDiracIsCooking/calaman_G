/**
 * @file steqr_bridge.h
 * @brief Device-launcher declaration shared between calaman.steqr's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in steqr.cu)
 * and by steqr.cu directly -- lasr_bridge.h's split. wwrStream_t (runtime.h)
 * and CompZ (common/enums.h) arrive by #include, since a GMF cannot import.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

namespace calaman::device {

/// @brief Enqueue ?steqr on the order-@p n tridiagonal (d, e) as ONE block
///
/// Returns without synchronizing. @p work holds the 2(n-1) saved rotations
/// (cosines then sines) when @p compz is not CompZ::N, else it may be null;
/// @p info is a device int (0, or the count of unconverged off-diagonals).
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void steqr(wwr::wwrStream_t stream, CompZ compz, int n, T *d, T *e, T *z, int ldz, T *work,
           int *info);

} // namespace calaman::device
