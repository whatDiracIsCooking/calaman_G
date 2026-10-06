/**
 * @file sterf_bridge.h
 * @brief Device-launcher declaration shared between calaman.sterf's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in sterf.cu)
 * and by sterf.cu directly -- lahqr_bridge.h's split. wwrStream_t arrives from
 * runtime.h by #include, since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Enqueue ?sterf on the order-@p n tridiagonal (d, e) on @p stream
///
/// One single-thread kernel runs sterf_serial (sterf.cuh) and writes its INFO
/// to the device int @p info; returns without synchronizing.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void sterf(wwr::wwrStream_t stream, int n, T *d, T *e, int *info);

} // namespace calaman::device
