/**
 * @file laruv_bridge.h
 * @brief Device-launcher declaration shared between calaman.laruv's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in laruv.cu)
 * and by laruv.cu directly. wwrStream_t arrives from runtime.h by #include,
 * since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Most outputs one ?laruv call yields: the rows of its multiplier table
inline constexpr int kLaruvMaxN = 128;

/// @brief Enqueue ?laruv: @p n (1..kLaruvMaxN) uniform (0,1) draws into @p x
///
/// One block of kLaruvMaxN threads; reads and then updates the device seed
/// @p iseed (4 ints). Returns without synchronizing.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laruv(wwr::wwrStream_t stream, int *iseed, int n, T *x);

} // namespace calaman::device
