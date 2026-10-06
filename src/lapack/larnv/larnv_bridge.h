/**
 * @file larnv_bridge.h
 * @brief Device-launcher declaration shared between calaman.larnv's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in larnv.cu)
 * and by larnv.cu directly. wwrStream_t arrives from runtime.h by #include,
 * since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Outputs per ?laruv call in ?larnv's walk: LV/2
inline constexpr int kLarnvChunk = 64;

/// @brief Enqueue ?larnv: @p n values of distribution @p idist into @p x
///
/// Two kernels: every 64-output chunk in parallel from a jump-ahead seed, then
/// one block that redoes the tail after a single-precision ?laruv retry and
/// writes the advanced seed to @p iseed. Returns without synchronizing.
///
/// @tparam T Element type; instantiated for float, double and the wwr complex types
template<typename T>
void larnv(wwr::wwrStream_t stream, int idist, int *iseed, std::size_t n, T *x);

} // namespace calaman::device
