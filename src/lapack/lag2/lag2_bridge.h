/**
 * @file lag2_bridge.h
 * @brief Device-launcher declaration shared between calaman.lag2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lag2.cu)
 * and by lag2.cu directly. wwrStream_t arrives from runtime.h by #include,
 * since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue SA <- A, element-converted, over the m-by-n matrix
///
/// Sets the device int @p info to 1 if any entry overflows the narrower type;
/// the caller has already zeroed it. Returns without synchronizing; launches
/// nothing when m or n is 0.
///
/// @tparam From, To One of (double, float), (float, double) and their complex pairs
template<typename From, typename To>
void lag2(wwr::wwrStream_t stream, std::size_t m, std::size_t n, const From *a, std::size_t lda,
          To *sa, std::size_t ldsa, int *info);

} // namespace calaman::device
