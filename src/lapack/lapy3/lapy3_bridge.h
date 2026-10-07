/**
 * @file lapy3_bridge.h
 * @brief Device-launcher declaration shared between calaman.lapy3's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by lapy3.cu
 * directly -- the split lapy2_bridge.h uses, so the purview's module linkage
 * cannot stop the declaration binding to the .cu definition. wwrStream_t
 * arrives from runtime.h (a GMF cannot import), which is why the module links
 * wwr_backend PRIVATE.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Batch @p n ?lapy3 calls: r[k] = sqrt(x[k]^2 + y[k]^2 + z[k]^2)
///
/// Enqueued on @p stream; returns without synchronizing. Launches nothing when
/// @p n is 0.
///
/// @tparam T Real element type (float, double)
template<typename T>
void lapy3(wwr::wwrStream_t stream, std::size_t n, const T *x, const T *y, const T *z, T *r);

} // namespace calaman::device
