/**
 * @file lapy2_bridge.h
 * @brief Device-launcher declaration shared between calaman.lapy2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by lapy2.cu
 * directly -- the split lartg_bridge.h uses, so the purview's module linkage
 * cannot stop the declaration binding to the .cu definition. wwrStream_t
 * arrives from runtime.h (a GMF cannot import), which is why the module links
 * wwr_backend PRIVATE.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Batch @p n ?lapy2 calls: r[k] = sqrt(x[k]^2 + y[k]^2), overflow-safe
///
/// Enqueued on @p stream; returns without synchronizing. Launches nothing when
/// @p n is 0.
///
/// @tparam T Real element type (float, double)
template<typename T>
void lapy2(wwr::wwrStream_t stream, std::size_t n, const T *x, const T *y, T *r);

} // namespace calaman::device
