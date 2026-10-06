/**
 * @file laev2_bridge.h
 * @brief Device-launcher declaration shared between calaman.laev2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by laev2.cu
 * directly -- the split lartg_bridge.h uses, so the purview's module linkage
 * cannot stop the declaration binding to the .cu definition. wwrStream_t
 * arrives from runtime.h (a GMF cannot import), which is why the module links
 * wwr_backend PRIVATE.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Batch @p n ?laev2 calls: eigen-decompose [a[k] b[k]; b[k] c[k]]
///
/// Enqueued on @p stream; returns without synchronizing. Launches nothing when
/// @p n is 0. rt1, rt2, cs1, sn1 must be distinct.
///
/// @tparam T Real element type (float, double)
template<typename T>
void laev2(wwr::wwrStream_t stream, std::size_t n, const T *a, const T *b, const T *c, T *rt1,
           T *rt2, T *cs1, T *sn1);

} // namespace calaman::device
