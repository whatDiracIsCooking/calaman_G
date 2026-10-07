/**
 * @file larmm_bridge.h
 * @brief Device-launcher declaration shared between calaman.larmm's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by larmm.cu
 * directly -- the split lapy2_bridge.h uses, so the purview's module linkage
 * cannot stop the declaration binding to the .cu definition. wwrStream_t
 * arrives from runtime.h (a GMF cannot import), which is why the module links
 * wwr_backend PRIVATE.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Batch @p n ?larmm calls: s[k] = larmm(anorm[k], bnorm[k], cnorm[k])
///
/// Enqueued on @p stream; returns without synchronizing. Launches nothing when
/// @p n is 0.
///
/// @tparam T Real element type (float, double)
template<typename T>
void larmm(wwr::wwrStream_t stream, std::size_t n, const T *anorm, const T *bnorm,
           const T *cnorm, T *s);

} // namespace calaman::device
