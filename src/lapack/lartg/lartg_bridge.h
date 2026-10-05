/**
 * @file lartg_bridge.h
 * @brief Device-launcher declaration shared between calaman.lartg's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lartg.cu
 * directly -- the same split lacgv_bridge.h / lascl2_bridge.h use: the
 * declaration lives in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop it binding to the definition compiled in the plain
 * .cu translation unit.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * stream straight through. Reading the backend define that header needs is why
 * the module links wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Generate @p n plane rotations from the arrays @p f, @p g
///
/// For each k in [0, n): (c[k], s[k]) rotate (f[k], g[k]) to (r[k], 0), by
/// LAPACK ?lartg's safe-scaled formula. Enqueued on @p stream; returns without
/// synchronizing. Launches nothing when @p n is 0. c, s, r must be distinct.
///
/// @tparam T Real element type (float, double)
template<typename T>
void lartg(wwr::wwrStream_t stream, std::size_t n, const T *f, const T *g, T *c, T *s, T *r);

} // namespace calaman::device
