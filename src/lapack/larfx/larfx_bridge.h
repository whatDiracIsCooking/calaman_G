/**
 * @file larfx_bridge.h
 * @brief Device-launcher declaration shared between calaman.larfx and its
 *        device-compiled translation unit
 *
 * Included by larfx.cppm in its GLOBAL MODULE FRAGMENT, and by larfx.cu
 * directly, the same split laqp2_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * This is larfx's one device stage -- the INLINE small-order path. LAPACK's
 * ?larfx applies H = I - tau*v*v^T with hand-unrolled code when the reflector's
 * order is < 11 and no workspace; above that it defers to ?larf. The GPU keeps
 * that split but not the ten unrolled bodies: a single per-vector reduction is
 * the same math -- one thread per independent vector (a column of C for side L,
 * a row for side R) folds the order-length reflector over it. It is per-element
 * device work, so it is expressed with wwr.extension.parallel_for (a device-code
 * header #included into a .cu), hence this bridge rather than a pure-host
 * composition. The order >= 11 case never reaches here; larfx.cppm routes it to
 * calaman.larf instead.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * handle's stream straight through. Reading the backend define that header needs
 * is why the module links wwr_backend PRIVATE -- see this directory's
 * CMakeLists.txt.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the inline small-order application of H = I - tau*v*v^T
///
/// Applies the reflector to @p count independent vectors, each of length
/// @p order, laid out so vector t starts at C + t * @p stride_between and its
/// r-th entry is at offset r * @p stride_within. For each vector it forms
/// sum = sum_r v[r] * x[r] and updates x[r] -= tau * sum * v[r] -- exactly
/// LAPACK ?larfx's inline body, parallelised one thread per vector. For side L
/// the vectors are C's columns (count = n, order = m, within = 1, between = ldc);
/// for side R they are C's rows (count = m, order = n, within = ldc, between = 1).
/// Enqueued on @p stream and returns without synchronizing; launches nothing when
/// @p count is 0. The caller guarantees @p order in 1..10 and tau != 0 --
/// larfx.cppm handles the empty, no-op and order >= 11 cases before calling.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream Stream the kernel is enqueued on; v and C live on its device
/// @param count Number of independent vectors (columns for L, rows for R)
/// @param order Length of v and of each vector (1..10)
/// @param stride_within Element stride within one vector
/// @param stride_between Stride between consecutive vectors
/// @param v Reflector device vector, length @p order, stride 1 (v(1) is stored)
/// @param tau The reflector scalar (nonzero)
/// @param C Device matrix base pointer, updated in place
template<typename T>
void larfx(wwr::wwrStream_t stream, std::size_t count, std::size_t order,
           std::size_t stride_within, std::size_t stride_between, const T *v, T tau, T *C);

} // namespace calaman::device
