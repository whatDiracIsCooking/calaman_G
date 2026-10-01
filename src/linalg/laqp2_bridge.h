/**
 * @file laqp2_bridge.h
 * @brief Device-launcher declaration shared between calaman.linalg:laqp2 and its
 *        device-compiled translation unit
 *
 * Included by laqp2.cppm in its GLOBAL MODULE FRAGMENT, and by laqp2.cu
 * directly, the same split lacpy_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * The one piece of laqp2 that is genuinely device elementwise work is the
 * LAWN 176 partial-norm DOWNDATE after each reflector -- a per-trailing-column
 * decision that reads the just-reflected row and the two partial-norm arrays.
 * It is expressed with wwr.extension.parallel_for, which is a device-code header
 * (#included into a .cu), hence this bridge rather than a pure-host composition.
 *
 * wwrStream_t arrives from the gpu* layer's include-only bridge header, since a
 * GMF cannot import; it is the SAME type wwr.runtime_api exports, so the module
 * passes its handle's stream straight through. Reading the backend define that
 * header needs is why the module links wwr_backend PRIVATE -- see this
 * directory's CMakeLists.txt.
 */

#pragma once

#include "extension/bridge/gpu_stream_bridge.h"

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the LAWN 176 partial-norm downdate over trailing columns j > i
///
/// For each of the @p count trailing columns (local index k = 0 .. count-1,
/// global column j = i + 1 + k) this applies the cheap downdate to @p vn1 in
/// place and, where the relative-precision test trips, instead records k in
/// @p flags so the host can recompute that column's norm exactly. Specifically,
/// with t = |row[j]| / vn1[j] and d = max(0, (1-t)(1+t)):
///   - if d * (vn1[j]/vn2[j])^2 <= tol3z: leave vn1[j]/vn2[j] untouched and set
///     flags[k] = 1 (host recomputes vn1[j] = nrm2 of the trailing column and
///     resets vn2[j] = vn1[j]);
///   - else: vn1[j] *= sqrt(d), vn2[j] unchanged, flags[k] = 0.
/// A column whose vn1[j] is already 0 is skipped (flags[k] = 0): a dead column
/// stays dead. Enqueued on @p stream and returns without synchronizing; launches
/// nothing when @p count is 0.
///
/// @p row points at A(i, i+1), i.e. the reflected row restricted to the trailing
/// columns, with stride @p lda between consecutive j; @p vn1 and @p vn2 point at
/// the trailing tails vn1[i+1] / vn2[i+1] (stride 1). @p flags has @p count
/// elements (stride 1).
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laqp2_downdate(wwr::wwrStream_t stream, std::size_t count, const T *row, std::size_t lda,
                    T *vn1, T *vn2, int *flags, T tol3z);

} // namespace calaman::device
