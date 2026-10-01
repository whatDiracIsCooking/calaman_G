/**
 * @file laqps_bridge.h
 * @brief Device-launcher declaration shared between calaman.laqps and its
 *        device-compiled translation unit
 *
 * Included by laqps.cppm in its GLOBAL MODULE FRAGMENT, and by laqps.cu
 * directly, the same split laqp2_bridge.h / lacpy_bridge.h use: the declaration
 * lives in the GMF, not the module purview, so a purview name's module linkage
 * cannot stop it binding to the definition compiled in the plain .cu
 * translation unit.
 *
 * laqps's one piece of genuinely device-elementwise work is the LAWN 176
 * partial-norm DOWNDATE that runs per in-block step. Unlike laqp2, the trailing
 * block is NOT yet updated when the downdate fires (the block update is deferred
 * to one gemm after the panel), so a column whose running norm has degraded
 * cannot be recomputed in place yet. The kernel therefore applies only the cheap
 * shrink and RAISES A PERSISTENT FLAG into a device mask; the host reads the mask
 * once after the panel gemm, compacts the flagged indices, and recomputes those
 * norms exactly with wwr::nrm2. This is the device mask/flag array the issue asks
 * for in place of the reference's serial LSTICC linked list.
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

/// @brief Enqueue the deferred LAWN 176 partial-norm downdate over the trailing
///        columns j > k of an in-block laqps step
///
/// For each of the @p count trailing columns (local index c = 0 .. count-1,
/// global column j = k + 1 + c) this applies the cheap Businger-Golub downdate
/// to @p vn1 in place and, where the relative-precision test trips, records the
/// column for a post-gemm exact recompute by writing @p flags[c] = 1 (otherwise
/// 0). Specifically, with t = |row[j]| / vn1[j] and d = max(0, (1-t)(1+t)):
///   - if d * (vn1[j]/vn2[j])^2 <= tol3z: leave vn1[j]/vn2[j] untouched, set
///     flags[c] = 1 (the host will recompute vn1[j]/vn2[j] exactly AFTER the
///     deferred block gemm, when the trailing column is finally up to date);
///   - else: vn1[j] *= sqrt(d), vn2[j] unchanged, flags[c] = 0.
/// A column whose vn1[j] is already 0 is skipped (flags[c] = 0): a dead column
/// stays dead.
///
/// The flags array persists ACROSS the block's steps: a 1 is never cleared to 0
/// by a later step (the kernel only ever writes 1 or leaves what a step decides),
/// so a column flagged on any step is recomputed once at block end. Enqueued on
/// @p stream; returns without synchronizing; launches nothing when @p count is 0.
///
/// @p row points at A(rk, k+1), the current reflected row restricted to the
/// trailing columns, stride @p lda between consecutive j; @p vn1 and @p vn2 point
/// at the trailing tails vn1[k+1] / vn2[k+1] (stride 1); @p flags points at the
/// trailing tail flags[k+1] (stride 1, @p count elements), the device mask whose
/// nonzero entries the host compacts and recomputes after the panel gemm.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laqps_downdate(wwr::wwrStream_t stream, std::size_t count, const T *row, std::size_t lda,
                    T *vn1, const T *vn2, int *flags, T tol3z);

/// @brief Zero @p count ints of a device flag array on @p stream
///
/// The per-step downdate kernel only ever RAISES a flag (writes 1) or leaves a
/// column's slot as the step decides; it never clears a 1 from an earlier step.
/// So the whole block's flag array must start zeroed before step 1, which this
/// does in one launch. Launches nothing when @p count is 0; does not synchronize.
void laqps_clear_flags(wwr::wwrStream_t stream, std::size_t count, int *flags);

} // namespace calaman::device
