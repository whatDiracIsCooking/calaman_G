/**
 * @file laln2_bridge.h
 * @brief Device-launcher declaration shared between calaman.laln2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by laln2.cu
 * directly -- the split lasy2_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports, so
 * the module passes its stream straight through. Reading the backend define that
 * header needs is why the module links wwr_backend PRIVATE (CMakeLists.txt).
 *
 * The parameter list is LAPACK's ?laln2, 1:1: LTRANS becomes bool, every scalar
 * stays its natural type, and the three outputs (@p x, @p scale, @p xnorm, @p
 * info) are device pointers the caller owns. @p a, @p b and @p x are the NA-by-NA
 * / NA-by-NW column-major slabs the reference expects.
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue the solve (ca*op(A) - w*D) X = SCALE*B, LAPACK's ?laln2, on
///        @p stream
///
/// One single-thread kernel solves the NA-by-NA system for X (NA is 1 or 2),
/// where w = @p wr + i*@p wi is real (@p nw == 1) or complex (@p nw == 2) and D =
/// diag(@p d1, @p d2). Writes @p x, @p scale, @p xnorm and @p info on the device
/// and returns without synchronizing. @p a, @p b are device inputs; the outputs
/// are device pointers.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laln2(wwr::wwrStream_t stream, bool ltrans, int na, int nw, T smin, T ca, const T *a, int lda,
           T d1, T d2, const T *b, int ldb, T wr, T wi, T *x, int ldx, T *scale, T *xnorm,
           int *info);

} // namespace calaman::device
