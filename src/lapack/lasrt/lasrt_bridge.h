/**
 * @file lasrt_bridge.h
 * @brief Device-launcher declaration shared between calaman.lasrt's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lasrt.cu
 * directly -- the same split laset_bridge.h uses: the declaration lives in the
 * GMF, not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * Unlike the geqp3 call graph, lasrt is NOT a BLAS composition: the whole sort
 * (quicksort reverting to insertion sort, an explicit stack) runs inside one
 * device kernel, so there is no host driver reading data back between steps --
 * the module wrapper is a thin front door that validates and forwards to this
 * launcher, like calaman.set_element.
 *
 * `id` is calaman::SortDir (common/enums.h), passed across by #include, not
 * import, for the same reason wwrStream_t is -- a GMF cannot import -- and that
 * it is a plain header enum is exactly what lets the .cu name it directly across
 * this boundary rather than decoding an int contract, the way laset passes
 * Region. wwrStream_t arrives from runtime.h, the SAME type wwr.runtime_api
 * exports, so the wrapper passes its stream straight through; reading the backend
 * define behind it is why the module links wwr_backend PRIVATE (CMakeLists.txt).
 */

#pragma once

#include "common/enums.h"
#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue the in-place sort of the length-@p n device array @p d
///
/// Sorts @p d into increasing (SortDir::I) or decreasing (SortDir::D) order on
/// @p stream and returns without synchronizing. Launches nothing when @p n <= 1
/// (already sorted). A single-thread kernel runs LAPACK's quicksort/insertion.
///
/// @tparam T Element type; instantiated for float, double
/// @param id Sort direction (SortDir::I or SortDir::D)
template<typename T>
void lasrt(wwr::wwrStream_t stream, SortDir id, int n, T *d);

} // namespace calaman::device
