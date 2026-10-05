/**
 * @file lascl2_bridge.h
 * @brief Device-launcher declaration shared between the module's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lascl2.cu
 * directly. The declaration must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what the .cu is.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather than an
 * `import`, since a GMF cannot import. It is the SAME type wwr.runtime_api
 * exports, so the wrapper passes its stream straight through. Reading the
 * backend define selected_backend.h needs is why the module links wwr_backend
 * PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the diagonal scaling x <-- D * x of the m-by-n column-major x
///
/// Multiplies each column-major element x(i,j) by d(i), the i-th entry of the
/// length-m diagonal vector. Enqueued on @p stream and returns without
/// synchronizing. Launches nothing when m or n is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param d Device diagonal vector, length m (row scale factors)
/// @param x Device matrix, column-major, leading dimension ldx; scaled in place
template<typename T>
void lascl2(wwr::wwrStream_t stream, std::size_t m, std::size_t n, const T *d, T *x,
            std::size_t ldx);

} // namespace calaman::device
