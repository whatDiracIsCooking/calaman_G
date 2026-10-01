/**
 * @file lacpy_bridge.h
 * @brief Device-launcher declaration shared between the module's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lacpy.cu
 * directly. The declaration must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what the .cu is.
 *
 * wwrStream_t arrives from the gpu* layer's include-only bridge header rather
 * than an `import`, since a GMF cannot import. It is the SAME type
 * wwr.runtime_api exports, so the wrapper passes its stream straight through.
 * Reading the backend define selected_backend.h needs is why the module links
 * wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 *
 * `region` is the integer contract with interface.cppm's copy_region enum:
 * 0 = full, 1 = upper triangle, 2 = lower triangle. A plain int rather than the
 * enum keeps the enum a pure module export (the .cu never imports the module).
 */

#pragma once

#include "extension/bridge/gpu_stream_bridge.h"

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the copy of the m-by-n column-major matrix A into B
///
/// Copies the region selected by @p region; leaves every other element of B
/// untouched. Enqueued on @p stream and returns without synchronizing. Launches
/// nothing when m or n is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param region 0 = full, 1 = upper triangle, 2 = lower triangle
template<typename T>
void lacpy(wwr::wwrStream_t stream, int region, std::size_t m, std::size_t n, const T *a,
           std::size_t lda, T *b, std::size_t ldb);

} // namespace calaman::device
