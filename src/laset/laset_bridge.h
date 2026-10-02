/**
 * @file laset_bridge.h
 * @brief Device-launcher declaration shared between the module's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by laset.cu
 * directly. The declaration must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what the .cu is.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather
 * than an `import`, since a GMF cannot import. It is the SAME type
 * wwr.runtime_api exports, so the wrapper passes its stream straight through.
 * Reading the backend define selected_backend.h needs is why the module links
 * wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 *
 * `region` is calaman::Region (common/enums.h). It arrives by #include, not
 * import, for the same reason wwrStream_t does -- and that it is a plain header
 * enum, not a module export, is exactly what lets the .cu name it directly
 * across this boundary rather than decoding an int contract.
 */

#pragma once

#include "common/enums.h"
#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the fill of the m-by-n column-major matrix A
///
/// Sets the diagonal of the region @p region selects to @p beta and its
/// off-diagonal elements to @p alpha; leaves every element outside that region
/// untouched. Enqueued on @p stream and returns without synchronizing. Launches
/// nothing when m or n is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param region Which part of A to set (Region::U, Region::L, Region::A)
template<typename T>
void laset(wwr::wwrStream_t stream, Region region, std::size_t m, std::size_t n, T alpha, T beta,
           T *a, std::size_t lda);

} // namespace calaman::device
