/**
 * @file inv_sqrt_bridge.h
 * @brief Device-launcher declaration shared between the module's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by inv_sqrt.cu
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

#include <cstdint>

namespace calaman::device {

/// @brief In place, map each of the first @p count elements of @p data to
///        1/sqrt(x) -- except those at or below @p floor, set to 0
///
/// Flooring drops near-null / non-positive eigenvalues; happening BEFORE the
/// square root, it also keeps the result free of NaN/Inf. Enqueued on @p stream,
/// returns without synchronizing, launches nothing when @p count < 1.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream Stream the launch is enqueued on
/// @param count Number of elements to transform
/// @param data Device buffer, overwritten in place
/// @param floor Elements at or below this are set to 0; should be non-negative
template<typename T>
void inverse_sqrt(wwr::wwrStream_t stream, std::int32_t count, T *data, T floor);

} // namespace calaman::device
