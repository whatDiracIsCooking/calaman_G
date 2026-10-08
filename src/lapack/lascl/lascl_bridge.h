/**
 * @file lascl_bridge.h
 * @brief Device-launcher declaration shared between the module's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lascl.cu
 * directly. The declaration must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what the .cu is. Neither element type appears here by
 * name: the launcher is generic in T and its real component type R (a second
 * parameter, so no fp_types.h is needed in this host GMF), as rscl_bridge.h.
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

/// @brief Enqueue the in-place multiply A <-- mul * A of the m-by-n column-major A
///
/// Multiplies each in-leading-block element A(i,j) by the real scalar @p mul.
/// Enqueued on @p stream and returns without synchronizing. Launches nothing
/// when m or n is 0. One guarded factor of the ?lascl scaling; the host wrapper
/// calls this once per factor in the over/underflow-safe sequence.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the factor
/// @param x Device matrix, column-major, leading dimension ldx; scaled in place
template<typename T, typename R>
void lascl(wwr::wwrStream_t stream, std::size_t m, std::size_t n, R mul, T *x, std::size_t ldx);

} // namespace calaman::device
