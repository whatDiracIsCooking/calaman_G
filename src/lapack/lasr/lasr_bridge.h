/**
 * @file lasr_bridge.h
 * @brief Device-launcher declaration shared between calaman.lasr's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lasr.cu)
 * and by lasr.cu directly -- lanst_bridge.h's split. wwrStream_t (runtime.h)
 * and Side/Pivot/Direct (common/enums.h) arrive by #include, since a GMF cannot
 * import.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue ?lasr on the m-by-n column-major A
///
/// One launch whose blocks each run lasr_block (lasr.h) on a slab of the
/// unrotated dimension; returns without synchronizing. Assumes m, n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of c and s
template<typename T, typename R>
void lasr(wwr::wwrStream_t stream, Side side, Pivot pivot, Direct direct, std::size_t m,
          std::size_t n, const R *c, const R *s, T *A, std::size_t lda);

} // namespace calaman::device
