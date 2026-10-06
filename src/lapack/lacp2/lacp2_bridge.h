/**
 * @file lacp2_bridge.h
 * @brief Device-launcher declaration shared between calaman.lacp2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lacp2.cu)
 * and by lacp2.cu directly -- lacpy_bridge.h's split. wwrStream_t and Region
 * arrive by #include, since a GMF cannot import.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue B(i,j) <- (A(i,j), 0) over the region of the m-by-n real A
///
/// Leaves every element of B outside @p region untouched. Enqueued on @p stream
/// and returns without synchronizing. Launches nothing when m or n is 0.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex, wwrDoubleComplex)
/// @tparam RealT    Its real component type; call with ComplexToRealType<ComplexT>
template<typename ComplexT, typename RealT>
void lacp2(wwr::wwrStream_t stream, Region region, std::size_t m, std::size_t n, const RealT *a,
           std::size_t lda, ComplexT *b, std::size_t ldb);

} // namespace calaman::device
