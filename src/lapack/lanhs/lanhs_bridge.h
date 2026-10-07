/**
 * @file lanhs_bridge.h
 * @brief Device-launcher declaration shared between calaman.lanhs's interface
 *        unit and its device-compiled translation unit
 *
 * Included in the interface's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lanhs.cu)
 * and by lanhs.cu directly -- lange_bridge.h's split. wwrStream_t (runtime.h)
 * and MatrixNorm (common/enums.h) arrive by #include, since a GMF cannot
 * import. R is T's real component type, a second parameter rather than
 * ComplexToRealType<T> so this GMF header needs no fp_types.h (as for lange).
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lanhs norm of an upper Hessenberg matrix
///
/// Two launches (per-column, or for inf per-row, partials into @p d_scratch,
/// then one fold into @p d_result); returns without synchronizing. Entries
/// below the subdiagonal are never read. Assumes @p n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param n         Order of A
/// @param d_A       Column-major matrix, leading dimension @p lda >= n
/// @param lda       Leading dimension of @p d_A
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of n elements
template<typename T, typename R>
void lanhs(wwr::wwrStream_t stream, MatrixNorm which, std::size_t n, const T *d_A,
           std::size_t lda, R *d_result, R *d_scratch);

} // namespace calaman::device
