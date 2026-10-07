/**
 * @file langt_bridge.h
 * @brief Device-launcher declaration shared between calaman.langt's interface
 *        unit and its device-compiled translation unit
 *
 * Included in interface.cppm's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in langt.cu)
 * and by langt.cu directly -- lange_bridge.h's split. wwrStream_t (runtime.h)
 * and MatrixNorm (common/enums.h) arrive by #include, since a GMF cannot
 * import. R is T's real component type, a second parameter rather than
 * ComplexToRealType<T> so this GMF header needs no fp_types.h (as for lange).
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?langt norm of the tridiagonal (dl, d, du)
///
/// One single-block launch; writes @p d_result and returns without
/// synchronizing. Assumes @p n >= 1 (the host wrapper handles n == 0).
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the result
/// @param stream   Stream the launch is enqueued on; all pointers live on its device
/// @param which    Which norm to compute
/// @param n        Order of the tridiagonal
/// @param d_dl     Sub-diagonal, length n-1 (unread when n == 1)
/// @param d_d      Diagonal, length n
/// @param d_du     Super-diagonal, length n-1 (unread when n == 1)
/// @param d_result Device scalar receiving the norm
template<typename T, typename R>
void langt(wwr::wwrStream_t stream, MatrixNorm which, std::size_t n, const T *d_dl, const T *d_d,
           const T *d_du, R *d_result);

} // namespace calaman::device
