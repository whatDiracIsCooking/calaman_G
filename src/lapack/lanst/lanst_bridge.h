/**
 * @file lanst_bridge.h
 * @brief Device-launcher declaration shared between calaman.lanst's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lanst.cu)
 * and by lanst.cu directly -- lange_bridge.h's split. wwrStream_t (runtime.h)
 * and MatrixNorm (common/enums.h) arrive by #include, since a GMF cannot import.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lanst norm @p which of the tridiagonal (d, e)
///
/// One single-block launch; writes @p d_result and returns without
/// synchronizing. Assumes @p n >= 1 (the host wrapper handles n == 0).
///
/// @tparam T Element type; instantiated for float, double
/// @param stream   Stream the launch is enqueued on; all pointers live on its device
/// @param which    Which norm to compute
/// @param n        Order of the tridiagonal
/// @param d_d      Diagonal, length n
/// @param d_e      Off-diagonal, length n-1 (unread when n == 1)
/// @param d_result Device scalar receiving the norm
template<typename T>
void lanst(wwr::wwrStream_t stream, MatrixNorm which, std::size_t n, const T *d_d, const T *d_e,
           T *d_result);

} // namespace calaman::device
