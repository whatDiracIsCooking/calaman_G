/**
 * @file lanst_bridge.h
 * @brief Device-launcher declaration shared between the calaman.lanst and
 *        calaman.lanht interface units and their one device-compiled TU
 *
 * Included in each interface's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lanst.cu)
 * and by lanst.cu directly -- lange_bridge.h's split; calaman.lanht reaches it
 * root-relative as "lapack/lanst/lanst_bridge.h". wwrStream_t (runtime.h) and
 * MatrixNorm (common/enums.h) arrive by #include, since a GMF cannot import.
 * R is T's real component type, a second parameter rather than
 * ComplexToRealType<T> so this GMF header needs no fp_types.h (as for lange).
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lanst (or, for a complex T, ?lanht) norm of (d, e)
///
/// One single-block launch; writes @p d_result and returns without
/// synchronizing. Assumes @p n >= 1 (the host wrapper handles n == 0).
///
/// @tparam T Off-diagonal type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the diagonal and the result
/// @param stream   Stream the launch is enqueued on; all pointers live on its device
/// @param which    Which norm to compute
/// @param n        Order of the tridiagonal
/// @param d_d      Real diagonal, length n
/// @param d_e      Off-diagonal, length n-1 (unread when n == 1)
/// @param d_result Device scalar receiving the norm
template<typename T, typename R>
void lanst(wwr::wwrStream_t stream, MatrixNorm which, std::size_t n, const R *d_d, const T *d_e,
           R *d_result);

} // namespace calaman::device
