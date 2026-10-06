/**
 * @file lansy_bridge.h
 * @brief Device-launcher declaration shared between calaman.lansy's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lansy.cu)
 * and by lansy.cu directly -- lange_bridge.h's split. wwrStream_t (runtime.h)
 * and MatrixNorm/Uplo (common/enums.h) arrive by #include, since a GMF cannot
 * import.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lansy norm @p which of the @p uplo triangle of A
///
/// Two launches (per-column partials into @p d_scratch, then one fold into
/// @p d_result); returns without synchronizing. Assumes @p n >= 1.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param uplo      Which triangle of A is read; the other is never touched
/// @param n         Order of A
/// @param d_A       Column-major matrix, leading dimension @p lda >= n
/// @param lda       Leading dimension of @p d_A
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of n elements
template<typename T>
void lansy(wwr::wwrStream_t stream, MatrixNorm which, Uplo uplo, std::size_t n, const T *d_A,
           std::size_t lda, T *d_result, T *d_scratch);

} // namespace calaman::device
