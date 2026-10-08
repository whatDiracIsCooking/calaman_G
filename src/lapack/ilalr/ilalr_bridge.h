/**
 * @file ilalr_bridge.h
 * @brief Device-launcher declaration shared between calaman.ilalr's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by ilalr.cu
 * directly: a purview declaration would get module linkage and never bind to
 * the plain-TU definition in the .cu. wwrStream_t arrives from runtime.h by
 * #include, since a GMF cannot import.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Enqueue the ?ilalr scan: rows of A up to its last non-zero row
///
/// Writes one per-column count into @p d_scratch, then their max into
/// @p d_result. Returns without synchronizing. Assumes m, n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @param d_A       Column-major m-by-n matrix, leading dimension @p lda
/// @param d_result  Device int receiving the count
/// @param d_scratch Device scratch of @p n ints
template<typename T>
void ilalr(wwr::wwrStream_t stream, int m, int n, const T *d_A, int lda, int *d_result,
           int *d_scratch);

} // namespace calaman::device
