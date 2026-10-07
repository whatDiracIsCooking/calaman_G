/**
 * @file lansy_bridge.h
 * @brief Device-launcher declaration shared between the calaman.lansy and
 *        calaman.lanhe interface units and their one device-compiled TU
 *
 * Included in each interface's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lansy.cu)
 * and by lansy.cu directly -- lange_bridge.h's split; calaman.lanhe reaches it
 * root-relative as "lapack/lansy/lansy_bridge.h". wwrStream_t (runtime.h) and
 * MatrixNorm/Uplo (common/enums.h) arrive by #include, since a GMF cannot
 * import. R is T's real component type, a second parameter rather than
 * ComplexToRealType<T> so this GMF header needs no fp_types.h (as for lange).
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lansy (or, with @p hermitian, ?lanhe) norm of one triangle
///
/// Two launches (per-column partials into @p d_scratch, then one fold into
/// @p d_result); returns without synchronizing. Assumes @p n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param uplo      Which triangle of A is read; the other is never touched
/// @param hermitian Read only the real part of each diagonal entry (?lanhe)
/// @param n         Order of A
/// @param d_A       Column-major matrix, leading dimension @p lda >= n
/// @param lda       Leading dimension of @p d_A
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of n elements
template<typename T, typename R>
void lansy(wwr::wwrStream_t stream, MatrixNorm which, Uplo uplo, bool hermitian, std::size_t n,
           const T *d_A, std::size_t lda, R *d_result, R *d_scratch);

} // namespace calaman::device
