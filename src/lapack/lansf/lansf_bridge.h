/**
 * @file lansf_bridge.h
 * @brief Device-launcher declaration shared between the calaman.lansf and
 *        calaman.lanhf interface units and their one device-compiled TU
 *
 * Included in each interface's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lansf.cu)
 * and by lansf.cu directly -- lansy_bridge.h's split; calaman.lanhf reaches it
 * root-relative as "lapack/lansf/lansf_bridge.h". R is T's real component
 * type, a second parameter so this GMF header needs no fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lansf (or, with @p hermitian, ?lanhf) norm of an RFP matrix
///
/// Two launches (per-column partials into @p d_scratch, then one fold into
/// @p d_result); returns without synchronizing. Assumes @p n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param transr    The (conjugate-)transposed RFP layout (TRANSR = 'T'/'C')
/// @param uplo      Which triangle of A the RFP array holds
/// @param hermitian Read only the real part of each diagonal entry (?lanhf)
/// @param n         Order of A
/// @param d_arf     RFP array, n(n+1)/2 elements
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of n elements
template<typename T, typename R>
void lansf(wwr::wwrStream_t stream, MatrixNorm which, bool transr, Uplo uplo, bool hermitian,
           std::size_t n, const T *d_arf, R *d_result, R *d_scratch);

} // namespace calaman::device
