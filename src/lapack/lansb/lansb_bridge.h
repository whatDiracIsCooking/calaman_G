/**
 * @file lansb_bridge.h
 * @brief Device-launcher declaration shared between the calaman.lansb and
 *        calaman.lanhb interface units and their one device-compiled TU
 *
 * Included in each interface's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lansb.cu)
 * and by lansb.cu directly -- lansp_bridge.h's split; calaman.lanhb reaches it
 * root-relative as "lapack/lansb/lansb_bridge.h". R is T's real component
 * type, a second parameter so this GMF header needs no fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lansb (or, with @p hermitian, ?lanhb) norm of a band matrix
///
/// Two launches (per-column partials into @p d_scratch, then one fold into
/// @p d_result); returns without synchronizing. Assumes @p n >= 1.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param uplo      Which triangle of A the band storage holds
/// @param hermitian Read only the real part of each diagonal entry (?lanhb)
/// @param n         Order of A
/// @param k         Number of super- (Uplo::U) or sub-diagonals (Uplo::L) of A
/// @param d_AB      Band storage, column-major, leading dimension @p ldab
/// @param ldab      Leading dimension of @p d_AB; ldab >= k+1
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of n elements
template<typename T, typename R>
void lansb(wwr::wwrStream_t stream, MatrixNorm which, Uplo uplo, bool hermitian, std::size_t n,
           std::size_t k, const T *d_AB, std::size_t ldab, R *d_result, R *d_scratch);

} // namespace calaman::device
