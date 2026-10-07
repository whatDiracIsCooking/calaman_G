/**
 * @file langb_bridge.h
 * @brief Device-launcher declaration shared between calaman.langb's interface
 *        unit and its device-compiled translation unit
 *
 * Included in interface.cppm's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in langb.cu)
 * and by langb.cu directly -- lange_bridge.h's split. wwrStream_t (runtime.h)
 * and MatrixNorm (common/enums.h) arrive by #include, since a GMF cannot
 * import. R is T's real component type, a second parameter rather than
 * ComplexToRealType<T> so this GMF header needs no fp_types.h (as for lange).
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?langb norm of an n-by-n band matrix in LAPACK band storage
///
/// Two launches (per-column, or for MatrixNorm::inf per-row, partials into
/// @p d_scratch, then one fold into @p d_result); returns without
/// synchronizing. Assumes @p n >= 1. A(i,j) lives at `d_AB[ku + i - j + j*ldab]`.
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which norm to compute
/// @param n         Order of A
/// @param kl        Number of sub-diagonals
/// @param ku        Number of super-diagonals
/// @param d_AB      Band storage, column-major, leading dimension @p ldab >= kl+ku+1
/// @param ldab      Leading dimension of @p d_AB
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of n elements
template<typename T, typename R>
void langb(wwr::wwrStream_t stream, MatrixNorm which, std::size_t n, std::size_t kl, std::size_t ku,
           const T *d_AB, std::size_t ldab, R *d_result, R *d_scratch);

} // namespace calaman::device
