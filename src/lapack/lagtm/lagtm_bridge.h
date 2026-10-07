/**
 * @file lagtm_bridge.h
 * @brief Device-launcher declaration shared between calaman.lagtm's interface
 *        unit and its device-compiled translation unit
 *
 * Included in interface.cppm's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in lagtm.cu)
 * and by lagtm.cu directly -- langt_bridge.h's split. R is T's real component
 * type, a second parameter so this GMF header needs no fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue B := alpha * op(A) * X + beta * B for the tridiagonal (dl, d, du)
///
/// One thread per element of B; returns without synchronizing. Assumes
/// @p n >= 1 and @p nrhs >= 1 (the host wrapper handles the empty cases).
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of alpha and beta
template<typename T, typename R>
void lagtm(wwr::wwrStream_t stream, Trans trans, std::size_t n, std::size_t nrhs, R alpha,
           const T *d_dl, const T *d_d, const T *d_du, const T *d_x, std::size_t ldx, R beta,
           T *d_b, std::size_t ldb);

} // namespace calaman::device
