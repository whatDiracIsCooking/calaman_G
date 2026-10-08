/**
 * @file la_syamv_bridge.h
 * @brief Device-launcher declaration shared between calaman.la_syamv's and
 *        calaman.la_heamv's interface units and la_syamv.cu
 *
 * Included in each interface.cppm's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in
 * la_syamv.cu) and by la_syamv.cu directly -- la_geamv_bridge.h's split.
 * la_heamv reaches it root-relative, "lapack/la_syamv/la_syamv_bridge.h". R is
 * T's real component type, a second parameter so this GMF header needs no
 * fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue y := alpha * |A| * |x| + beta * |y| for A held in one triangle, with the nudge
///
/// One thread per y(i), summing j = 0 .. n-1 ascending; returns without
/// synchronizing. @p x and @p y point at the LOGICAL first element (the far end
/// of a negative stride). Assumes @p n >= 1. Serves ?la_syamv and ?la_heamv.
///
/// @tparam T Element type of A and x; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of alpha, beta, y and safe1
/// @param safe1 The nudge, (n + 1) times the underflow threshold
template<typename T, typename R>
void la_syamv(wwr::wwrStream_t stream, Uplo uplo, std::size_t n, R alpha, const T *a,
              std::size_t lda, const T *x, std::ptrdiff_t incx, R beta, R *y, std::ptrdiff_t incy,
              R safe1);

} // namespace calaman::device
