/**
 * @file la_gbamv_bridge.h
 * @brief Device-launcher declaration shared between calaman.la_gbamv's
 *        interface unit and its device-compiled translation unit
 *
 * Included in interface.cppm's GLOBAL MODULE FRAGMENT (a purview name gets
 * module linkage and could never bind to the plain-TU definition in
 * la_gbamv.cu) and by la_gbamv.cu directly -- la_geamv_bridge.h's split. R is
 * T's real component type, a second parameter so this GMF header needs no
 * fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue y := alpha * |op(A)| * |x| + beta * |y| for a band A, with the nudge
///
/// One thread per y(i), summing the band in ascending j; returns without
/// synchronizing. @p x and @p y point at the LOGICAL first element (the far end
/// of a negative stride). Assumes @p m, @p n >= 1, kl < m, ku < n.
///
/// @tparam T Element type of A and x; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of alpha, beta, y and safe1
/// @param safe1 The nudge, (n + 1) times the underflow threshold
template<typename T, typename R>
void la_gbamv(wwr::wwrStream_t stream, Trans trans, std::size_t m, std::size_t n, std::size_t kl,
              std::size_t ku, R alpha, const T *ab, std::size_t ldab, const T *x,
              std::ptrdiff_t incx, R beta, R *y, std::ptrdiff_t incy, R safe1);

} // namespace calaman::device
