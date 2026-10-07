/**
 * @file lacn2_bridge.h
 * @brief Device-launcher declarations shared between calaman.lacn2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT and by lacn2.cu
 * directly (rscl_bridge.h's split). Each launcher enqueues one elementwise pass
 * over a length-n device vector on @p stream and returns without synchronizing;
 * the caller checks the runtime's sticky error. Instantiated for float, double.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief x[i] <-- value for i in [0, n)
template<typename T>
void lacn2_fill(wwr::wwrStream_t stream, int n, T value, T *x);

/// @brief x <-- e_j, the 0-based unit vector (?lacn2's label 50)
template<typename T>
void lacn2_unit(wwr::wwrStream_t stream, int n, int j, T *x);

/// @brief x[i] <-- (x[i] >= 0 ? 1 : -1) and isgn[i] <-- that sign as an int
template<typename T>
void lacn2_sign(wwr::wwrStream_t stream, int n, T *x, int *isgn);

/// @brief *flag <-- 1 when any sign of x differs from isgn; never writes 0
template<typename T>
void lacn2_sign_changed(wwr::wwrStream_t stream, int n, const T *x, const int *isgn, int *flag);

/// @brief x[i] <-- (-1)^i (1 + i/(n-1)), ?lacn2's final-stage test vector (n >= 2)
template<typename T>
void lacn2_altsgn(wwr::wwrStream_t stream, int n, T *x);

} // namespace calaman::device
