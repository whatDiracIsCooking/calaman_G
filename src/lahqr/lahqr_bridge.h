/**
 * @file lahqr_bridge.h
 * @brief Device-launcher declaration shared between calaman.lahqr's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lahqr.cu
 * directly -- the split laexc_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports.
 * Reading the backend define that header needs is why the module links
 * wwr_backend PRIVATE (CMakeLists.txt).
 *
 * The parameter list is LAPACK's ?lahqr, 1:1: WANTT / WANTZ are bools, every
 * index (@p n, @p ilo, @p ihi, @p iloz, @p ihiz -- all 1-based, and the leading
 * dimensions) is int, and @p h / @p z are caller-owned device matrices. @p wr /
 * @p wi are device eigenvalue vectors the kernel writes, @p info a device int.
 * No WORK array: the kernel runs the whole iteration in a single thread with its
 * own on-stack scratch (the length-3 Householder vector).
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue the double-shift Francis QR Schur reduction, LAPACK's ?lahqr,
///        on @p stream
///
/// One single-thread kernel drives the QR iteration on the active window
/// @p ilo..@p ihi (1-based) of the upper Hessenberg @p h, writing eigenvalues to
/// @p wr / @p wi, accumulating the transform into @p z when @p wantz, and writing
/// @p info on the device (0 on success, >0 the index that failed to converge).
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void lahqr(wwr::wwrStream_t stream, bool wantt, bool wantz, int n, int ilo, int ihi, T *h, int ldh,
           T *wr, T *wi, int iloz, int ihiz, T *z, int ldz, int *info);

} // namespace calaman::device
