/**
 * @file laqr5_bridge.h
 * @brief Device-launcher declaration shared between calaman.laqr5's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by laqr5.cu
 * directly -- the split laexc_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports.
 *
 * The parameter list is LAPACK's ?laqr5 restricted to the non-accumulated
 * (KACC22 = 0) sweep: no U / WV / WH workspace and no ILOZ/IHIZ-wide block
 * multiplies, so the only arrays are the shifts @p sr / @p si (device, in/out --
 * reordered into real/complex pairs), the Hessenberg @p h (device, in/out) and
 * the accumulator @p z (device, touched only when @p wantz). The order-2 and
 * order-3 reflector scratch V is on-kernel; one thread runs the whole sweep.
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue one non-accumulated multishift QR sweep, LAPACK's ?laqr5
///        (KACC22 = 0), on @p stream
///
/// One single-thread kernel chases chains of @p nshfts / 2 double-shift bulges
/// through rows/columns @p ktop..@p kbot (1-based) of the Hessenberg @p h,
/// applying each order-2/3 reflector directly (no far-from-diagonal matrix
/// multiply). @p sr / @p si are reordered in place; @p z accumulates the sweep
/// from the right over rows @p iloz..@p ihiz when @p wantz; @p h is the full
/// Schur factor when @p wantt. All matrices are caller-owned device memory.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laqr5(wwr::wwrStream_t stream, bool wantt, bool wantz, int n, int ktop, int kbot, int nshfts,
           T *sr, T *si, T *h, int ldh, int iloz, int ihiz, T *z, int ldz);

} // namespace calaman::device
