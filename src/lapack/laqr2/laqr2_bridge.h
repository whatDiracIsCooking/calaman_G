/**
 * @file laqr2_bridge.h
 * @brief Device-launcher declaration shared between calaman.laqr2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by laqr2.cu
 * directly -- the split laexc_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports.
 *
 * The parameter list is LAPACK's ?laqr2, 1:1: the deflation window H(KWTOP..
 * KBOT) of @p h is diagonalised in the caller-owned device scratch @p v / @p t
 * (both JW-by-JW) and the deflated eigenvalues land in @p sr / @p si; @p ns and
 * @p nd (device ints) report the shift and deflation counts. One thread runs the
 * whole AED -- its own on-stack scratch holds the spike vector and the window
 * reflectors, so @p nh / @p nv / @p wv (the reference's slab-blocking workspace)
 * are accepted for signature parity and the slab updates run unblocked.
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue one aggressive-early-deflation pass, LAPACK's ?laqr2, on
///        @p stream
///
/// One single-thread kernel diagonalises the trailing JW = min(@p nw, @p kbot -
/// @p ktop + 1) window of the Hessenberg @p h (via an inlined ?lahqr), deflates
/// the spike-tip-negligible eigenvalues, and reflects the undeflatable ones back
/// to Hessenberg form, accumulating the window transform into @p v and applying
/// it to @p h (and @p z when @p wantz). The deflated/undeflated counts land in
/// device ints @p nd / @p ns and the window eigenvalues in @p sr / @p si. @p t /
/// @p v are caller-owned JW-by-JW device scratch; all matrices are column-major.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laqr2(wwr::wwrStream_t stream, bool wantt, bool wantz, int n, int ktop, int kbot, int nw,
           T *h, int ldh, int iloz, int ihiz, T *z, int ldz, int *ns, int *nd, T *sr, T *si, T *v,
           int ldv, int nh, T *t, int ldt, int nv, T *wv, int ldwv);

} // namespace calaman::device
