/**
 * @file laqr3_bridge.h
 * @brief Device-launcher declarations shared between calaman.laqr3's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by laqr3.cu
 * directly -- the split laqr2_bridge.h / laexc_bridge.h use: the declarations
 * live in the GMF, not the module purview, so a purview name's module linkage
 * cannot stop them binding to the definitions compiled in the plain .cu.
 *
 * TWO launchers, not one: ?laqr3's window Schur form is the recursive ?laqr4, a
 * HOST composition no device kernel can call, so interface.cppm sequences the AED
 * pass as laqr3_setup (extract the window into T, V := I), then calaman.laqr4 on
 * the host, then laqr3_finish (the deflation / reorder / re-Hessenberg / slab tail
 * -- ?laqr2's body from the clean-margin step on). The JW <= NMIN regime, where
 * ?laqr3 equals ?laqr2 exactly, is not launched here: it is delegated whole to
 * calaman.laqr2 in interface.cppm. laqr3_finish reads INFQR from the device int
 * calaman.laqr4 wrote and recomputes S / KWTOP from H (the Schur touched only
 * T / V).
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launchers are
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Extract the AED window into the spike-triangular @p t and set @p v := I
///
/// First half of one ?laqr3 aggressive-early-deflation pass, enqueued on
/// @p stream as one single-thread kernel. Copies the upper triangle and
/// subdiagonal of the trailing JW = min(@p nw, @p kbot - @p ktop + 1) window of
/// the Hessenberg @p h into @p t (strictly-lower entries zeroed, so @p t is a
/// clean Hessenberg matrix for the window Schur form), and writes the JW-by-JW
/// identity into @p v. The caller enters here only for the general large-window
/// case (JW > NMIN >= 2); nothing is deflated yet. All matrices are column-major.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laqr3_setup(wwr::wwrStream_t stream, int ktop, int kbot, int nw, T *h, int ldh, T *v, int ldv,
                 T *t, int ldt);

/// @brief Resume one ?laqr3 AED pass after the window Schur form is in @p t / @p v
///
/// Second half, enqueued on @p stream as one single-thread kernel: ?laqr2's body
/// from the clean-margin step on. Runs the spike-tip deflation test, moves the
/// undeflatable eigenvalues up and bubble-sorts the survivors (both over inlined
/// ?trexc swaps), reads the eigenvalues back into @p sr / @p si via ?lanv2, and --
/// when any eigenvalue deflated or the spike vanished -- reflects the spike
/// (?larfg), re-Hessenbergs (?gehrd), accumulates (?ormhr) and applies the three
/// unblocked slab updates to @p h (fully when @p wantt) and @p z (when @p wantz).
/// The shift / deflation counts land in device ints @p ns / @p nd. @p infqr is the
/// device int the window Schur form wrote (0, or the row that failed to converge);
/// S and KWTOP are recomputed from @p h. All matrices are column-major.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laqr3_finish(wwr::wwrStream_t stream, bool wantt, bool wantz, int n, int ktop, int kbot,
                  int nw, T *h, int ldh, int iloz, int ihiz, T *z, int ldz, int *ns, int *nd, T *sr,
                  T *si, T *v, int ldv, T *t, int ldt, const int *infqr);

} // namespace calaman::device
