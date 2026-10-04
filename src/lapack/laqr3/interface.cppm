/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laqr3 -- aggressive early deflation for
 *        the real nonsymmetric eigensolver, LAPACK's ?laqr3 (the recursive
 *        variant whose window Schur form comes from ?laqr4)
 *
 * One device routine: on the trailing JW = min(@p nw, @p kbot - @p ktop + 1)
 * window of the upper-Hessenberg @p h, run aggressive early deflation -- the
 * window Schur form (via the recursive multishift ?laqr4 when JW > NMIN, else
 * ?lahqr, exactly as the reference), deflate the eigenvalues whose spike coupling
 * is negligible, reflect the survivors back to Hessenberg form, and apply the
 * accumulated window transform to @p h (and @p z when @p wantz). The
 * deflated/undeflated counts land in device ints @p nd / @p ns and the window
 * eigenvalues in @p sr / @p si (at the positions ?laqr3 documents). Enqueued on
 * the stream; the large-window path synchronizes internally (?laqr4 is a host
 * loop), then returns, like calaman.laqr4. @p h, @p z, @p sr, @p si, the @p v /
 * @p t scratch and the ?laqr4 @p work are device memory the caller owns; nothing
 * is allocated here.
 *
 * ?laqr3 IS ?laqr2 EXCEPT the window Schur form: ?laqr2 always uses ?lahqr,
 * ?laqr3 uses ?laqr4 when JW exceeds the ILAENV(12) crossover NMIN = 75. Because
 * ?laqr4 is a HOST composition no device kernel can call, this driver is a host
 * composition too (calaman.laqr4's shape): for the small-window regime JW <= NMIN,
 * where ?laqr3 equals ?laqr2 exactly, it DELEGATES the whole pass -- quick returns
 * and the 1-by-1 window included -- to the shipped calaman.laqr2 kernel; for
 * JW > NMIN it sequences laqr3_setup (extract the window into @p t, @p v := I),
 * calaman.laqr4 on the host (the window Schur form), and laqr3_finish (the
 * deflation / reorder / re-Hessenberg / slab tail). @p ns doubles as the device
 * INFQR scratch between those two kernels: ?laqr4 writes the convergence flag
 * there and laqr3_finish reads it back before overwriting it with the shift count.
 *
 * REAL ONLY: float / double, the scope ?slaqr3 / ?dlaqr3 has. The reference's
 * NH / NV slab-blocking and WV workspace are accepted for signature parity and
 * ignored (slab updates run unblocked); NW must satisfy NW <= the kernel's
 * on-stack window bound. @p work / @p lwork govern ?laqr4 exactly as the
 * reference does; see the parameter notes.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launchers declared only in the GMF.
 *
 * Usage:
 *   import calaman.laqr3;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_h: N x N Hessenberg; d_z: N x N accumulator; d_v/d_t: NW x NW scratch;
 *   // d_work: ?laqr4 scratch for n = NW (see @p work)
 *   calaman::laqr3<double>(stream, true, true, n, ktop, kbot, nw, d_h, ldh, 1, n,
 *                          d_z, ldz, d_ns, d_nd, d_sr, d_si, d_v, ldv, nw, d_t,
 *                          ldt, nw, d_wv, ldwv, d_work, lwork);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "laqr3_bridge.h"

export module calaman.laqr3;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.wrappers.common; // real_fp
import calaman.laqr2;       // the small-window (?lahqr) pass, delegated whole
import calaman.laqr4;       // the recursive multishift QR -- the window Schur form

// export import, not a plain import: laqr3 RETURNS calaman::Status, so a consumer
// of `import calaman.laqr3;` must see Status's member functions, not just its
// name -- the same re-export laqr2 / laqr4 / laqr5 / lahqr do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief One aggressive-early-deflation pass (LAPACK ?laqr3, ?laqr4 variant) on
///        @p stream
///
/// Diagonalises the trailing JW = min(@p nw, @p kbot - @p ktop + 1) window of the
/// Hessenberg @p h (window Schur form via calaman.laqr4 when JW > NMIN = 75, else
/// the delegated calaman.laqr2 / ?lahqr path), deflates the spike-negligible
/// eigenvalues, reflects the survivors back to Hessenberg form, and applies the
/// window transform to @p h (fully when @p wantt, else only enough to preserve
/// eigenvalues) and to @p z (rows @p iloz..@p ihiz) when @p wantz. The number of
/// deflations lands in @p nd and of shifts in @p ns (device ints); the window
/// eigenvalues in @p sr / @p si. A @p ktop > @p kbot or @p nw < 1 writes only
/// *@p ns = *@p nd = 0, as in the reference. No argument checking beyond those
/// quick returns. All matrices are column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launches are enqueued on; every pointer lives on its device
/// @param wantt Fully update @p h for the Schur factor when true, else eigenvalues only
/// @param wantz Accumulate the window transform into @p z when true
/// @param n Order of @p h (and @p z when @p wantz)
/// @param ktop 1-based first row/column of the isolated diagonal block
/// @param kbot 1-based last row/column of the isolated diagonal block
/// @param nw Deflation window size; 1 <= @p nw <= @p kbot - @p ktop + 1
/// @param h Device N-by-N Hessenberg matrix, leading dimension @p ldh; updated in place
/// @param ldh Leading dimension of @p h
/// @param iloz First row of @p z the transform touches (used only when @p wantz)
/// @param ihiz Last row of @p z the transform touches (used only when @p wantz)
/// @param z Device N-by-N accumulator, leading dimension @p ldz; touched only when @p wantz
/// @param ldz Leading dimension of @p z
/// @param ns Device int; the number of shift (unconverged) eigenvalues returned.
///        Also reused as the device INFQR scratch between the window Schur form
///        and the finish kernel on the JW > NMIN path.
/// @param nd Device int; the number of converged (deflated) eigenvalues
/// @param sr Device length->= @p kbot real parts of the window eigenvalues
/// @param si Device length->= @p kbot imaginary parts of the window eigenvalues
/// @param v Device JW-by-JW scratch (the window orthogonal transform), ld @p ldv
/// @param ldv Leading dimension of @p v (>= @p nw)
/// @param nh Reference slab-blocking width; accepted and ignored (unblocked here)
/// @param t Device JW-by-JW scratch (the window Schur form), leading dimension @p ldt
/// @param ldt Leading dimension of @p t (>= @p nw)
/// @param nv Reference slab-blocking height; accepted and ignored (unblocked here)
/// @param wv Device slab scratch; accepted and ignored (unblocked here)
/// @param ldwv Leading dimension of @p wv
/// @param work Device scratch for the calaman.laqr4 window Schur form on the
///        JW > NMIN path, sized for n = @p nw as calaman.laqr4's @p work
///        documents (>= 2*NWMAX*NWMAX + NSMAX*NSMAX + 4 for that n / @p lwork);
///        unused on the delegated JW <= NMIN path
/// @param lwork Governs calaman.laqr4's NWMAX / NSMAX; pass the value the oracle gets
/// @return Success, or the runtime error a kernel launch or calaman.laqr4 reported
export template<wwr::real_fp T>
Status laqr3(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
             const int ktop, const int kbot, const int nw, T *const h, const int ldh,
             const int iloz, const int ihiz, T *const z, const int ldz, int *const ns,
             int *const nd, T *const sr, T *const si, T *const v, const int ldv, const int nh,
             T *const t, const int ldt, const int nv, T *const wv, const int ldwv, T *const work,
             const int lwork) {
  // ILAENV(12, 'DLAQR3', ...) -- the ?lahqr/?laqr4 crossover NMIN the reference
  // consults for the window Schur form. IPARMQ returns the constant 75 for ISPEC
  // 12, so a window of JW <= NMIN uses ?lahqr (where ?laqr3 == ?laqr2) and a
  // larger one recurses through ?laqr4. (A function-local constexpr, not a
  // module-scope one: calaman.laqr4 exposes an `inline` kNmin of its own, and two
  // inline module-scope variables of the same name clash under import.)
  constexpr int kNmin = 75;
  const int jw = (ktop <= kbot && nw >= 1) ? std::min(nw, kbot - ktop + 1) : 0;

  // ==== JW <= NMIN: ?laqr3 == ?laqr2 (window Schur form via ?lahqr). Delegate
  //      the whole pass -- quick returns and the 1-by-1 window included -- to the
  //      shipped calaman.laqr2 kernel. work / lwork are unused here. ====
  if (jw <= kNmin) {
    return laqr2<T>(stream, wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, ns, nd, sr,
                    si, v, ldv, nh, t, ldt, nv, wv, ldwv);
  }

  // ==== JW > NMIN: the recursive path. Extract the window, diagonalise it on the
  //      host with calaman.laqr4, then resume the deflation / reorder / slab tail.
  //      @p ns doubles as the device INFQR the window Schur form writes and the
  //      finish kernel reads (it reads INFQR before overwriting *ns). ====
  const int kwtop = kbot - jw + 1;
  int *const d_infqr = ns;

  device::laqr3_setup<T>(stream, ktop, kbot, nw, h, ldh, v, ldv, t, ldt);
  CLM_TRY(wwr::wwrGetLastError());

  // DLAQR4('V','V', JW, 1, JW, T, LDT, SR(KWTOP), SI(KWTOP), 1, JW, V, LDV, WORK,
  // LWORK, INFQR): the eigenvalues land at SR/SI positions KWTOP.. (1-based).
  CLM_TRY(laqr4<T>(stream, true, true, jw, 1, jw, t, ldt, sr + (kwtop - 1), si + (kwtop - 1), 1, jw,
                   v, ldv, work, lwork, d_infqr));

  device::laqr3_finish<T>(stream, wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, ns,
                          nd, sr, si, v, ldv, t, ldt, d_infqr);
  // The launchers return void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laqr2 does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laqr3<float>(wwr::wwrStream_t, bool, bool, int, int, int, int, float *, int,
                                    int, int, float *, int, int *, int *, float *, float *, float *,
                                    int, int, float *, int, int, float *, int, float *, int);
extern template Status laqr3<double>(wwr::wwrStream_t, bool, bool, int, int, int, int, double *,
                                     int, int, int, double *, int, int *, int *, double *, double *,
                                     double *, int, int, double *, int, int, double *, int,
                                     double *, int);

} // namespace calaman
