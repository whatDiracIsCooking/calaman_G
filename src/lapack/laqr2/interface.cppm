/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laqr2 -- aggressive early deflation for
 *        the real nonsymmetric eigensolver, LAPACK's ?laqr2 (small-window variant)
 *
 * One device routine: on the trailing JW = min(@p nw, @p kbot - @p ktop + 1)
 * window of the upper-Hessenberg @p h, run aggressive early deflation -- the
 * window Schur form (via ?lahqr, not the recursive ?laqr4 of ?laqr3), deflate the
 * eigenvalues whose spike coupling is negligible, reflect the survivors back to
 * Hessenberg form, and apply the accumulated window transform to @p h (and @p z
 * when @p wantz). The deflated/undeflated counts land in device ints @p nd / @p ns
 * and the window eigenvalues in @p sr / @p si (at the positions ?laqr2 documents).
 * Enqueued on the stream and returns WITHOUT synchronizing, like a BLAS call.
 * @p h, @p z, @p sr, @p si and the @p v / @p t scratch are device memory the
 * caller owns; nothing is allocated here.
 *
 * A stream, not a device handle, is the whole requirement: the pass is one
 * single-thread kernel that allocates nothing, matching calaman.laqr5. The window
 * work (the inlined ?lahqr, the ?trexc reorder over ?laexc swaps, the ?gehrd
 * re-Hessenberg and ?ormhr accumulate) is O(JW^3) and each slab update O(N*JW),
 * all in that one thread -- no host round-trip, so @p ns / @p nd are device ints.
 *
 * SMALL-WINDOW VARIANT (?laqr2, not ?laqr3): the window Schur form comes from
 * ?lahqr, so this routine does not recurse into itself. REAL ONLY: float /
 * double, the scope ?slaqr2 / ?dlaqr2 has. The reference's NH / NV slab-blocking
 * and WV workspace are accepted for signature parity and ignored (slab updates
 * run unblocked); NW must satisfy NW <= the kernel's on-stack window bound.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.laqr2;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_h: N x N Hessenberg; d_z: N x N accumulator; d_v/d_t: NW x NW scratch
 *   calaman::laqr2<double>(stream, true, true, n, ktop, kbot, nw, d_h, ldh, 1, n,
 *                          d_z, ldz, d_ns, d_nd, d_sr, d_si, d_v, ldv, nw, d_t,
 *                          ldt, nw, d_wv, ldwv);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "laqr2_bridge.h"

export module calaman.laqr2;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import calaman.common;  // real_fp

// export import, not a plain import: laqr2 RETURNS calaman::Status, so a consumer
// of `import calaman.laqr2;` must see Status's member functions, not just its
// name -- the same re-export laqr5 / lahqr do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief One aggressive-early-deflation pass (LAPACK ?laqr2, ?lahqr variant) on
///        @p stream
///
/// Enqueues the AED pass and returns without synchronizing. Diagonalises the
/// trailing JW = min(@p nw, @p kbot - @p ktop + 1) window of the Hessenberg @p h,
/// deflates the spike-negligible eigenvalues, reflects the survivors back to
/// Hessenberg form, and applies the window transform to @p h (fully when
/// @p wantt, else only enough to preserve eigenvalues) and to @p z (rows
/// @p iloz..@p ihiz) when @p wantz. The number of deflations lands in @p nd and
/// of shifts in @p ns (device ints); the window eigenvalues in @p sr / @p si.
/// A @p ktop > @p kbot or @p nw < 1 writes only *@p ns = *@p nd = 0, as in the
/// reference. No argument checking beyond those quick returns. All matrices are
/// column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
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
/// @param ns Device int; the number of shift (unconverged) eigenvalues returned
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
/// @return Success, or the runtime error the kernel launch reported
export template<calaman::real_fp T>
Status laqr2(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
             const int ktop, const int kbot, const int nw, T *const h, const int ldh,
             const int iloz, const int ihiz, T *const z, const int ldz, int *const ns,
             int *const nd, T *const sr, T *const si, T *const v, const int ldv, const int nh,
             T *const t, const int ldt, const int nv, T *const wv, const int ldwv) {
  device::laqr2(stream, wantt, wantz, n, ktop, kbot, nw, h, ldh, iloz, ihiz, z, ldz, ns, nd, sr, si,
                v, ldv, nh, t, ldt, nv, wv, ldwv);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laqr5 does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laqr2<float>(wwr::wwrStream_t, bool, bool, int, int, int, int, float *, int,
                                    int, int, float *, int, int *, int *, float *, float *, float *,
                                    int, int, float *, int, int, float *, int);
extern template Status laqr2<double>(wwr::wwrStream_t, bool, bool, int, int, int, int, double *,
                                     int, int, int, double *, int, int *, int *, double *, double *,
                                     double *, int, int, double *, int, int, double *, int);

} // namespace calaman
