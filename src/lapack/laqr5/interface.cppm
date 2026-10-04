/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laqr5 -- one non-accumulated multishift
 *        QR sweep with bulge chasing, LAPACK's ?laqr5 (KACC22 = 0)
 *
 * One device routine: chase chains of @p nshfts / 2 double-implicit-shift bulges
 * through the isolated diagonal block in rows/columns @p ktop..@p kbot (1-based)
 * of the upper-Hessenberg @p h, driving the multishift QR iteration ?laqr0 is
 * built on. The shifts @p sr + i*@p si are reordered in place into real and
 * complex-conjugate pairs; @p h becomes the full Schur factor when @p wantt; the
 * sweep is accumulated into @p z from the right over rows @p iloz..@p ihiz when
 * @p wantz. Enqueued on the stream and returns WITHOUT synchronizing, like a
 * BLAS call; the caller synchronizes when it needs the outputs. @p h, @p z,
 * @p sr, @p si are device memory the caller owns; nothing is allocated here.
 *
 * A stream, not a device handle, is the whole requirement: the sweep is one
 * single-thread kernel that allocates nothing, matching calaman.laexc. The
 * per-reflector arithmetic (?laqr1's shift polynomial, the order-2/3 ?larfg
 * reflectors) is O(1) and each reflector is applied to the O(N) affected
 * rows/columns of @p h (and @p z) in that same thread -- no host round-trip.
 *
 * KACC22 = 0 ONLY: the reference's accumulated far-from-diagonal update (KACC22
 * >= 1, its U / WV / WH workspace and DGEMM block multiplies) is not exposed;
 * the non-accumulated sweep is complete on its own, and the oracle is pinned to
 * KACC22 = 0 so device and reference share the arithmetic path. REAL ONLY:
 * float / double, constrained by wwr::real_fp -- the scope ?slaqr5 / ?dlaqr5 has.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.laqr5;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_h: N x N Hessenberg; d_z: N x N accumulator; d_sr/d_si: NSHFTS shifts
 *   calaman::laqr5<double>(stream, true, true, n, ktop, kbot, nshfts,
 *                          d_sr, d_si, d_h, ldh, iloz, ihiz, d_z, ldz);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "laqr5_bridge.h"

export module calaman.laqr5;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.wrappers.common; // real_fp

// export import, not a plain import: laqr5 RETURNS calaman::Status, so a consumer
// of `import calaman.laqr5;` must see Status's member functions, not just its
// name -- the same re-export laexc / lanv2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief One non-accumulated multishift QR sweep (LAPACK ?laqr5, KACC22 = 0)
///        on @p stream
///
/// Enqueues the bulge-chasing sweep and returns without synchronizing. Chases
/// @p nshfts / 2 double-shift bulges through rows/columns @p ktop..@p kbot
/// (1-based) of the Hessenberg @p h, reordering @p sr / @p si in place into
/// real/complex-conjugate pairs, forming the full Schur factor in @p h when
/// @p wantt, and accumulating the sweep into @p z (rows @p iloz..@p ihiz) from
/// the right when @p wantz. A @p nshfts < 2 or @p ktop >= @p kbot is a no-op,
/// as in the reference. No argument checking beyond those quick returns. All
/// matrices are column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param wantt Form the full quasi-triangular Schur factor in @p h when true
/// @param wantz Accumulate the sweep into @p z from the right when true
/// @param n Order of @p h (and @p z when @p wantz)
/// @param ktop 1-based first row/column of the isolated diagonal block
/// @param kbot 1-based last row/column of the isolated diagonal block
/// @param nshfts Number of shifts; positive and even (an odd value drops one)
/// @param sr Device length-@p nshfts real parts of the shifts; reordered in place
/// @param si Device length-@p nshfts imaginary parts of the shifts; reordered in place
/// @param h Device N-by-N Hessenberg matrix, leading dimension @p ldh; updated in place
/// @param ldh Leading dimension of @p h
/// @param iloz First row of @p z the sweep touches (used only when @p wantz)
/// @param ihiz Last row of @p z the sweep touches (used only when @p wantz)
/// @param z Device N-by-N accumulator, leading dimension @p ldz; touched only when @p wantz
/// @param ldz Leading dimension of @p z
/// @return Success, or the runtime error the kernel launch reported
export template<wwr::real_fp T>
Status laqr5(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
             const int ktop, const int kbot, const int nshfts, T *const sr, T *const si, T *const h,
             const int ldh, const int iloz, const int ihiz, T *const z, const int ldz) {
  device::laqr5(stream, wantt, wantz, n, ktop, kbot, nshfts, sr, si, h, ldh, iloz, ihiz, z, ldz);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laexc does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laqr5<float>(wwr::wwrStream_t, bool, bool, int, int, int, int, float *,
                                    float *, float *, int, int, int, float *, int);
extern template Status laqr5<double>(wwr::wwrStream_t, bool, bool, int, int, int, int, double *,
                                     double *, double *, int, int, int, double *, int);

} // namespace calaman
