/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lahqr -- double-shift Francis QR for the
 *        Schur form of a small upper Hessenberg matrix, LAPACK's ?lahqr
 *
 * One device routine: compute the real Schur form of the active window
 * @p ilo..@p ihi (1-based) of the N-by-N upper Hessenberg @p h by the double-
 * shift Francis QR iteration. On return the window of @p h is quasi-triangular
 * (the full Schur form when @p wantt, else only the eigenvalues are reliable),
 * @p wr / @p wi hold the N eigenvalues (@p wi > 0 then < 0 for a complex pair),
 * and -- when @p wantz -- the orthogonal transform is accumulated into rows
 * @p iloz..@p ihiz of @p z. @p info is a device int: 0 on success, or the index
 * of the row that failed to converge within the iteration budget. Enqueued on
 * the given stream and returns WITHOUT synchronizing, like a BLAS call; the
 * caller synchronizes when it needs the outputs. All matrices are device memory
 * the caller owns; nothing is allocated here (the reference takes no WORK).
 *
 * A stream, not a device handle, is the whole requirement: the iteration is one
 * kernel that allocates nothing, matching calaman.laexc. The whole computation
 * runs on the device in a single thread -- the deflation search, the Francis
 * double/exceptional shift, the bulge-chasing length-3 reflector sweep and the
 * ?lanv2 standardisation of each converged 2x2 block -- so @p info is a device
 * scalar, not a host one, and there is no host round-trip.
 *
 * REAL ONLY: LAPACK's double-shift QR is the real path (float / double),
 * constrained by calaman::real_fp -- the scope the reference has.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lahqr;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_h: N x N Hessenberg; d_z: N x N accumulator; d_wr/d_wi: length-N; d_info
 *   calaman::lahqr<double>(stream, true, true, n, 1, n, d_h, ldh,
 *                          d_wr, d_wi, 1, n, d_z, ldz, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "lahqr_bridge.h"

export module calaman.lahqr;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import calaman.common;  // real_fp

// export import, not a plain import: lahqr RETURNS calaman::Status, so a consumer
// of `import calaman.lahqr;` must see Status's member functions, not just its
// name -- the same re-export laexc / lanv2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Double-shift Francis QR for the Schur form of an upper Hessenberg
///        matrix (LAPACK ?lahqr) on @p stream
///
/// Enqueues the QR iteration on the active window @p ilo..@p ihi (1-based) of the
/// Hessenberg @p h and returns without synchronizing. Writes the Schur form into
/// @p h (full when @p wantt), the eigenvalues into @p wr / @p wi, accumulates the
/// orthogonal transform into rows @p iloz..@p ihiz of @p z when @p wantz, and
/// writes @p info on the device: 0 on success, or the index that failed to
/// converge. A zero @p n writes nothing; @p ilo == @p ihi stores the single
/// eigenvalue. No argument checking beyond those quick returns, like the
/// reference. All matrices are column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param wantt Compute the full Schur form of @p h when true, else eigenvalues only
/// @param wantz Accumulate the orthogonal transform into @p z when true
/// @param n Order of @p h (and @p z when @p wantz)
/// @param ilo 1-based first index of the active window
/// @param ihi 1-based last index of the active window
/// @param h Device N-by-N upper Hessenberg matrix, leading dimension @p ldh; updated in place
/// @param ldh Leading dimension of @p h
/// @param wr Device length-N real parts of the eigenvalues
/// @param wi Device length-N imaginary parts of the eigenvalues
/// @param iloz 1-based first row of @p z the transform touches
/// @param ihiz 1-based last row of @p z the transform touches
/// @param z Device N-by-N accumulator, leading dimension @p ldz; touched only when @p wantz
/// @param ldz Leading dimension of @p z
/// @param info Device int; 0 on success, or the index that failed to converge
/// @return Success, or the runtime error the kernel launch reported
export template<calaman::real_fp T>
Status lahqr(const wwr::wwrStream_t stream, const bool wantt, const bool wantz, const int n,
             const int ilo, const int ihi, T *const h, const int ldh, T *const wr, T *const wi,
             const int iloz, const int ihiz, T *const z, const int ldz, int *const info) {
  device::lahqr(stream, wantt, wantz, n, ilo, ihi, h, ldh, wr, wi, iloz, ihiz, z, ldz, info);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laexc does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lahqr<float>(wwr::wwrStream_t, bool, bool, int, int, int, float *, int,
                                    float *, float *, int, int, float *, int, int *);
extern template Status lahqr<double>(wwr::wwrStream_t, bool, bool, int, int, int, double *, int,
                                     double *, double *, int, int, double *, int, int *);

} // namespace calaman
