/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lasy2 -- solve the order-(1..2) Sylvester
 *        equation op(TL)*X + ISGN*X*op(TR) = SCALE*B, LAPACK's ?lasy2
 *
 * One device routine: solve for the N1-by-N2 matrix X (1 <= N1,N2 <= 2), where
 * TL is N1xN1, TR is N2xN2, B is N1xN2 and ISGN is +/-1, with op(T) = T or T^T
 * per @p ltranl / @p ltranr. SCALE (<= 1) guards against overflow. Enqueued on
 * the given stream and returns WITHOUT synchronizing, like a BLAS call; the
 * caller synchronizes when it needs the outputs. @p tl, @p tr, @p b are device
 * inputs the caller owns; @p scale, @p x, @p xnorm and @p info are device
 * pointers this writes. Nothing is allocated here.
 *
 * A stream, not a device handle, is the whole requirement: the solve is one
 * kernel that allocates nothing, so it needs no device index and no memory pool
 * -- matching calaman.lacgv. The entire computation runs on the device (a single
 * thread) because the inputs are device-resident and X is at most 2x2; there is
 * no host round-trip, so the five outputs are device scalars, not host ones.
 *
 * REAL ONLY: LAPACK ships no complex ?lasy2, so the surface is float / double,
 * constrained by wwr::real_fp -- the same scope the reference has. There is no
 * bounds checking, matching the reference ("in the interests of speed, this
 * routine does not check the inputs"); @p info reports only the near-singular
 * perturbation (1) the solve applied, not an argument error.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lasy2;      // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_tl, d_tr, d_b device inputs; d_scale, d_x, d_xnorm, d_info device outputs
 *   calaman::lasy2<double>(stream, false, false, 1, 2, 2, d_tl, 2, d_tr, 2, d_b, 2,
 *                          d_scale, d_x, 2, d_xnorm, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include "lasy2_bridge.h"

export module calaman.lasy2;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.wrappers.common; // real_fp

// export import, not a plain import: lasy2 RETURNS calaman::Status, so a consumer
// of `import calaman.lasy2;` must see Status's member functions, not just its
// name -- the same re-export lacgv and diff_norm do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Solve op(TL)*X + ISGN*X*op(TR) = SCALE*B for X (LAPACK ?lasy2) on @p stream
///
/// Enqueues the order-(@p n1, @p n2) Sylvester solve and returns without
/// synchronizing. Writes @p scale, the @p n1-by-@p n2 solution @p x, its
/// infinity-norm @p xnorm, and @p info (0, or 1 when TL and TR had too-close
/// eigenvalues and a near-singular pivot was perturbed) -- all on the device. A
/// zero @p n1 or @p n2 writes only *info = 0. No argument checking, like the
/// reference. All matrices are column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param ltranl Use op(TL) = TL^T when true, TL otherwise
/// @param ltranr Use op(TR) = TR^T when true, TR otherwise
/// @param isgn Sign of the equation; +1 or -1
/// @param n1 Order of TL (0, 1 or 2)
/// @param n2 Order of TR (0, 1 or 2)
/// @param tl Device N1xN1 matrix TL, leading dimension @p ldtl
/// @param tr Device N2xN2 matrix TR, leading dimension @p ldtr
/// @param b Device N1xN2 right-hand side B, leading dimension @p ldb
/// @param scale Device scalar; the overflow-guarding scale factor is written here
/// @param x Device N1xN2 solution, leading dimension @p ldx
/// @param xnorm Device scalar; the infinity-norm of @p x is written here
/// @param info Device int; 0 on success, 1 if a near-singular pivot was perturbed
/// @return Success, or the runtime error the kernel launch reported
export template<wwr::real_fp T>
Status lasy2(const wwr::wwrStream_t stream, const bool ltranl, const bool ltranr, const int isgn,
             const int n1, const int n2, const T *const tl, const int ldtl, const T *const tr,
             const int ldtr, const T *const b, const int ldb, T *const scale, T *const x,
             const int ldx, T *const xnorm, int *const info) {
  device::lasy2(stream, ltranl, ltranr, isgn, n1, n2, tl, ldtl, tr, ldtr, b, ldb, scale, x, ldx,
                xnorm, info);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as lacgv does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lasy2<float>(wwr::wwrStream_t, bool, bool, int, int, int, const float *, int,
                                    const float *, int, const float *, int, float *, float *, int,
                                    float *, int *);
extern template Status lasy2<double>(wwr::wwrStream_t, bool, bool, int, int, int, const double *,
                                     int, const double *, int, const double *, int, double *,
                                     double *, int, double *, int *);

} // namespace calaman
