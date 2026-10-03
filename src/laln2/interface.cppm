/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laln2 -- solve the scaled 1x1 or 2x2 real
 *        system (ca*op(A) - w*D) X = SCALE*B, LAPACK's ?laln2
 *
 * One device routine: solve for the NA-by-NA matrix X (NA is 1 or 2), where A is
 * NA-by-NA real, ca and D = diag(@p d1, @p d2) are real, and w = @p wr + i*@p wi
 * is real (@p nw == 1) or complex (@p nw == 2). When w is complex X and B are
 * NA-by-2 slabs (column 1 real, column 2 imaginary). op(A) is A or A^T per @p
 * ltrans. SCALE (<= 1) and a possible SMINI perturbation of C = ca*op(A) - w*D
 * guard against overflow and near-singularity. Enqueued on the given stream and
 * returns WITHOUT synchronizing, like a BLAS call; the caller synchronizes when
 * it needs the outputs. @p a, @p b are device inputs the caller owns; @p x, @p
 * scale, @p xnorm and @p info are device pointers this writes. Nothing is
 * allocated here.
 *
 * A stream, not a device handle, is the whole requirement: the solve is one
 * kernel that allocates nothing, so it needs no device index and no memory pool
 * -- matching calaman.lasy2. The entire computation runs on the device (a single
 * thread) because the inputs are device-resident and X is at most 2x2; there is
 * no host round-trip, so the four outputs are device scalars, not host ones. The
 * complex divisions go through calaman::ladiv_scalar (ladiv.h), called in the
 * kernel per the module's #86 dependency.
 *
 * REAL ELEMENT TYPE: the reference is SLALN2 / DLALN2 (w may be complex, but A
 * and the arithmetic are real), so the surface is float / double, constrained by
 * wwr::real_fp. There is no bounds checking, matching the reference; @p info
 * reports only the SMINI perturbation (1) the solve applied, not an argument
 * error.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include "laln2_bridge.h"

export module calaman.laln2;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.wrappers.common; // real_fp

// export import, not a plain import: laln2 RETURNS calaman::Status, so a consumer
// of `import calaman.laln2;` must see Status's member functions, not just its
// name -- the same re-export lasy2 and ladiv do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Solve (ca*op(A) - w*D) X = SCALE*B for X (LAPACK ?laln2) on @p stream
///
/// Enqueues the order-@p na solve and returns without synchronizing. Writes the
/// NA-by-NW solution @p x, the overflow-guarding @p scale, its infinity-norm @p
/// xnorm, and @p info (0, or 1 when C = ca*op(A) - w*D was perturbed to keep its
/// smallest singular value above @p smin) -- all on the device. No argument
/// checking, like the reference. All matrices are column-major with the given
/// leading dimensions; when @p nw is 2, @p b and @p x hold the real part in
/// column 1 and the imaginary part in column 2.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param ltrans Use op(A) = A^T when true, A otherwise
/// @param na Order of A (1 or 2)
/// @param nw 1 if w is real, 2 if w is complex
/// @param smin Desired lower bound on the singular values of C
/// @param ca Scalar multiplying A
/// @param a Device NAxNA matrix A, leading dimension @p lda
/// @param lda Leading dimension of @p a (>= na)
/// @param d1 The (1,1) element of the diagonal D
/// @param d2 The (2,2) element of the diagonal D (unused when na == 1)
/// @param b Device NAxNW right-hand side B, leading dimension @p ldb
/// @param ldb Leading dimension of @p b (>= na)
/// @param wr Real part of the scalar w
/// @param wi Imaginary part of w (unused when nw == 1)
/// @param x Device NAxNW solution, leading dimension @p ldx
/// @param ldx Leading dimension of @p x (>= na)
/// @param scale Device scalar; the overflow-guarding scale factor is written here
/// @param xnorm Device scalar; the infinity-norm of @p x is written here
/// @param info Device int; 0 on success, 1 if C was perturbed
/// @return Success, or the runtime error the kernel launch reported
export template<wwr::real_fp T>
Status laln2(const wwr::wwrStream_t stream, const bool ltrans, const int na, const int nw,
             const T smin, const T ca, const T *const a, const int lda, const T d1, const T d2,
             const T *const b, const int ldb, const T wr, const T wi, T *const x, const int ldx,
             T *const scale, T *const xnorm, int *const info) {
  device::laln2(stream, ltrans, na, nw, smin, ca, a, lda, d1, d2, b, ldb, wr, wi, x, ldx, scale,
                xnorm, info);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as lasy2 does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laln2<float>(wwr::wwrStream_t, bool, int, int, float, float, const float *,
                                    int, float, float, const float *, int, float, float, float *,
                                    int, float *, float *, int *);
extern template Status laln2<double>(wwr::wwrStream_t, bool, int, int, double, double,
                                     const double *, int, double, double, const double *, int,
                                     double, double, double *, int, double *, double *, int *);

} // namespace calaman
