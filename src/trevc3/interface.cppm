/**
 * @file interface.cppm
 * @brief Primary interface for calaman.trevc3 -- the left/right eigenvectors of
 *        a real quasi-triangular (Schur) matrix, LAPACK's ?trevc3 (HOWMNY = 'A')
 *
 * One device routine: back-substitute the eigenvectors of the upper
 * quasi-triangular @p t into @p vr (when @p want_right) and/or @p vl (when @p
 * want_left). Each eigenvalue -- real (1-by-1 block) or a complex-conjugate pair
 * (2-by-2 block) -- yields one column (real) or two (real then imaginary part),
 * in the diagonal order of @p t, each normalized to unit infinity norm. Enqueued
 * on the stream and returns WITHOUT synchronizing, like a BLAS call. @p t is a
 * device input; @p vl, @p vr and the length-3N scratch @p work are caller-owned
 * device memory this writes. Nothing is allocated here.
 *
 * HOWMNY = 'A' ONLY, eigenvectors of T directly (no OVER back-transform onto an
 * input Q, the blocked ?trevc3's DGEMM feature): SELECT is unused and M is N.
 * The per-block scaled solve is LAPACK's ?laln2, inlined in the .cu rather than
 * imported (calaman.laln2 is a launch functor, unusable from inside this kernel)
 * -- so, like calaman.laexc, the kernel reaches no sibling module. REAL ONLY:
 * float / double, constrained by wwr::real_fp -- the scope ?strevc3 / ?dtrevc3
 * has. `extern template` below pairs with instantiations.cpp, instantiating the
 * wrapper once here so an importer never re-instantiates a body naming the
 * .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.trevc3;    // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_t: N x N real Schur form; d_vr: N x N; d_work: length 3N scratch
 *   calaman::trevc3<double>(stream, false, true, n, d_t, ldt,
 *                           nullptr, 1, d_vr, ldvr, d_work);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "trevc3_bridge.h"

export module calaman.trevc3;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.wrappers.common; // real_fp

// export import, not a plain import: trevc3 RETURNS calaman::Status, so a
// consumer of `import calaman.trevc3;` must see Status's member functions, not
// just its name -- the same re-export laqr5 / laln2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Eigenvectors of a real quasi-triangular matrix (LAPACK ?trevc3,
///        HOWMNY = 'A') on @p stream
///
/// Enqueues the back-substitution and returns without synchronizing. Writes the
/// right eigenvectors into @p vr when @p want_right and the left eigenvectors
/// into @p vl when @p want_left -- one column per real eigenvalue, two (real
/// then imaginary part) per complex-conjugate pair, in the diagonal order of
/// @p t, each normalized so its largest-magnitude entry is 1. No argument
/// checking, like the reference; SELECT is unused (M == @p n). All matrices are
/// column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param want_left Compute the left eigenvectors into @p vl when true
/// @param want_right Compute the right eigenvectors into @p vr when true
/// @param n Order of @p t
/// @param t Device N-by-N real quasi-triangular (Schur) matrix, leading dimension @p ldt
/// @param ldt Leading dimension of @p t (>= n)
/// @param vl Device N-by-N left-eigenvector output; written only when @p want_left
/// @param ldvl Leading dimension of @p vl (>= n when @p want_left, else >= 1)
/// @param vr Device N-by-N right-eigenvector output; written only when @p want_right
/// @param ldvr Leading dimension of @p vr (>= n when @p want_right, else >= 1)
/// @param work Device length-3N scratch the back-substitution uses
/// @return Success, or the runtime error the kernel launch reported
export template<wwr::real_fp T>
Status trevc3(const wwr::wwrStream_t stream, const bool want_left, const bool want_right,
              const int n, const T *const t, const int ldt, T *const vl, const int ldvl,
              T *const vr, const int ldvr, T *const work) {
  device::trevc3(stream, want_left, want_right, n, t, ldt, vl, ldvl, vr, ldvr, work);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laqr5 does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status trevc3<float>(wwr::wwrStream_t, bool, bool, int, const float *, int, float *,
                                     int, float *, int, float *);
extern template Status trevc3<double>(wwr::wwrStream_t, bool, bool, int, const double *, int,
                                      double *, int, double *, int, double *);

} // namespace calaman
