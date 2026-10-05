/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laexc -- swap two adjacent diagonal
 *        blocks of a real Schur form by an orthogonal similarity, LAPACK's ?laexc
 *
 * One device routine: in the N-by-N upper quasi-triangular matrix @p t (a real
 * Schur form) swap the @p n1-by-@p n1 diagonal block at @p j1 (1-based) with the
 * adjacent @p n2-by-@p n2 block (n1, n2 in {1, 2}) by an orthogonal similarity,
 * and -- when @p wantq -- accumulate that transform into @p q. @p info is a
 * device int: 0 on a completed swap, 1 when the swap was rejected (the blocks sit
 * too close to each other to swap stably). Enqueued on the given stream and
 * returns WITHOUT synchronizing, like a BLAS call; the caller synchronizes when
 * it needs the outputs. @p t and @p q are device matrices the caller owns;
 * nothing is allocated here (the reference's WORK array is on-kernel scratch).
 *
 * A stream, not a device handle, is the whole requirement: the swap is one kernel
 * that allocates nothing, matching calaman.lasy2. The whole computation runs on
 * the device in a single thread: the block arithmetic (the Sylvester solve, the
 * length-3 reflectors, the ?lanv2 re-standardisation) is O(1), and the resulting
 * rotations/reflectors are applied to the O(N) affected rows/columns of @p t
 * (and @p q) in that same thread -- no host round-trip, so @p info is a device
 * scalar, not a host one.
 *
 * REAL ONLY: a complex Schur form is already triangular with no 2x2 blocks to
 * swap, so LAPACK ships no complex ?laexc; the surface is float / double,
 * constrained by calaman::real_fp -- the scope the reference has.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.laexc;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_t: N x N Schur form; d_q: N x N accumulator; d_info: device int
 *   calaman::laexc<double>(stream, true, n, d_t, ldt, d_q, ldq, j1, n1, n2, d_info);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "laexc_bridge.h"

export module calaman.laexc;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import calaman.common;  // real_fp

// export import, not a plain import: laexc RETURNS calaman::Status, so a consumer
// of `import calaman.laexc;` must see Status's member functions, not just its
// name -- the same re-export lasy2 / lanv2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Swap two adjacent diagonal blocks of a real Schur form (LAPACK ?laexc)
///        on @p stream
///
/// Enqueues the orthogonal-similarity swap and returns without synchronizing.
/// Swaps the @p n1-by-@p n1 block at @p j1 (1-based) with the adjacent
/// @p n2-by-@p n2 block, updates @p t in place, accumulates the transform into
/// @p q when @p wantq, and writes @p info on the device: 0 on success, 1 when the
/// swap was rejected (the blocks are too close to swap stably, and @p t / @p q
/// are left unchanged). A zero @p n, @p n1 or @p n2, or @p j1 + @p n1 > @p n,
/// writes only *info = 0. No argument checking beyond those quick returns, like
/// the reference. All matrices are column-major with the given leading dimensions.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; every pointer lives on its device
/// @param wantq Accumulate the orthogonal transform into @p q when true
/// @param n Order of @p t (and @p q when @p wantq)
/// @param t Device N-by-N Schur form, leading dimension @p ldt; updated in place
/// @param ldt Leading dimension of @p t
/// @param q Device N-by-N accumulator, leading dimension @p ldq; touched only when @p wantq
/// @param ldq Leading dimension of @p q
/// @param j1 1-based index of the first row/column of the first block
/// @param n1 Order of the first block (1 or 2)
/// @param n2 Order of the second block (1 or 2)
/// @param info Device int; 0 on success, 1 if the swap was rejected
/// @return Success, or the runtime error the kernel launch reported
export template<calaman::real_fp T>
Status laexc(const wwr::wwrStream_t stream, const bool wantq, const int n, T *const t,
             const int ldt, T *const q, const int ldq, const int j1, const int n1, const int n2,
             int *const info) {
  device::laexc(stream, wantq, n, t, ldt, q, ldq, j1, n1, n2, info);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as lasy2 does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laexc<float>(wwr::wwrStream_t, bool, int, float *, int, float *, int, int,
                                    int, int, int *);
extern template Status laexc<double>(wwr::wwrStream_t, bool, int, double *, int, double *, int, int,
                                     int, int, int *);

} // namespace calaman
