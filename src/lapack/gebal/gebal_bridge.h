/**
 * @file gebal_bridge.h
 * @brief Device-launcher declarations shared between calaman.gebal's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by gebal.cu
 * directly -- the same split lacpy_bridge.h / laqp2_bridge.h use: the
 * declarations live in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop them binding to the definitions compiled in the
 * plain .cu translation unit.
 *
 * Unlike the geqp3 call graph, gebal is NOT a BLAS composition: its two stages
 * are hand-written kernels (a full-grid permutation scan and a single
 * cooperating-block Parlett-Reinsch sweep). The host DRIVER -- which branches on
 * device data every step and so must read it back -- lives in interface.cppm,
 * where it can `import wwr.runtime_api` for the memcpy/memset/stream-sync calls.
 * Each kernel is reached through one launcher declared here; the host checks
 * wwrGetLastError() after each, exactly as the reference driver did.
 *
 * COMPLEX TYPES DO NOT APPEAR HERE. This header is parsed in two contexts that
 * have no common complex builder -- wwr.complex is a module (interface.cppm's
 * GMF cannot import) and complex.h's complex constructors are gated to a device
 * pass (the host GMF parse does not get them). So every launcher is generic in
 * the element type @c T and, where it needs the real scale/norm type, in a
 * second parameter @c R, which the caller spells as calaman::ComplexToRealType<T>.
 * The .cu names the concrete wwrFloatComplex / wwrDoubleComplex only in its
 * explicit instantiations, in device context.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the host driver passes
 * its stream straight through. Reading the backend define that header needs is
 * why the module links wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>
#include <limits>

namespace calaman::device {

/// @brief Safety cap on Parlett-Reinsch sweeps.
///
/// A sweep is a strict descent (every accepted update cuts c+r for that index by
/// at least 5%), so LAPACK runs the loop uncapped. The cap exists only so a
/// pathological input cannot spin the host loop forever.
inline constexpr int kGebalDefaultMaxSweeps = 1000;

/// @brief Device scratch, counted in ints, that the gebal driver needs for an
///        n-by-n matrix.
///
/// Layout: [0, n) row flags, [n, 2n) column flags, [2n] one scalar reused for
/// the permutation search result and for the sweep's "something changed" flag.
constexpr int gebal_workspace_ints(const int n) { return (n > 0) ? (2 * n + 1) : 1; }

/// @brief Over/underflow guards, straight from LAPACK's SLAMCH-derived constants.
///
/// SFMIN1 = safe_min / precision, where precision is eps*base (LAPACK's 'P'),
/// i.e. std::numeric_limits<R>::epsilon(). The guards keep every intermediate
/// inside [SFMIN2, SFMAX2], which is what makes the powers-of-two similarity
/// exact rather than merely accurate. @c R is the real scale/norm type.
template<typename R>
struct scale_limits {
  R sfmin1;
  R sfmax1;
  R sfmin2;
  R sfmax2;
};

/// @brief Build the scale guards for real type @c R from std::numeric_limits.
template<typename R>
scale_limits<R> make_scale_limits() {
  const R sfmin1 = std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon();
  const R sfmin2 = sfmin1 * R(2);
  return scale_limits<R>{sfmin1, R(1) / sfmin1, sfmin2, R(1) / sfmin2};
}

// ── permutation stage launchers ──────────────────────────────────────────────

/// @brief Set @p scale[0 .. n) to real 1 -- the identity similarity the
///        permutation stage overwrites only where it isolates an eigenvalue.
///
/// @tparam R Real scale type (float or double); call with ComplexToRealType<T>.
template<typename R>
void gebal_fill_ones(wwr::wwrStream_t stream, int n, R *scale);

/// @brief Flag every row and column of the window [r0,r1]x[c0,c1] that owns an
///        off-diagonal non-zero.
///
/// A row left unflagged holds zeros everywhere in the window except its own
/// diagonal -- LAPACK's test for a row that isolates an eigenvalue; the same
/// pass answers the question for columns. Both flag arrays must be pre-zeroed.
///
/// @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex).
template<typename T>
void gebal_mark_nonzero(wwr::wwrStream_t stream, const T *A, int lda, int r0, int r1, int c0,
                        int c1, int *row_flag, int *col_flag);

/// @brief Write the scalar @p v into the single int at @p p (a one-thread kernel).
///
/// Used to seed the permutation search sentinel (-1 for the max scan, n for the
/// min scan); a memset cannot write an arbitrary int value.
void gebal_set_int(wwr::wwrStream_t stream, int *p, int v);

/// @brief Largest unflagged index in [lo, hi]; leaves @p out alone when none.
void gebal_pick_max_unflagged(wwr::wwrStream_t stream, const int *flag, int lo, int hi, int *out);

/// @brief Smallest unflagged index in [lo, hi]; leaves @p out alone when none.
void gebal_pick_min_unflagged(wwr::wwrStream_t stream, const int *flag, int lo, int hi, int *out);

/// @brief Record the transposition the way LAPACK does: scale[m] = j_one_based.
///
/// @tparam R Real scale type; call with ComplexToRealType<T>.
template<typename R>
void gebal_record_perm(wwr::wwrStream_t stream, R *scale, int m, int j_one_based);

/// @brief Swap columns @p j and @p m of A over all @p n rows.
///
/// LAPACK swaps only the leading L entries; the rows below L are already
/// isolated and hold zeros in every window column, so swapping those zeros too
/// is a no-op and the full-length swap is equivalent (and bound-free).
///
/// @tparam T Element type.
template<typename T>
void gebal_swap_cols(wwr::wwrStream_t stream, T *A, int lda, int n, int j, int m);

/// @brief Swap rows @p j and @p m of A over all @p n columns. Same equivalence.
///
/// @tparam T Element type.
template<typename T>
void gebal_swap_rows(wwr::wwrStream_t stream, T *A, int lda, int n, int j, int m);

// ── scaling stage launcher ───────────────────────────────────────────────────

/// @brief Run one full Parlett-Reinsch sweep over indices k0..l0 in a single
///        cooperating block.
///
/// The sweep is Gauss-Seidel: index i is scaled before index i+1 measures its
/// norms, as in LAPACK -- which is why the whole sweep runs in one block
/// (__syncthreads() is the only barrier cheap enough to sit between consecutive
/// indices). Magnitudes use the off-diagonal 1-norm over the window; complex
/// uses CABS1. @p noconv is raised to 1 if any index was scaled, so the host
/// knows to issue another sweep. @p scale is updated with the accepted factors.
///
/// @tparam T Element type.
/// @tparam R Real scale/norm type; call with ComplexToRealType<T>.
template<typename T, typename R>
void gebal_sweep(wwr::wwrStream_t stream, int n, T *A, int lda, int k0, int l0, R *scale,
                 int *noconv, scale_limits<R> lim);

} // namespace calaman::device
