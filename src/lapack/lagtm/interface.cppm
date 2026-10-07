/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lagtm -- tridiagonal matrix-matrix
 *        multiply, LAPACK's ?lagtm
 *
 * B := alpha * op(A) * X + beta * B, where A is the n-by-n general tridiagonal
 * with sub-diagonal dl, diagonal d and super-diagonal du, op is selected by a
 * Trans (N, T, or C -- C conjugates for a complex T), and X, B are n-by-nrhs
 * column-major. Enqueued on the stream and returns WITHOUT synchronizing.
 * Allocates nothing: one launch (lagtm.cu), no workspace.
 *
 * Mapping from DLAGTM (docs/architecture.md §4): TRANS becomes Trans; s/d/c/z
 * become one template over T, with alpha and beta real (ComplexToRealType<T>)
 * as in the reference; LDX/LDB are kept.
 * Divergence -- alpha and beta are GENERAL real scalars. DLAGTM honours only
 * alpha in {-1, 0, 1} (anything else is taken as 0) and beta in {-1, 0, 1}
 * (anything else is taken as 1); on those admissible values the two agree,
 * including beta == 0 overwriting B without reading it and alpha == 0 never
 * reading A or X. Outside them this routine computes the formula, not
 * DLAGTM's substitution.
 *
 * Usage:
 *   import calaman.lagtm;  // also re-exports Trans and Status
 *   calaman::lagtm<double>(stream, calaman::Trans::N, n, nrhs, 1.0, d_dl, d_d, d_du,
 *                          d_x, ldx, 0.0, d_b, ldb);
 */

module;

// CLM_TRY's header is not needed: the body has one fallible step, returned
// directly. lagtm_bridge.h declares the .cu launcher in the GMF.
#include "lagtm_bridge.h"

export module calaman.lagtm;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // Trans (:enums), ComplexToRealType

// export import: lagtm RETURNS calaman::Status, so a consumer must see its
// members, not just its name -- the re-export langt does.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lagtm;` alone names calaman::Trans.
export using calaman::Trans;

/// @brief B := alpha * op(A) * X + beta * B for the tridiagonal (dl, d, du) (?lagtm)
///
/// Returns without synchronizing; enqueues nothing when @p n or @p nrhs is 0,
/// or when alpha == 0 and beta == 1. The return carries a launch failure.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the work is enqueued on; all pointers live on its device
/// @param trans  op(A): A (N), A^T (T) or A^H (C; same as T for a real T)
/// @param n      Order of A and row count of X and B
/// @param nrhs   Column count of X and B
/// @param alpha  Real scalar on op(A) * X; any value (see the file header)
/// @param d_dl   Device sub-diagonal, length n-1 (unread when n <= 1)
/// @param d_d    Device diagonal, length n
/// @param d_du   Device super-diagonal, length n-1 (unread when n <= 1)
/// @param d_x    Device X, column-major, leading dimension @p ldx >= n
/// @param ldx    Leading dimension of X
/// @param beta   Real scalar on B; beta == 0 overwrites B without reading it
/// @param d_b    Device B, column-major, leading dimension @p ldb >= n; updated
/// @param ldb    Leading dimension of B
/// @return Success, or the launch error
export template<typename T>
Status lagtm(const wwr::wwrStream_t stream, const Trans trans, const std::size_t n,
             const std::size_t nrhs, const ComplexToRealType<T> alpha, const T *const d_dl,
             const T *const d_d, const T *const d_du, const T *const d_x, const std::size_t ldx,
             const ComplexToRealType<T> beta, T *const d_b, const std::size_t ldb) {
  using R = ComplexToRealType<T>;
  if (n == 0 || nrhs == 0 || (alpha == R{0} && beta == R{1})) {
    return wwr::wwrSuccess;
  }
  device::lagtm<T, R>(stream, trans, n, nrhs, alpha, d_dl, d_d, d_du, d_x, ldx, beta, d_b, ldb);
  return wwr::wwrGetLastError();
}

extern template Status lagtm<float>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, float,
                                    const float *, const float *, const float *, const float *,
                                    std::size_t, float, float *, std::size_t);
extern template Status lagtm<double>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, double,
                                     const double *, const double *, const double *,
                                     const double *, std::size_t, double, double *, std::size_t);
extern template Status lagtm<wwr::wwrFloatComplex>(
    wwr::wwrStream_t, Trans, std::size_t, std::size_t, float, const wwr::wwrFloatComplex *,
    const wwr::wwrFloatComplex *, const wwr::wwrFloatComplex *, const wwr::wwrFloatComplex *,
    std::size_t, float, wwr::wwrFloatComplex *, std::size_t);
extern template Status lagtm<wwr::wwrDoubleComplex>(
    wwr::wwrStream_t, Trans, std::size_t, std::size_t, double, const wwr::wwrDoubleComplex *,
    const wwr::wwrDoubleComplex *, const wwr::wwrDoubleComplex *, const wwr::wwrDoubleComplex *,
    std::size_t, double, wwr::wwrDoubleComplex *, std::size_t);

} // namespace calaman
