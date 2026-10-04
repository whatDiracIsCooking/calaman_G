/**
 * @file larf1f.cppm
 * @brief The calaman.larf1f module -- apply a reflector with an implicit unit v(1)
 *
 * One host routine that applies the elementary (Householder) reflector
 * H = I - tau * v * v^T to a matrix C in place, exactly as calaman.larf, but with
 * v(1) IMPLICITLY 1: the slot v points at is neither read nor assumed-stored (it
 * typically holds the diagonal of a packed QR factor), and the stored tail
 * v(2:) begins at v + incv. This is LAPACK's ?larf1f, which ?larfg feeds directly
 * (its output reflector has that implicit unit head), so it is the form the
 * pivoted-QR panel calls.
 *
 * The algebra is calaman.larf's; only the unit head is special-cased. For side L
 * (C := H*C), with w = C^T*v: the tail contributes via gemv over C(2:m,:) and
 * ger over C(2:m,:), while v(1)=1 is folded in by two BLAS-1 calls on C's first
 * row -- an axpy into w (w += C(1,:)^T) and an axpy on C (C(1,:) -= tau*w^T).
 * Side R (C := C*H) is the transpose of this story over C's first column. The
 * split is forced, not chosen: v(1) is not stored, so it cannot ride the gemv,
 * and a 0-length tail gemv would early-return without writing w (reference BLAS).
 *
 * When the whole vector IS the implicit 1 (m == 1 for side L, n == 1 for side R)
 * H = (1 - tau) and the update collapses to a single scal on that row/column --
 * also the branch that avoids the degenerate 0-length tail. The empty case
 * (m == 0 or n == 0) enqueues nothing, and tau == 0 is a true no-op on C.
 *
 * w is a caller-provided device workspace (length n for side L, m for side R);
 * the routine allocates nothing. Requires the handle's DEFAULT (host) pointer
 * mode -- all scalars are host addresses, as in calaman.larf. Templated over
 * float and double.
 *
 * Usage:
 *   import calaman.larf1f;   // names calaman::larf1f, calaman::Side
 *   import wwr.blas;         // wwrblasHandle_t, wwrblasCreate, ...
 *   // d_C: m x n device matrix, ldc; d_v: reflector (v(1) ignored, tail at d_v+1);
 *   // d_w: workspace length n
 *   calaman::larf1f<double>(handle, calaman::Side::L, m, n, d_v, 1, tau, d_C, ldc,
 *                           d_w);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.larf1f;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_OP_*, WWRBLAS_STATUS_*
import wwr.wrappers.blas; // gemv, ger, axpy, scal
import calaman.common;    // kZero<T>, kOne<T> (:constants), Side (:enums)

// export import, not a plain import: larf1f RETURNS calaman::Status, so a consumer
// of `import calaman.larf1f;` must see Status's member functions, not just its
// name -- the same re-export calaman.larf does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Side (L / R) lives in calaman.common's :enums partition. Re-export it so
// `import calaman.larf1f;` alone names calaman::Side, as the usage example relies
// on -- the same re-export calaman.larf does.
export using calaman::Side;

/// @brief Apply H = I - tau*v*v^T to C in place, with v(1) an implicit 1
///
/// side L computes C := H*C, side R computes C := C*H. v(1) is treated as 1 and
/// never read; the stored tail v(2:) starts at @p v + @p incv. Short-circuits: a
/// failed BLAS call returns its status and skips the rest. Does nothing and
/// returns success when @p m or @p n is 0; tau == 0 is a no-op on C.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; v, w and C live on its device
/// @param side Side::L for C := H*C, Side::R for C := C*H
/// @param m Row count of C
/// @param n Column count of C
/// @param v Reflector device vector with an implicit unit head; v(1) is ignored
///          and the tail v(2:) (length m-1 for L, n-1 for R) begins at @p v + @p incv
/// @param incv Stride between elements of v
/// @param tau The reflector scalar; tau == 0 leaves C unchanged
/// @param C Device matrix, m by n, column-major, updated in place
/// @param ldc Leading dimension of C (>= m)
/// @param w Device workspace vector, length n (side L) or m (side R), stride 1;
///          overwritten with the intermediate C^T*v / C*v
/// @return The first failing BLAS Status, otherwise the final update's Status
export template<typename T>
Status larf1f(wwr::wwrblasHandle_t handle, const Side side, const int m, const int n, const T *v,
              const int incv, const T tau, T *C, const int ldc, T *w) {
  if (m == 0 || n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  const T neg_tau = -tau;
  const T one_minus_tau = kOne<T> - tau;

  if (side == Side::L) {
    // The whole reflector is the implicit 1: H = (1 - tau), scale C's one row.
    if (m == 1) {
      return wwr::scal<T>(handle, n, &one_minus_tau, C, ldc);
    }
    // Tail: w := C(2:m,:)^T * v(2:m). v(2:m) is the stored head at v + incv.
    CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, m - 1, n, &kOne<T>, C + 1, ldc, v + incv, incv,
                         &kZero<T>, w, 1));
    // v(1) = 1 folds in by BLAS-1 on C's first row: w += C(1,:)^T, then
    // C(1,:) -= tau * w^T.
    CLM_TRY(wwr::axpy<T>(handle, n, &kOne<T>, C, ldc, w, 1));
    CLM_TRY(wwr::axpy<T>(handle, n, &neg_tau, w, 1, C, ldc));
    // Tail: C(2:m,:) -= tau * v(2:m) * w^T.
    return wwr::ger<T>(handle, m - 1, n, &neg_tau, v + incv, incv, w, 1, C + 1, ldc);
  }

  // side R, the transpose story over C's first column.
  if (n == 1) {
    return wwr::scal<T>(handle, m, &one_minus_tau, C, 1);
  }
  // Tail: w := C(:,2:n) * v(2:n). C(:,2:n) starts one column in, at C + ldc.
  CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, m, n - 1, &kOne<T>, C + ldc, ldc, v + incv, incv,
                       &kZero<T>, w, 1));
  // v(1) = 1 folds in by BLAS-1 on C's first column: w += C(:,1), then
  // C(:,1) -= tau * w.
  CLM_TRY(wwr::axpy<T>(handle, m, &kOne<T>, C, 1, w, 1));
  CLM_TRY(wwr::axpy<T>(handle, m, &neg_tau, w, 1, C, 1));
  // Tail: C(:,2:n) -= tau * w * v(2:n)^T.
  return wwr::ger<T>(handle, m, n - 1, &neg_tau, w, 1, v + incv, incv, C + ldc, ldc);
}

} // namespace calaman
