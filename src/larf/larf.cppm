/**
 * @file larf.cppm
 * @brief The calaman.larf module -- apply an elementary reflector
 *
 * One host routine that applies the elementary (Householder) reflector
 * H = I - tau * v * v^T to a matrix C, in place, for side L (C := H*C) or side R
 * (C := C*H). This is LAPACK's ?larf, composed from exactly two wrapped BLAS-2
 * calls and NO custom kernel -- the same pure-host pattern as calaman.diff_norm
 * (reaching the OUTERMOST WarpWraps layer, wwr.wrappers.blas, because gemv and ger
 * are stock BLAS with nothing backend-specific to write).
 *
 * The algebra, and why gemv's trans and ger's operand order differ by side:
 *
 *   side L:  C := H*C = C - tau * v * (v^T * C).  Let w = C^T * v (length n).
 *            Then C := C - tau * v * w^T.
 *     w = C^T * v : gemv with op(C) = C^T, i.e. trans = T (C is m x n, v length
 *                   m, w length n), alpha = 1, beta = 0.
 *     C -= tau v w^T : ger(m, n, alpha = -tau, x = v (len m), y = w (len n)).
 *
 *   side R:  C := C*H = C - tau * (C * v) * v^T.  Let w = C * v (length m).
 *            Then C := C - tau * w * v^T.
 *     w = C * v   : gemv with op(C) = C, i.e. trans = N (v length n, w length m),
 *                   alpha = 1, beta = 0.
 *     C -= tau w v^T : ger(m, n, alpha = -tau, x = w (len m), y = v (len n)).
 *
 * So the single difference is which of {v, w} plays gemv's input vector and ger's
 * left (column) operand: side L feeds v into the ger column and w into gemv's x
 * via trans = T; side R feeds w into the ger column and v into gemv's x via
 * trans = N. ger's (m, n, ldc) and the rank-1 structure A := alpha*x*y^T + A are
 * identical either way.
 *
 * w is a caller-provided device workspace (LAPACK's convention), length n for
 * side L and m for side R -- the routine allocates nothing, keeping the shipped
 * surface allocation-free like calaman.diff_norm and calaman.lacpy.
 *
 * tau == 0 makes H the identity: the gemv still runs (writing w) but ger's
 * alpha = 0 leaves C untouched, so the result is a true no-op on C, matching
 * LAPACK. The empty case (m == 0 or n == 0) enqueues nothing.
 *
 * Requires the handle's DEFAULT (host) pointer mode: alpha/beta for gemv are
 * &kOne<T>/&kZero<T> and ger's alpha is &neg_tau, all HOST addresses, exactly as
 * diff_norm passes &kNegativeOne<T>. v, w and C are device pointers the caller
 * owns.
 *
 * Templated over `float` and `double` (ger has no complex-unconjugated subtlety
 * to settle here; complex would pick geru/gerc and conj(v) deliberately later).
 *
 * Usage:
 *   import calaman.larf;     // names calaman::larf, calaman::Side
 *   import wwr.blas;         // wwrblasHandle_t, wwrblasCreate, ...
 *   // d_C: m x n device matrix, ldc; d_v: reflector; d_w: workspace length n
 *   calaman::larf<double>(handle, calaman::Side::L, m, n, d_v, 1, tau, d_C, ldc,
 *                         d_w);
 */

export module calaman.larf;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_OP_*, WWRBLAS_STATUS_*
import wwr.wrappers.blas; // gemv, ger
import calaman.common;    // kZero<T>, kOne<T> (:constants), Side (:enums)

namespace calaman {

// Side (L / R) lives in calaman.common's :enums partition. Re-export it so
// `import calaman.larf;` alone names calaman::Side, as the usage example relies
// on -- the same re-export diff_norm does for calaman::Norm. The downstream
// laqp2 module reaches Side through this re-export (`import calaman.larf;`).
export using calaman::Side;

/// @brief Apply the elementary reflector H = I - tau*v*v^T to C, in place
///
/// side L computes C := H*C, side R computes C := C*H, via one gemv (w := C^T*v
/// for L, C*v for R) and one ger (C := C - tau * outer product). Short-circuits:
/// if the gemv does not succeed its status is returned and the ger is not
/// enqueued. Does nothing and returns success when @p m or @p n is 0. tau == 0 is
/// a no-op on C (ger's alpha is 0), matching LAPACK.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; v, w and C live on its device
/// @param side Side::L for C := H*C, Side::R for C := C*H
/// @param m Row count of C
/// @param n Column count of C
/// @param v Reflector device vector, length m (side L) or n (side R), stride @p incv
/// @param incv Stride between elements of v
/// @param tau The reflector scalar; tau == 0 leaves C unchanged
/// @param C Device matrix, m by n, column-major, updated in place
/// @param ldc Leading dimension of C (>= m)
/// @param w Device workspace vector, length n (side L) or m (side R), stride 1;
///          overwritten with the intermediate C^T*v / C*v
/// @return The gemv status if it failed, otherwise the ger status
export template<typename T>
wwr::wwrblasStatus_t larf(wwr::wwrblasHandle_t handle, const Side side, const int m, const int n,
                          const T *v, const int incv, const T tau, T *C, const int ldc, T *w) {
  if (m == 0 || n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // gemv: w := op(C) * v. side L takes op = C^T (trans = T, w length n); side R
  // takes op = C (trans = N, w length m). m and n are always C's dimensions.
  const wwr::wwrblasOperation_t trans = side == Side::L ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;
  const auto gemv_status =
      wwr::gemv<T>(handle, trans, m, n, &kOne<T>, C, ldc, v, incv, &kZero<T>, w, 1);
  if (gemv_status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return gemv_status;
  }

  // ger: C := C - tau * (column)(row)^T. side L is v * w^T (v the column), side R
  // is w * v^T (w the column). alpha = -tau is a host scalar (host pointer mode).
  const T neg_tau = -tau;
  const T *ger_x = side == Side::L ? v : w;
  const int ger_incx = side == Side::L ? incv : 1;
  const T *ger_y = side == Side::L ? w : v;
  const int ger_incy = side == Side::L ? 1 : incv;
  return wwr::ger<T>(handle, m, n, &neg_tau, ger_x, ger_incx, ger_y, ger_incy, C, ldc);
}

} // namespace calaman
