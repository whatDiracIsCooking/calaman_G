/**
 * @file shifted_cocg_bridge.h
 * @brief Per-pair state and kernel-launch declarations shared between
 *        calaman.shifted_cocg and its device-compiled translation unit
 *
 * Included by shifted_cocg.cppm in its GLOBAL MODULE FRAGMENT and by
 * shifted_cocg.cu directly, so the declarations bind to the definitions in the
 * .cu (the feast_bridge.h split). Real types only: a host GMF cannot import
 * wwr.complex, so every complex scalar is a (re, im) pair of arrays.
 *
 * A "pair" is one (shift e, column j), indexed p = e * k + j. Blocks of n x k
 * (V, X_e, P_e) are column-major with leading dimension n; the ne blocks of
 * X and P sit n * k elements apart. done_step[p] is -1 while the pair iterates
 * and the step it converged at after (0: a zero right-hand side).
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Device pointers to the per-pair recurrence state, each ne * k long.
///
/// eta_m and zeta_m are the D-Lanczos pivot and solution coefficient of
/// (z_e I - A) x = b_j at step m; ieta = 1 / eta; res the relative residual estimate.
template<typename T>
struct CocgPairs {
  T *eta_r, *eta_i;
  T *zeta_r, *zeta_i;
  T *ieta_r, *ieta_i;
  T *res;
  int *done_step;
};

/// @brief bnorm_j = ||b_j||_2, v_cur = B / bnorm (a zero column stays zero),
///        v_prev = 0, beta_cur = 0, and every pair's state reset for step 1.
template<typename T>
void cocg_start(wwr::wwrStream_t stream, int n, int k, int ne, const T *d_B, T *d_v_prev,
                T *d_v_cur, T *d_beta_cur, T *d_bnorm, CocgPairs<T> pairs);

/// @brief One Lanczos step per column, with @p d_w = A v_cur on entry:
///        w -= beta_cur v_prev, alpha = v_cur^T w, w -= alpha v_cur,
///        beta_next = ||w||_2, w /= beta_next.
///
/// beta_next <= eps (|alpha| + beta_cur) is an invariant subspace: it is
/// written as 0 and w zeroed. A NaN beta_next is kept, to surface in the pairs.
template<typename T>
void cocg_lanczos(wwr::wwrStream_t stream, int n, int k, const T *d_v_prev, const T *d_v_cur,
                  T *d_w, const T *d_beta_cur, T *d_alpha, T *d_beta_next);

/// @brief Advance every iterating pair's eta, zeta and residual estimate to
///        step @p step (1-based), marking done_step = step where res <= @p tol.
template<typename T>
void cocg_shift(wwr::wwrStream_t stream, int k, int ne, int step, T tol, const T *d_zr,
                const T *d_zi, const T *d_alpha, const T *d_beta_cur, const T *d_beta_next,
                const T *d_bnorm, CocgPairs<T> pairs);

/// @brief P_e = (v_cur + beta_cur P_e) / eta, X_e += zeta P_e, for each pair
///        iterating at, or converged at, step @p step.
template<typename T>
void cocg_update(wwr::wwrStream_t stream, int n, int k, int ne, int step, const T *d_v_cur,
                 const T *d_beta_cur, CocgPairs<T> pairs, T *d_pr, T *d_pi, T *d_xr, T *d_xi);

/// @brief cocg_update with X_e folded into one real block: P_e as there, then
///        out(:, j) += sum_e Re[ c_ej zeta P_e(:, j) ], c_ej = (d_cr + i d_ci)[e * k + j].
template<typename T>
void cocg_update_accumulate(wwr::wwrStream_t stream, int n, int k, int ne, int step,
                            const T *d_v_cur, const T *d_beta_cur, CocgPairs<T> pairs, T *d_pr,
                            T *d_pi, const T *d_cr, const T *d_ci, T *d_out);

} // namespace calaman::device
