/**
 * @file lanczos_bridge.h
 * @brief Device status block and kernel-launch declarations shared between
 *        calaman.lanczos and its device-compiled translation unit (lanczos.cu)
 *
 * Included by the module partitions in their GLOBAL MODULE FRAGMENT and by the
 * .cu library directly, so the declarations bind to the definitions compiled in
 * the plain .cu translation unit (the feast_bridge.h split).
 *
 * The projected matrix T is dense ncv x ncv, column-major, ld ncv. Every launcher
 * reads its scalars (alpha, beta) from device memory: the step loop runs in
 * device pointer mode and makes no host round-trip per step. Between them, the
 * launchers write every entry of T a cycle uses, so T needs no memset: steps
 * 0..ncv-1 cover a fresh cycle, and after a restart the arrowhead covers
 * columns/rows 0..k-1 and steps k..ncv-1 the rest.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief What one Lanczos cycle leaves on the device for the host to read back.
///
/// One device-to-host copy of this block per cycle is the loop's only sync.
/// Reset it with lanczos_status_reset before each cycle.
struct LanczosStatus {
  int breakdown;      ///< nonzero once a step's beta fell under the breakdown guard
  int breakdown_step; ///< FIRST 0-based step j whose beta_j tripped the guard; -1 if none
  int eig_info;       ///< syevd devInfo for the projected problem
};

/// @brief Reset @p d_status to {breakdown 0, breakdown_step -1, eig_info 0}.
void lanczos_status_reset(wwr::wwrStream_t stream, LanczosStatus *d_status);

/// @brief Close step @p j: T(j,j) = alpha_j, T(j+1,j) = T(j,j+1) = beta_j, v_{j+1} = w / beta_j.
///
/// Also zeros column/row j past the band. Breakdown when NOT beta_j > eps * ||T_j||_F
/// (the leading (j+1)^2 block; NaN trips): the coupling is written as 0, the status
/// records it, and @p d_v_next (w, n elements) is left undivided, as after any earlier trip.
template<typename T>
void lanczos_step(wwr::wwrStream_t stream, int n, int ncv, int j, const T *d_alpha,
                  const T *d_beta, T *d_v_next, T *d_T, LanczosStatus *d_status);

/// @brief Write the thick-restart arrowhead: columns and rows 0..k-1 of T in full.
///
/// T(i,i) = theta_i and T(k,i) = T(i,k) = beta_m * s_{m,i} (= @p d_s_last_row[i * inc])
/// for i < k, zeros elsewhere in them; (k, k) is left for the next alpha. The kept
/// pairs must be compacted to the leading k slots. No-op unless 0 < k < ncv.
template<typename T>
void lanczos_arrowhead(wwr::wwrStream_t stream, int ncv, int k, const T *d_theta,
                       const T *d_s_last_row, int inc, const T *d_beta_m, T *d_T);

} // namespace calaman::device
