/**
 * @file lanczos_bridge.h
 * @brief Device status block and kernel-launch declarations shared between
 *        calaman.lanczos and its device-compiled translation unit
 *
 * Included by the module partitions in their GLOBAL MODULE FRAGMENT and by the
 * .cu library directly, so the declarations bind to the definitions compiled in
 * the plain .cu translation unit (the feast_bridge.h split).
 *
 * The projected matrix T is dense ncv x ncv, column-major, ld ncv. Every launcher
 * reads its scalars (alpha, beta) from device memory: the step loop runs in
 * device pointer mode and makes no host round-trip per step.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief What one Lanczos cycle leaves on the device for the host to read back.
///
/// One device-to-host copy of this block per cycle is the loop's only sync.
struct LanczosStatus {
  int breakdown;      ///< nonzero once a step's beta fell under the breakdown guard
  int breakdown_step; ///< 0-based step j whose beta_j tripped the guard; -1 if none
  int eig_info;       ///< syevd devInfo for the projected problem
};

/// @brief Close step @p j: write alpha_j / beta_j into T, then v_{j+1} = w / beta_j.
///
/// @p d_v_next holds w on entry (n elements) and is normalised in place. If
/// beta_j <= eps * ||T|| (estimated from the alpha/beta history), sets the status
/// breakdown flag and step instead and leaves @p d_v_next undivided.
/// @tparam T float or double
template<typename T>
void lanczos_step(wwr::wwrStream_t stream, int n, int ncv, int j, const T *d_alpha,
                  const T *d_beta, T *d_v_next, T *d_T, LanczosStatus *d_status);

/// @brief Write the thick-restart arrowhead into the leading (k+1) x (k+1) block of T.
///
/// theta_i on the diagonal for i < k, beta_m * s_{m,i} in row and column k
/// (s_{m,i} = @p d_s_last_row[i * inc]), zero elsewhere in the block; (k, k) is
/// left for the next step's alpha.
/// @tparam T float or double
template<typename T>
void lanczos_arrowhead(wwr::wwrStream_t stream, int ncv, int k, const T *d_theta,
                       const T *d_s_last_row, int inc, const T *d_beta_m, T *d_T);

} // namespace calaman::device
