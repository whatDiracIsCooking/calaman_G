/**
 * @file buffer_size.cppm
 * @brief Device workspace layout and sizing for cg_unitary
 *
 * Partition of calaman.cg_unitary.
 *
 * One caller-provided buffer, laid out by the single CgSlices::carve through
 * calaman::carve_workspace: cg_unitary_bufferSize runs it over a null base and
 * make_cg_slices over the real one, so the size query cannot drift from the
 * carving. Two rules decide fixed versus scratch: the solver's matrices all
 * coexist across an iteration and so accumulate, while the matrix exponential
 * and the cost function are never live at the same instant -- the search forms
 * its base exponential, then sweeps gradients -- and so alias.
 */

module;

#include "cg_unitary_bridge.h"

export module calaman.cg_unitary:buffer_size;

import std;
import wwr.blas;            // WWRBLAS_STATUS_* (invalid-value / success returns)
import wwr.solver;         // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_STATUS_*
import wwr.wrappers.common; // usual_fp, ComplexToRealType, RealToComplexType
import calaman.common;      // WorkspaceLayout, carve_workspace
import calaman.error_handling; // Status
import calaman.expm;        // expm_bufferSize
import :cost_function;

export namespace calaman {

/**
 * @brief Pointers into the solver's single workspace buffer.
 *
 * Assembled once per solve by make_cg_slices and passed down into the line
 * search, so the search allocates nothing and the layout lives in one place:
 * carve(). Every matrix slice is packed (leading dimension n), which the
 * Frobenius reductions and the cost functor's device-mode dot both require.
 */
template<wwr::usual_fp T>
struct CgSlices {
  using RealT = wwr::ComplexToRealType<T>;
  using CplxT = wwr::RealToComplexType<RealT>;

  // ── persistent across the iteration ───────────────────────────────────
  T *psi = nullptr;       ///< Euclidean gradient at the current point
  T *grad = nullptr;      ///< Riemannian gradient G_k, at the group identity
  T *grad_next = nullptr; ///< G_{k+1}
  T *dir = nullptr;       ///< search direction H_k, in u(n)
  T *tmp = nullptr;       ///< Psi W^H, then H W_t, then G_{k+1} - G_k
  T *w_new = nullptr;     ///< R W, and the trial point inside the search
  T *rot = nullptr;       ///< exp(mu H), and the search's base exponential
  T *rot_acc = nullptr;   ///< accumulated power R_i inside the search
  T *rot_tmp = nullptr;   ///< target of R_i R_1; gemm cannot alias its output
  T *psi_trial = nullptr; ///< Euclidean gradient at a trial point

  // ── line-search scalars, all device-resident ──────────────────────────
  T *dots = nullptr;          ///< derivative dot products, kCgMaxSamples
  T *cost_dots = nullptr;     ///< sampled cost dot products, kCgMaxSamples
  RealT *coeffs_real = nullptr; ///< Table 1 polynomial, kCgMaxSamples
  RealT *args = nullptr;        ///< unit-circle root arguments, kCgMaxSamples
  RealT *cost_vals = nullptr;   ///< sampled cost values, kCgMaxSamples
  RealT *mu = nullptr;          ///< the chosen step size, one element
  CplxT *coeffs_cplx = nullptr; ///< Table 2 Fourier polynomial, kCgMaxSamples
  RealT *colsum = nullptr;      ///< matrix_norm1 scratch, n reals
  T *power_v = nullptr;         ///< spectral_radius warm-start vector, n
  T *power_work = nullptr;      ///< spectral_radius workspace, 2n
  int *ints = nullptr;          ///< 4 ints: found, converged, num_args, info

  // ── aliased scratch ───────────────────────────────────────────────────
  void *scratch = nullptr; ///< expm workspace, or the cost functor's
  std::size_t scratch_bytes = 0;

  /// @brief Lay the slices out from @p layout -- the ONLY description of the
  ///        layout, run for sizing and carving alike (see carve_workspace).
  ///        All regions are FIXED except the one expm/cost SCRATCH block,
  ///        carved last.
  void carve(WorkspaceLayout &layout, const int n, const std::size_t expm_bytes,
             const std::size_t cost_bytes) {
    const std::size_t dn = static_cast<std::size_t>(n);
    const std::size_t nn = dn * dn;
    const std::size_t samples = static_cast<std::size_t>(kCgMaxSamples);

    psi = layout.fixed<T>(nn);
    grad = layout.fixed<T>(nn);
    grad_next = layout.fixed<T>(nn);
    dir = layout.fixed<T>(nn);
    tmp = layout.fixed<T>(nn);
    w_new = layout.fixed<T>(nn);
    rot = layout.fixed<T>(nn);
    rot_acc = layout.fixed<T>(nn);
    rot_tmp = layout.fixed<T>(nn);
    psi_trial = layout.fixed<T>(nn);

    dots = layout.fixed<T>(samples);
    cost_dots = layout.fixed<T>(samples);
    coeffs_real = layout.fixed<RealT>(samples);
    args = layout.fixed<RealT>(samples);
    cost_vals = layout.fixed<RealT>(samples);
    mu = layout.fixed<RealT>(samples);
    coeffs_cplx = layout.fixed<CplxT>(samples);
    colsum = layout.fixed<RealT>(dn);
    power_v = layout.fixed<T>(dn);
    power_work = layout.fixed<T>(2 * dn);
    ints = layout.fixed<int>(4);

    // The exponential and the cost function never overlap in time.
    scratch_bytes = std::max(expm_bytes, cost_bytes);
    scratch = layout.scratch<std::byte>(scratch_bytes);
  }
};

/**
 * @brief Device workspace, in bytes, required by cg_unitary for an n x n problem.
 *
 * Ten n x n blocks plus O(n) vectors, and one scratch region large enough for
 * whichever of the matrix exponential or the cost function needs more. At n = 6
 * in double complex that is about 9 KB; at n = 128, about 1.2 MB.
 *
 * @param cusolver_handle Queried for the exponential's own workspace size.
 * @param n               Matrix dimension.
 * @param cost_bytes      The cost functor's bufferSize(n).
 * @param lwork_bytes     Output: bytes required.
 */
template<wwr::usual_fp T>
Status cg_unitary_bufferSize(wwr::wwrsolverDnHandle_t cusolver_handle, const int n,
                             const std::size_t cost_bytes, std::size_t *lwork_bytes) {
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  std::size_t expm_bytes = 0;
  const Status status = expm_bufferSize<T>(cusolver_handle, n, &expm_bytes);
  if (!status.ok()) {
    return status;
  }

  *lwork_bytes = carve_workspace<CgSlices<T>>(nullptr, nullptr, n, expm_bytes, cost_bytes);
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Carve a workspace into CgSlices, through the same carve() the sizing ran.
 *
 * @param d_work      Buffer of at least cg_unitary_bufferSize bytes.
 * @param n           Matrix dimension.
 * @param expm_bytes  The exponential's workspace size, from expm_bufferSize.
 * @param cost_bytes  The cost functor's bufferSize(n).
 */
template<wwr::usual_fp T>
CgSlices<T> make_cg_slices(void *d_work, const int n, const std::size_t expm_bytes,
                           const std::size_t cost_bytes) {
  CgSlices<T> s;
  carve_workspace(d_work, &s, n, expm_bytes, cost_bytes);
  return s;
}

} // namespace calaman
