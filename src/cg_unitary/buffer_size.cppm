/**
 * @file buffer_size.cppm
 * @brief Device workspace layout and sizing for cg_unitary
 *
 * Partition of calaman.cg_unitary.
 *
 * One caller-provided buffer, sized with WorkspaceBuilder. Two rules decide
 * add_fixed versus add_scratch: the solver's matrices all coexist across an
 * iteration and so accumulate, while the matrix exponential and the cost
 * function are never live at the same instant -- the search forms its base
 * exponential, then sweeps gradients -- and so alias.
 */

module;

#include "cg_unitary_bridge.h"

export module calaman.cg_unitary:buffer_size;

import std;
import wwr.blas;            // WWRBLAS_STATUS_* (invalid-value / success returns)
import wwr.solver;         // wwrsolverDnHandle_t, wwrsolverStatus_t, WWRSOLVER_STATUS_*
import wwr.wrappers.common; // usual_fp, ComplexToRealType, RealToComplexType
import calaman.common;      // WorkspaceBuilder, align_up
import calaman.error_handling; // Status
import calaman.expm;        // expm_bufferSize
import :cost_function;

export namespace calaman {

/**
 * @brief Pointers into the solver's single workspace buffer.
 *
 * Assembled once per solve by make_cg_slices and passed down into the line
 * search, so the search allocates nothing and the sizing lives in one place.
 * Every matrix slice is packed (leading dimension n), which the Frobenius
 * reductions and the cost functor's device-mode dot both require.
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

  /// Number of n x n matrix blocks the layout reserves.
  static constexpr int kMatrixBlocks = 10;
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
  using RealT = wwr::ComplexToRealType<T>;
  using CplxT = wwr::RealToComplexType<RealT>;
  if (n < 1) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  std::size_t expm_bytes = 0;
  const Status status = expm_bufferSize<T>(cusolver_handle, n, &expm_bytes);
  if (!status.ok()) {
    return status;
  }

  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t samples = static_cast<std::size_t>(kCgMaxSamples);
  const std::size_t dn = static_cast<std::size_t>(n);

  WorkspaceBuilder wb;
  wb.add_fixed<T>(nn, CgSlices<T>::kMatrixBlocks); // the ten matrix blocks
  wb.add_fixed<T>(samples);                        // dots
  wb.add_fixed<T>(samples);                        // cost_dots
  wb.add_fixed<RealT>(samples);                    // coeffs_real
  wb.add_fixed<RealT>(samples);                    // args
  wb.add_fixed<RealT>(samples);                    // cost_vals
  wb.add_fixed<RealT>(samples);                    // mu
  wb.add_fixed<CplxT>(samples);                    // coeffs_cplx
  wb.add_fixed<RealT>(dn);                         // colsum
  wb.add_fixed<T>(dn);                             // power_v
  wb.add_fixed<T>(2 * dn);                         // power_work
  wb.add_fixed<int>(4);                            // ints

  // The exponential and the cost function never overlap in time.
  wb.add_scratch<std::byte>(expm_bytes);
  wb.add_scratch<std::byte>(cost_bytes);

  *lwork_bytes = wb.total();
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief Carve a workspace into CgSlices. Must mirror cg_unitary_bufferSize.
 *
 * @param d_work      Buffer of at least cg_unitary_bufferSize bytes.
 * @param n           Matrix dimension.
 * @param expm_bytes  The exponential's workspace size, from expm_bufferSize.
 * @param cost_bytes  The cost functor's bufferSize(n).
 */
template<wwr::usual_fp T>
CgSlices<T> make_cg_slices(void *d_work, const int n, const std::size_t expm_bytes,
                           const std::size_t cost_bytes) {
  using RealT = wwr::ComplexToRealType<T>;
  using CplxT = wwr::RealToComplexType<RealT>;

  CgSlices<T> s;
  auto *p = static_cast<std::byte *>(d_work);

  const auto bump = [&p](const std::size_t bytes) {
    auto *here = p;
    p += align_up(bytes, std::size_t{256});
    return here;
  };

  const std::size_t block = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(T);
  s.psi = reinterpret_cast<T *>(bump(block));
  s.grad = reinterpret_cast<T *>(bump(block));
  s.grad_next = reinterpret_cast<T *>(bump(block));
  s.dir = reinterpret_cast<T *>(bump(block));
  s.tmp = reinterpret_cast<T *>(bump(block));
  s.w_new = reinterpret_cast<T *>(bump(block));
  s.rot = reinterpret_cast<T *>(bump(block));
  s.rot_acc = reinterpret_cast<T *>(bump(block));
  s.rot_tmp = reinterpret_cast<T *>(bump(block));
  s.psi_trial = reinterpret_cast<T *>(bump(block));

  const std::size_t samples = static_cast<std::size_t>(kCgMaxSamples);
  s.dots = reinterpret_cast<T *>(bump(samples * sizeof(T)));
  s.cost_dots = reinterpret_cast<T *>(bump(samples * sizeof(T)));

  s.coeffs_real = reinterpret_cast<RealT *>(bump(samples * sizeof(RealT)));
  s.args = reinterpret_cast<RealT *>(bump(samples * sizeof(RealT)));
  s.cost_vals = reinterpret_cast<RealT *>(bump(samples * sizeof(RealT)));
  s.mu = reinterpret_cast<RealT *>(bump(samples * sizeof(RealT)));
  s.coeffs_cplx = reinterpret_cast<CplxT *>(bump(samples * sizeof(CplxT)));

  const std::size_t dn = static_cast<std::size_t>(n);
  s.colsum = reinterpret_cast<RealT *>(bump(dn * sizeof(RealT)));
  s.power_v = reinterpret_cast<T *>(bump(dn * sizeof(T)));
  s.power_work = reinterpret_cast<T *>(bump(2 * dn * sizeof(T)));
  s.ints = reinterpret_cast<int *>(bump(std::size_t{4} * sizeof(int)));

  s.scratch = p;
  s.scratch_bytes = (expm_bytes > cost_bytes) ? expm_bytes : cost_bytes;
  return s;
}

} // namespace calaman
