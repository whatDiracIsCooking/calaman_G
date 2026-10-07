/**
 * @file rayleigh_ritz.cppm
 * @brief Rayleigh-Ritz on the filtered subspace; its pairs' residuals and their scale
 *
 * The :rayleigh_ritz partition of calaman.feast.
 *
 *   basis <- Q from qr(basis)                    orthogonalize
 *   W      = A basis                             op.apply
 *   H      = basis^T W                           gemm, m0 x m0
 *   H      = Y diag(ritz) Y^T                    syevd
 *   pairs inside [Emin, Emax] rotated first      device::feast_select
 *   X      = basis Y,   A X = W Y                two ritz_rotates
 *
 * The filtered subspace is orthonormalized before projecting, not handled as the
 * paper's H y = lambda (basis^T basis) y: that Gram matrix's conditioning goes
 * like rho^2 and worsens as the filter sharpens (README, departure 1).
 *
 * A X comes from W Y rather than a second product with A: the same vectors, at
 * n m0^2 flops instead of n^2 m0.
 *
 * Every function here expects the handles' streams to be @p stream. A non-success
 * eigensolver result surfaces as its own Status (solver domain); a kernel-launch
 * failure through wwr::wwrGetLastError() (runtime domain).
 */

module;

#include "feast_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.feast:rayleigh_ritz;

import std;
import wwr.runtime_api;         // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.blas;                // wwrblasHandle_t, WWRBLAS_*
import wwr.solver;              // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.wrappers.blas;       // gemm
import wwr.wrappers.solver;     // syevd
import wwr.extension.blas;      // ScopedPointerMode (forces host mode for the BLAS calls)
import calaman.common;          // kOne, kZero, real_fp
import calaman.orthogonalize;   // orthogonalize
import calaman.ritz;            // ritz_rotate
import calaman.linear_operator; // linear_operator
import calaman.lacn2;           // lacn2, the hook-less model's ||A||_1 estimate
import :buffer_size;
import :resolvent; // feast_norm1_hook
export import calaman.error_handling; // Status, PointerModeStatus

namespace calaman {

/**
 * @brief Ritz pairs of A on span(s.basis): values into @p d_lambda, vectors into @p d_X.
 *
 * s.basis is overwritten with an orthonormal basis of itself. On return the pairs
 * with values in [Emin, Emax] come first -- status->m of them, ascending -- and
 * the other m0 - m pairs follow. Factorization failures land in s.qr_info and
 * s.eig_info; the count lands in s.status.
 *
 * @param op   A, applied once to the m0-column basis.
 * @param d_X  Out: n x m0, leading dimension n.
 */
template<calaman::real_fp T, linear_operator<T> Op>
Status rayleigh_ritz(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
                     wwr::wwrStream_t stream, Op &op, const int n, const int m0, const T Emin,
                     const T Emax, const FeastSlices<T> &s, T *d_lambda, T *d_X) {
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  orthogonalize<T>(cusolver_handle, n, m0, s.basis, s.scratch, s.lwork_qr, s.qr_info,
                   s.qr_info + 1);

  CLM_TRY(op.apply(stream, m0, s.basis, s.a_basis));

  CLM_TRY(wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, m0, m0, n,
                            &kOne<T>, s.basis, n, s.a_basis, n, &kZero<T>, s.projected, m0));

  // H is symmetric to roundoff; syevd reads only its lower triangle.
  CLM_TRY(wwr::syevd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                        wwr::WWRBLAS_FILL_MODE_LOWER, m0, s.projected, m0, s.ritz, s.scratch,
                        s.lwork_eig, s.eig_info));

  device::feast_select(stream, m0, Emin, Emax, s.ritz, s.projected, d_lambda, s.rotated, s.status);
  CLM_TRY(wwr::wwrGetLastError());

  // Already in host mode, so ritz_rotate's own guard costs handle-state calls, no sync.
  CLM_TRY(ritz_rotate<T>(cublas_handle, n, m0, m0, s.basis, n, s.rotated, m0, d_X, n));
  CLM_TRY(ritz_rotate<T>(cublas_handle, n, m0, m0, s.a_basis, n, s.rotated, m0, s.a_ritz, n));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief ||A||_1 into s.norm_a, the residuals' scale, and onto the host in @p norm.
 *
 * A feast_norm1_hook model supplies it; any other gets lacn2's lower bound,
 * driven by op.apply (A symmetric, so A^T x = A x) in the Rayleigh-Ritz blocks,
 * which are free before the loop. Synchronizes @p stream once per solve.
 */
template<calaman::real_fp T, linear_operator<T> Op>
Status residual_scale(wwr::wwrblasHandle_t cublas_handle, wwr::wwrStream_t stream, Op &op,
                      const int n, const FeastSlices<T> &s, T &norm) {
  if constexpr (feast_norm1_hook<Op, T>) {
    CLM_TRY(op.norm1_estimate(stream, s.norm_a));
    CLM_TRY(wwr::wwrMemcpyAsync(&norm, s.norm_a, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  } else {
    wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
    const wwr::extension::ScopedPointerMode mode{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                                 PointerModeStatus{&pm_status}};
    CLM_TRY(pm_status);

    T *const x = s.basis;    // lacn2's x; op.apply cannot write it in place,
    T *const ax = s.a_basis; // so A x lands here and is copied back
    int kase = 0;
    std::array<int, 3> isave{};
    norm = T(0);
    do {
      CLM_TRY(lacn2<T>(cublas_handle, n, s.a_ritz, x, s.norm_work, s.norm_work_bytes, norm, kase,
                       isave));
      if (kase != 0) {
        CLM_TRY(op.apply(stream, 1, x, ax));
        CLM_TRY(wwr::wwrMemcpyAsync(x, ax, sizeof(T) * static_cast<std::size_t>(n),
                                    wwr::wwrMemcpyDeviceToDevice, stream));
      }
    } while (kase != 0);
    CLM_TRY(wwr::wwrMemcpyAsync(s.norm_a, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  }
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief The relative residual of each in-interval Ritz pair from the last
 *        rayleigh_ritz, into s.residuals, and their maximum into s.status.
 *
 *   ||A x - lambda x||_1 / ((||A||_1 + |lambda|) ||x||_1)
 *
 * is the normwise backward error of (lambda, x): the size of the smallest
 * relative perturbation of A for which the pair is exact. So a tolerance on it
 * reads the same whatever the scale of A or of the interval.
 */
template<calaman::real_fp T>
Status residuals(wwr::wwrStream_t stream, const int n, const int m0, const T *d_X,
                 const T *d_lambda, const FeastSlices<T> &s) {
  device::feast_residuals(stream, n, m0, d_X, s.a_ritz, d_lambda, s.norm_a, s.residuals, s.status);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
