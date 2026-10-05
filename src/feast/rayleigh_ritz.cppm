/**
 * @file rayleigh_ritz.cppm
 * @brief Rayleigh-Ritz on the filtered subspace, and the residuals of its Ritz pairs
 *
 * The :rayleigh_ritz partition of calaman.feast.
 *
 *   basis <- Q from qr(basis)                    orthogonalize
 *   W      = A basis                             symm
 *   H      = basis^T W                           gemm, m0 x m0
 *   H      = Y diag(ritz) Y^T                    syevd
 *   pairs inside [Emin, Emax] rotated first      device::feast_select
 *   X      = basis Y,   A X = W Y                two gemms
 *
 * The filtered subspace is not orthonormal -- rho(A) shrinks each eigenvector
 * direction by a different factor -- so it is orthonormalized before projecting.
 * The paper instead solves the reduced generalized problem
 * H y = lambda (basis^T basis) y; but basis^T basis is only as well conditioned
 * as rho is flat across the subspace -- its smallest eigenvalues go like
 * rho(lambda)^2 for the eigenvalues farthest outside the interval that the
 * subspace still carries -- so it worsens as the filter sharpens or m0 grows. QR
 * never squares that.
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
import wwr.runtime_api;     // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*
import wwr.solver;          // wwrsolverDnHandle_t, WWRSOLVER_EIG_MODE_VECTOR
import wwr.wrappers.blas;   // symm, gemm
import wwr.wrappers.solver; // syevd
import wwr.extension.blas;  // ScopedPointerMode (forces host mode for the BLAS calls)
import calaman.common;      // kOne, kZero, real_fp
import calaman.orthogonalize; // orthogonalize
import :buffer_size;
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
 * @param uplo Which triangle of @p d_A is stored.
 * @param d_X  Out: n x m0, leading dimension n.
 */
template<calaman::real_fp T>
Status feast_rayleigh_ritz(wwr::wwrblasHandle_t cublas_handle,
                           wwr::wwrsolverDnHandle_t cusolver_handle, wwr::wwrStream_t stream,
                           const wwr::wwrblasFillMode_t uplo, const int n, const T *d_A,
                           const int lda, const int m0, const T Emin, const T Emax,
                           const FeastSlices<T> &s, T *d_lambda, T *d_X) {
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  const wwr::extension::ScopedPointerMode mode{cublas_handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                               PointerModeStatus{&pm_status}};
  CLM_TRY(pm_status);

  orthogonalize<T>(cusolver_handle, n, m0, s.basis, s.scratch, s.lwork_qr, s.qr_info,
                   s.qr_info + 1);

  CLM_TRY(wwr::symm<T, int>(cublas_handle, wwr::WWRBLAS_SIDE_LEFT, uplo, n, m0, &kOne<T>, d_A, lda,
                            s.basis, n, &kZero<T>, s.a_basis, n));

  CLM_TRY(wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, m0, m0, n,
                            &kOne<T>, s.basis, n, s.a_basis, n, &kZero<T>, s.projected, m0));

  // H is symmetric to roundoff; syevd reads only its lower triangle.
  CLM_TRY(wwr::syevd<T>(cusolver_handle, wwr::WWRSOLVER_EIG_MODE_VECTOR,
                        wwr::WWRBLAS_FILL_MODE_LOWER, m0, s.projected, m0, s.ritz, s.scratch,
                        s.lwork_eig, s.eig_info));

  device::feast_select(stream, m0, Emin, Emax, s.ritz, s.projected, d_lambda, s.rotated, s.status);
  CLM_TRY(wwr::wwrGetLastError());

  CLM_TRY(wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, m0, m0,
                            &kOne<T>, s.basis, n, s.rotated, m0, &kZero<T>, d_X, n));

  CLM_TRY(wwr::gemm<T, int>(cublas_handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, m0, m0,
                            &kOne<T>, s.a_basis, n, s.rotated, m0, &kZero<T>, s.a_ritz, n));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief ||A||_1 into s.norm_a, the scale feast_residuals measures against.
 *        Once per solve.
 */
template<calaman::real_fp T>
Status feast_matrix_norm(wwr::wwrStream_t stream, const wwr::wwrblasFillMode_t uplo, const int n,
                         const T *d_A, const int lda, const FeastSlices<T> &s) {
  device::feast_sym_norm1(stream, uplo == wwr::WWRBLAS_FILL_MODE_LOWER, n, d_A, lda, s.colsum,
                          s.norm_a);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/**
 * @brief The relative residual of each in-interval Ritz pair from the last
 *        feast_rayleigh_ritz, into s.residuals, and their maximum into s.status.
 *
 *   ||A x - lambda x||_1 / ((||A||_1 + |lambda|) ||x||_1)
 *
 * is the normwise backward error of (lambda, x): the size of the smallest
 * relative perturbation of A for which the pair is exact. So a tolerance on it
 * reads the same whatever the scale of A or of the interval.
 */
template<calaman::real_fp T>
Status feast_residuals(wwr::wwrStream_t stream, const int n, const int m0, const T *d_X,
                       const T *d_lambda, const FeastSlices<T> &s) {
  device::feast_residuals(stream, n, m0, d_X, s.a_ritz, d_lambda, s.norm_a, s.residuals, s.status);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
