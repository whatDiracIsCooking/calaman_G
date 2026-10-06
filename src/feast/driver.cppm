/**
 * @file driver.cppm
 * @brief FEAST: the eigenpairs of a real symmetric matrix with eigenvalues in [Emin, Emax]
 *
 * The :driver partition of calaman.feast.
 *
 * Polizzi, "Density-matrix-based algorithm for solving eigenvalue problems",
 * Phys. Rev. B 79 (2009) 115112, arXiv:0901.2665.
 *
 * Subspace iteration with a rational filter. rho(A) approximates the spectral
 * projector onto the eigenvectors with eigenvalues in the interval, so applying
 * it to m0 >= m vectors and extracting Ritz pairs converges to the m wanted
 * eigenpairs -- at a rate set by how sharply rho falls away outside the interval,
 * which is what more quadrature nodes buy.
 *
 * One iteration:
 *
 *   1. basis = rho(A) Q = sum_e Re[ w_e (Z_e I - A)^{-1} Q ]      :contour_filter
 *   2. Rayleigh-Ritz on span(basis): Ritz pairs (lambda, Q)        :rayleigh_ritz
 *   3. residuals of the pairs inside [Emin, Emax]                  :rayleigh_ritz
 *   4. stop once the count m repeats and every residual is below tol
 *
 * All on one stream. The Ne shifted systems go to the batched BLAS as one
 * factorization per solve and one solve per iteration, and each iteration ends
 * in a single device-to-host copy of a small status block -- the only
 * synchronization in the loop.
 */

module;

#include "feast_bridge.h"

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.feast:driver;

import std;
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize, wwrMemcpyDeviceToHost, wwrSuccess
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_*, wwrblasFillMode_t
import wwr.solver;          // wwrsolverDnHandle_t
import calaman.common;  // real_fp
export import calaman.iterative; // IterationInfo, stop_reason, converged
import :buffer_size;
import :compute_quadrature;
import :contour_filter;
import :rayleigh_ritz;
export import calaman.error_handling; // Status -- the cross-domain return type

export namespace calaman {

/// Why the solver stopped.
enum class FeastStopReason {
  Converged,        ///< m repeated and every residual fell below the tolerance
  MaxIterations,    ///< the iteration budget ran out
  SubspaceTooSmall, ///< every Ritz value fell inside the interval: m0 is too small
  NumericalFailure, ///< a factorization broke down, or a kernel failed
};
static_assert(stop_reason<FeastStopReason>);

/// Tuning for feast.
template<calaman::real_fp T>
struct FeastOptions {
  int max_iterations = 20;

  /**
   * Converged when every returned pair's relative residual
   *
   *   ||A x - lambda x||_1 / ((||A||_1 + |lambda|) ||x||_1)
   *
   * -- its normwise backward error -- is below this.
   */
  T tol = std::is_same_v<T, float> ? T(1e-5) : T(1e-12);
};

/// What the solver did.
template<calaman::real_fp T>
struct FeastInfo : IterationInfo<FeastStopReason> {
  int m = 0;             ///< eigenvalues found in [Emin, Emax]: the leading m of d_lambda, d_Q
  T max_residual = T(0); ///< largest relative residual among those m, at the last iteration
};

/**
 * @brief The eigenpairs of the real symmetric A with eigenvalues in [Emin, Emax].
 *
 * @tparam T  float or double.
 * @tparam Ne Quadrature nodes on the contour, 4 or 8. More nodes sharpen the
 *            filter, so fewer iterations, at Ne n^2 complex elements of workspace
 *            and Ne solves per iteration.
 *
 * @param cublas_handle   BLAS handle, its stream set to @p stream.
 * @param cusolver_handle Dense-solver handle, its stream set to @p stream.
 * @param stream          Stream all work is enqueued on.
 * @param uplo            Which triangle of @p d_A is stored; the other is never read.
 * @param n               Matrix dimension.
 * @param d_A             n x n real symmetric, column-major, leading dimension @p lda.
 * @param lda             Leading dimension of @p d_A, >= n.
 * @param Emin, Emax      The interval, Emin < Emax, both finite.
 * @param m0              Subspace size, 1 <= m0 <= n. Must exceed the number of
 *                        eigenvalues in the interval; about 1.5 times it is usual.
 * @param d_lambda        Out: m0 Ritz values. The first info->m are the
 *                        eigenvalues in [Emin, Emax], ascending; the rest come
 *                        from the remaining directions of the subspace and carry
 *                        no accuracy guarantee.
 * @param d_Q             In: n x m0 starting subspace, leading dimension n, of
 *                        full column rank -- random is the usual choice. Out: the
 *                        Ritz vectors, orthonormal, matching d_lambda.
 * @param d_work          Device workspace, from feast_bufferSize<T, Ne>.
 * @param lwork_bytes     Its size.
 * @param opts            Tuning.
 * @param info            Host out, may be null.
 *
 * @return A Status: success whenever the iteration reached one of its own
 *         stopping conditions -- including MaxIterations and SubspaceTooSmall,
 *         which are outcomes rather than errors; read @p info to tell them apart.
 *         INVALID_VALUE for bad arguments, ALLOC_FAILED for a short workspace, and
 *         any other failing step's own-domain status with reason NumericalFailure.
 *
 * Convergence is declared only once m has been seen twice running, so a solve
 * takes at least two iterations. From a random start a Ritz value that has not
 * converged yet can sit inside the interval for an iteration or two, inflating m;
 * requiring it to repeat keeps that from ending the iteration early.
 *
 * SubspaceTooSmall means every Ritz value landed inside the interval on some
 * iteration after the first: the interval may hold more eigenvalues than m0 can
 * carry, and the ones returned cannot be trusted to be all of them. Raise m0.
 */
template<calaman::real_fp T, std::size_t Ne = 8>
  requires(Ne == 4 || Ne == 8)
Status feast(wwr::wwrblasHandle_t cublas_handle, wwr::wwrsolverDnHandle_t cusolver_handle,
             wwr::wwrStream_t stream, const wwr::wwrblasFillMode_t uplo, const int n, const T *d_A,
             const int lda, const T Emin, const T Emax, const int m0, T *d_lambda, T *d_Q,
             void *d_work, const std::size_t lwork_bytes, const FeastOptions<T> &opts = {},
             FeastInfo<T> *info = nullptr) {
  FeastInfo<T> local{};
  const auto publish = [&](const FeastStopReason reason, const Status st) {
    local.reason = reason;
    if (info != nullptr) {
      *info = local;
    }
    return st;
  };

  if (n < 1 || m0 < 1 || m0 > n || lda < n) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (uplo != wwr::WWRBLAS_FILL_MODE_LOWER && uplo != wwr::WWRBLAS_FILL_MODE_UPPER) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (!std::isfinite(Emin) || !std::isfinite(Emax) || !(Emin < Emax)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (d_A == nullptr || d_lambda == nullptr || d_Q == nullptr || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  FeastSlices<T> s;
  std::size_t required = 0;
  CLM_TRY(make_feast_slices<T, Ne>(cusolver_handle, n, m0, d_work, &s, &required));
  if (lwork_bytes < required) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  const device::FeastContour<T> contour = make_feast_contour<T, Ne>(Emin, Emax);

  // ── once per solve: factor the resolvents, and ||A||_1 ──────────────────
  // NumericalFailure steps publish the reason before returning the step's own
  // Status, so they keep the check-and-publish shape rather than a bare CLM_TRY.
  Status status = factor_resolvents<T>(cublas_handle, stream, uplo, n, d_A, lda, contour, s);
  if (!status.ok()) {
    return publish(FeastStopReason::NumericalFailure, status);
  }

  status = matrix_norm<T>(stream, uplo, n, d_A, lda, s);
  if (!status.ok()) {
    return publish(FeastStopReason::NumericalFailure, status);
  }

  device::FeastStatus<T> h{};
  int m_prev = -1; // no count yet

  for (int k = 0; k < opts.max_iterations; ++k) {
    status = apply_filter<T>(cublas_handle, stream, n, m0, d_Q, contour, s, s.basis);
    if (!status.ok()) {
      return publish(FeastStopReason::NumericalFailure, status);
    }

    status = rayleigh_ritz<T>(cublas_handle, cusolver_handle, stream, uplo, n, d_A, lda, m0, Emin,
                              Emax, s, d_lambda, d_Q);
    if (!status.ok()) {
      return publish(FeastStopReason::NumericalFailure, status);
    }

    status = residuals<T>(stream, n, m0, d_Q, d_lambda, s);
    if (!status.ok()) {
      return publish(FeastStopReason::NumericalFailure, status);
    }

    // The loop's one synchronization: everything the host decides on. A failed
    // copy or sync is published with its own runtime-domain Status.
    if (const Status copy = wwr::wwrMemcpyAsync(&h, s.status, sizeof(h),
                                                wwr::wwrMemcpyDeviceToHost, stream);
        !copy.ok()) {
      return publish(FeastStopReason::NumericalFailure, copy);
    }
    if (const Status sync = wwr::wwrStreamSynchronize(stream); !sync.ok()) {
      return publish(FeastStopReason::NumericalFailure, sync);
    }

    local.iterations = k + 1;
    local.m = h.m;
    local.max_residual = h.max_residual;

    for (std::size_t e = 0; e < Ne; ++e) {
      if (h.lu_info[e] != 0) {
        return publish(FeastStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
      }
    }
    if (h.qr_info[0] != 0 || h.qr_info[1] != 0 || h.eig_info != 0) {
      return publish(FeastStopReason::NumericalFailure, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
    }

    // Not judged on the first iteration, whose Ritz values from a random start
    // can crowd into the interval before they settle.
    if (k > 0 && h.m == m0) {
      return publish(FeastStopReason::SubspaceTooSmall, wwr::WWRBLAS_STATUS_SUCCESS);
    }

    if (h.m == m_prev && h.max_residual < opts.tol) {
      return publish(FeastStopReason::Converged, wwr::WWRBLAS_STATUS_SUCCESS);
    }
    m_prev = h.m;
  }

  return publish(FeastStopReason::MaxIterations, wwr::WWRBLAS_STATUS_SUCCESS);
}

} // namespace calaman
