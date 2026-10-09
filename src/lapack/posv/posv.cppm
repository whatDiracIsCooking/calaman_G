/**
 * @file posv.cppm
 * @brief The calaman.posv module -- solve A X = B for a Hermitian/symmetric
 *        positive definite matrix, LAPACK's ?posv
 *
 * The driver: Cholesky-factor A (wwr::potrf), then solve with the factor
 * (wwr::potrs) -- both the vendor cuSOLVER/hipSOLVER routines, backend-neutral,
 * so this module owns no device code. Templated over the four element types;
 * complex A is HERMITIAN. posv_bufferSize wraps wwr::potrf_bufferSize, so a query
 * and the routine size the SAME vendor workspace and cannot drift.
 *
 * SYNCHRONIZES @p stream once, to read the factorization's info onto the host:
 * the reference ?posv solves only when ?potrf succeeds, so a non-zero info (a
 * leading minor not positive definite) skips the solve and rides out through
 * @p d_info.
 *
 * Allocation-free shipped surface: A, B, @p d_work and @p d_info are
 * caller-provided device pointers; the solver handle must be bound to @p stream.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

export module calaman.posv;

import std;                 // std::max
import wwr.blas;            // WWRBLAS_FILL_MODE_*, WWRBLAS_STATUS_*
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.runtime_api;     // wwrStream_t, wwrMemcpyAsync, wwrStreamSynchronize
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.solver; // potrf, potrf_bufferSize, potrs
import calaman.common;      // Uplo, usual_fp

// export import: posv RETURNS calaman::Status, so a consumer must see it.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Device workspace, in elements of T, required by posv() (LAPACK ?posv)
///
/// Forwards to wwr::potrf_bufferSize: posv's only workspace is the vendor
/// factorization's.
///
/// @tparam T Element type; one of the four usual_fp types
/// @param solver_handle Solver handle, queried for the potrf workspace size
/// @param uplo Which triangle of A is referenced
/// @param n Order of the matrix A
/// @param lwork Output: required workspace in elements of T
/// @return Status: SUCCESS, WWRBLAS_STATUS_INVALID_VALUE for n < 0, or the
///         solver-domain status if the query fails
export template<usual_fp T>
Status posv_bufferSize(wwr::wwrsolverDnHandle_t solver_handle, const Uplo uplo, const int n,
                       int *lwork) {
  if (n < 0) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  *lwork = 0;
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  const wwr::wwrblasFillMode_t fill =
      uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
  // lda is irrelevant to the vendor size; pass n.
  CLM_TRY(wwr::potrf_bufferSize<T>(solver_handle, fill, n, static_cast<T *>(nullptr), n, lwork));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Solve A X = B for a Hermitian/symmetric positive definite A (LAPACK ?posv)
///
/// Factors A = U^H U (@p uplo == Uplo::U) or L L^H (Uplo::L) in place (wwr::potrf),
/// recording the factorization status in @p d_info, then overwrites @p d_B with
/// the solution X (wwr::potrs). A non-zero @p d_info leaves @p d_B untouched, as
/// the reference ?posv does. The returned Status carries only device/solver
/// errors; the numerical outcome rides @p d_info.
///
/// @tparam T Element type; one of the four usual_fp types
/// @param solver_handle Solver handle, bound to @p stream
/// @param stream Stream all work is enqueued on
/// @param uplo Which triangle of A is referenced and holds the factor
/// @param n Order of the matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; its @p uplo triangle holds the factor on return
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @param d_work Device workspace, >= posv_bufferSize<T>() elements of T
/// @param lwork Size of @p d_work in elements of T
/// @param d_info Device int: 0 on success, i > 0 if the order-i leading minor is not PD
/// @return Status: SUCCESS, the first error, or WWRBLAS_STATUS_ALLOC_FAILED if lwork is too small
export template<usual_fp T>
Status posv(wwr::wwrsolverDnHandle_t solver_handle, wwr::wwrStream_t stream, const Uplo uplo,
            const int n, const int nrhs, T *d_A, const int lda, T *d_B, const int ldb, T *d_work,
            const int lwork, int *d_info) {
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  int required = 0;
  CLM_TRY(posv_bufferSize<T>(solver_handle, uplo, n, &required));
  if (lwork < required || (required > 0 && d_work == nullptr)) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  const wwr::wwrblasFillMode_t fill =
      uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
  CLM_TRY(wwr::potrf<T>(solver_handle, fill, n, d_A, lda, d_work, lwork, d_info));

  // The reference ?posv solves only when ?potrf succeeds: read info and stop if
  // the factorization failed (the routine's only synchronization).
  int info_host = 0;
  CLM_TRY(wwr::wwrMemcpyAsync(&info_host, d_info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  if (info_host != 0 || nrhs == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // potrs reports into d_info too; it is 0 for the arguments validated above.
  CLM_TRY(wwr::potrs<T>(solver_handle, fill, n, nrhs, d_A, lda, d_B, ldb, d_info));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
