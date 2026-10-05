/**
 * @file sysv.cppm
 * @brief The calaman.sysv module -- solve A X = B for a symmetric indefinite
 *        matrix, LAPACK's ?sysv
 *
 * The driver: factor A by the Bunch-Kaufman method (wwr::sytrf, the vendor
 * cuSOLVER/hipSOLVER routine, backend-neutral), then solve with the factor
 * (calaman.sytrs). Templated over the four element types; COMPLEX IS SYMMETRIC,
 * not Hermitian -- the Hermitian cousin is calaman.hesv.
 *
 * The factorization half is a single vendor call, so this module owns no device
 * code of its own; the only kernels it pulls in are calaman.sytrs's D^-1 applies.
 * sysv_bufferSize wraps wwr::sytrf_bufferSize, so a query and the routine size the
 * SAME vendor workspace and cannot drift.
 *
 * SYNCHRONIZES @p stream once, to read the factorization's info onto the host:
 * the reference ?sysv does not solve against a singular D, so a non-zero info
 * (singular factor, or a vendor bad-argument report) skips calaman.sytrs and
 * rides out through @p d_info, exactly as @p rank/@p info ride in pstrf.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv, B, @p d_work and
 * @p d_info are caller-provided device pointers. Both handles must be bound to
 * @p stream, and the BLAS handle must be in DEFAULT (host) pointer mode -- the
 * sytrs BLAS scalars are host constants.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import supplies.
#include "error_handling/error_macros.h"

export module calaman.sysv;

import std;                 // std::max
import wwr.blas;            // wwrblasHandle_t, WWRBLAS_FILL_MODE_*
import wwr.solver;          // wwrsolverDnHandle_t
import wwr.runtime_api;     // wwrStream_t, wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;         // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.solver; // sytrf, sytrf_bufferSize
import calaman.common;      // Uplo, usual_fp
import calaman.sytrs;       // sytrs -- the solve against the factor

// export import, not a plain import: sysv RETURNS calaman::Status, so a consumer
// of `import calaman.sysv;` must see Status's member functions -- the re-export
// pstrf/sytrs do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Device workspace, in elements of T, required by sysv() (LAPACK ?sysv)
///
/// Forwards to wwr::sytrf_bufferSize: sysv's only workspace is the vendor
/// factorization's. The returned count is in ELEMENTS of T, the lwork sytrf and
/// sysv take.
///
/// @tparam T Element type; one of the four usual_fp types
/// @param solver_handle Solver handle, queried for the sytrf workspace size
/// @param n Order of the symmetric matrix A
/// @param lwork Output: required workspace in elements of T
/// @return Status: SUCCESS, WWRBLAS_STATUS_INVALID_VALUE for n < 0, or the
///         solver-domain status if the query fails
export template<usual_fp T>
Status sysv_bufferSize(wwr::wwrsolverDnHandle_t solver_handle, const int n, int *lwork) {
  if (n < 0) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  *lwork = 0;
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  // lda is irrelevant to the vendor size; pass n, as expm's getrf query does.
  CLM_TRY(wwr::sytrf_bufferSize<T>(solver_handle, n, static_cast<T *>(nullptr), n, lwork));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Solve A X = B for a symmetric indefinite matrix A (LAPACK ?sysv)
///
/// Factors A = U D U^T (@p uplo == Uplo::U) or L D L^T (Uplo::L) in place by the
/// Bunch-Kaufman method (wwr::sytrf), recording the pivots in @p d_ipiv and the
/// factorization status in @p d_info, then overwrites @p d_B with the solution X
/// (calaman.sytrs). Complex is symmetric (no conjugation). A non-zero @p d_info
/// (singular D, or a vendor bad-argument report) leaves @p d_B untouched, as the
/// reference ?sysv does. The returned Status carries only device/BLAS/solver
/// errors; the numerical outcome rides @p d_info.
///
/// @tparam T Element type; one of the four usual_fp types
/// @param blas_handle BLAS handle in host pointer mode, bound to @p stream (for sytrs)
/// @param solver_handle Solver handle, bound to @p stream (for sytrf)
/// @param stream Stream all work is enqueued on; both handles must be bound to it
/// @param uplo Which triangle of A is referenced and holds the factor
/// @param n Order of the symmetric matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; its @p uplo triangle holds the factor on return
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device pivot array, length n; filled with the 1-based Bunch-Kaufman pivots
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @param d_work Device workspace, >= sysv_bufferSize<T>() elements of T
/// @param lwork Size of @p d_work in elements of T
/// @param d_info Device int: 0 on success, > 0 for a singular factor, < 0 for a bad argument
/// @return Status: SUCCESS, the first error, or WWRBLAS_STATUS_ALLOC_FAILED if lwork is too small
export template<usual_fp T>
Status sysv(wwr::wwrblasHandle_t blas_handle, wwr::wwrsolverDnHandle_t solver_handle,
            wwr::wwrStream_t stream, const Uplo uplo, const int n, const int nrhs, T *d_A,
            const int lda, int *d_ipiv, T *d_B, const int ldb, T *d_work, const int lwork,
            int *d_info) {
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  int required = 0;
  CLM_TRY(sysv_bufferSize<T>(solver_handle, n, &required));
  if (lwork < required || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  const wwr::wwrblasFillMode_t fill =
      uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
  CLM_TRY(wwr::sytrf<T>(solver_handle, fill, n, d_A, lda, d_ipiv, d_work, lwork, d_info));

  // The reference ?sysv does not solve against a singular D: read info and stop
  // if the factorization flagged one (the routine's only synchronization).
  int info_host = 0;
  CLM_TRY(wwr::wwrMemcpyAsync(&info_host, d_info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  if (info_host != 0 || nrhs == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  CLM_TRY(sytrs<T>(blas_handle, uplo, n, nrhs, d_A, lda, d_ipiv, d_B, ldb));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
