/**
 * @file hesv.cppm
 * @brief The calaman.hesv module -- solve A X = B for a Hermitian indefinite
 *        matrix, LAPACK's ?hesv
 *
 * The driver: factor A by the Hermitian Bunch-Kaufman method (calaman.hetf2 --
 * ours, since the vendors ship no hetrf), then solve with the factor
 * (calaman.hetrs). COMPLEX ONLY (c/z): a real Hermitian matrix is symmetric, which
 * is calaman.sysv.
 *
 * Unlike calaman.sysv, the factorization is ours rather than a vendor call, so
 * both halves run on the one BLAS handle and no solver handle is involved.
 * hesv_bufferSize forwards to hetf2_bufferSize, so a query and the routine size the
 * SAME workspace. SYNCHRONIZES the handle's stream once, to read the factorization
 * info: a non-zero info (a zero/NaN pivot) skips the solve and rides out through
 * @p d_info, as the reference ?hesv does.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv, B, @p d_work and
 * @p d_info are caller-provided device pointers. Requires the handle's DEFAULT
 * (host) pointer mode.
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

export module calaman.hesv;

import std;             // std::max
import wwr.blas;        // wwrblasHandle_t, wwrblasGetStream
import wwr.runtime_api; // wwrStream_t, wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // Uplo, complex_fp
import calaman.hetf2;   // hetf2, hetf2_bufferSize -- the factorization
import calaman.hetrs;   // hetrs -- the solve against the factor

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Device workspace, in elements of T, required by hesv() (= hetf2's, 2n)
export template<complex_fp T>
Status hesv_bufferSize(const int n, int *lwork) {
  return hetf2_bufferSize<T>(n, lwork);
}

/// @brief Solve A X = B for a Hermitian indefinite matrix A (LAPACK ?hesv)
///
/// Factors A = U D U^H (@p uplo == Uplo::U) or L D L^H (Uplo::L) in place by the
/// Hermitian Bunch-Kaufman method (calaman.hetf2), recording the pivots in
/// @p d_ipiv and the status in @p d_info, then overwrites @p d_B with the solution
/// X (calaman.hetrs). Complex only. A non-zero @p d_info (a zero/NaN pivot) leaves
/// @p d_B untouched, as the reference ?hesv does. The returned Status carries only
/// device/BLAS errors; the numerical outcome rides @p d_info.
///
/// @tparam T Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param handle GPU BLAS handle in host pointer mode; all arrays live on its device
/// @param uplo Which triangle of A is referenced and holds the factor
/// @param n Order of the Hermitian matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; its @p uplo triangle holds the factor on return
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device int array, length n; filled with the 1-based Bunch-Kaufman pivots
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @param d_work Device workspace, >= hesv_bufferSize<T>() elements of T
/// @param lwork Size of @p d_work in elements of T
/// @param d_info Device int: 0 on success, else the 1-based first zero/NaN pivot column
/// @return Status: SUCCESS, the first error, or WWRBLAS_STATUS_ALLOC_FAILED if lwork is too small
export template<complex_fp T>
Status hesv(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, const int nrhs, T *d_A,
            const int lda, int *d_ipiv, T *d_B, const int ldb, T *d_work, const int lwork,
            int *d_info) {
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  int required = 0;
  CLM_TRY(hesv_bufferSize<T>(n, &required));
  if (lwork < required || d_work == nullptr) {
    return wwr::WWRBLAS_STATUS_ALLOC_FAILED;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  CLM_TRY(hetf2<T>(handle, uplo, n, d_A, lda, d_ipiv, d_info, d_work));

  // The reference ?hesv does not solve against a singular D: read info and stop.
  int info_host = 0;
  CLM_TRY(wwr::wwrMemcpyAsync(&info_host, d_info, sizeof(int), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  if (info_host != 0 || nrhs == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  CLM_TRY(hetrs<T>(handle, uplo, n, nrhs, d_A, lda, d_ipiv, d_B, ldb));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
