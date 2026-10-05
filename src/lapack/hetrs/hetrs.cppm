/**
 * @file hetrs.cppm
 * @brief The calaman.hetrs module -- solve A X = B for a Hermitian matrix already
 *        factored by the Bunch-Kaufman method, LAPACK's ?hetrs
 *
 * Given the factor A = U D U^H (Uplo::U) or L D L^H (Uplo::L) and the pivot
 * sequence @p d_ipiv that ?hetrf produced, overwrites the right-hand sides @p d_B
 * with the solution X. COMPLEX ONLY (c/z): a real Hermitian matrix is symmetric,
 * which is calaman.sytrs.
 *
 * Structure follows the reference level-2 ?hetrs exactly, and differs from the
 * symmetric calaman.sytrs only where Hermitian-ness bites:
 *   * the rank-1 updates are geru (unconjugated), as in the complex symmetric case;
 *   * the 1x1 D apply divides by the REAL diagonal D(k,k);
 *   * the 2x2 D apply conjugates the off-diagonal on one row;
 *   * each back-substitution gemv is conjugate-transpose (OP_C) bracketed by
 *     conjugating the target B row (the reference's ZLACGV + ZGEMV('C') + ZLACGV).
 * A host loop walks the pivot sequence (read once device->host); only that read
 * synchronizes.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv and B are
 * caller-provided device pointers. @p d_ipiv is the device pivot array (1-based,
 * the Bunch-Kaufman sign convention). Requires the handle's DEFAULT (host)
 * pointer mode -- the BLAS scalars are host constants, as in sytrs.
 */

module;

#include "hetrs_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

export module calaman.hetrs;

import std;               // std::vector, std::max
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_*, wwrblasGetStream
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;       // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas; // geru, gemv, swap
import calaman.common;    // Uplo, complex_fp, kOne, kNegativeOne

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Solve A X = B for a Bunch-Kaufman-factored Hermitian A (LAPACK ?hetrs)
///
/// Overwrites @p d_B with X, using the factor in @p d_A and the pivots in
/// @p d_ipiv that ?hetrf produced for the same @p uplo. Complex only. The handle
/// must be in host pointer mode and bound to the stream all work is enqueued on.
///
/// @tparam T Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param handle GPU BLAS handle in host pointer mode; A, d_ipiv and B live on its device
/// @param uplo Which triangle of A holds the factor (as passed to the factorization)
/// @param n Order of the Hermitian matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; the Bunch-Kaufman factor, read only
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device pivot array, length n; the 1-based sequence ?hetrf wrote
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<complex_fp T>
Status hetrs(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, const int nrhs,
             const T *d_A, const int lda, const int *d_ipiv, T *d_B, const int ldb) {
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0 || nrhs == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  std::vector<int> piv(static_cast<std::size_t>(n));
  CLM_TRY(wwr::wwrMemcpyAsync(piv.data(), d_ipiv, static_cast<std::size_t>(n) * sizeof(int),
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  using std::size_t;
  const T *const neg_one = &kNegativeOne<T>;
  const T *const one = &kOne<T>;
  const bool upper = uplo == Uplo::U;

  // Conjugate the target B row, gemv('C'), conjugate back -- the reference's
  // ZLACGV/ZGEMV('C')/ZLACGV, so a plain conjugate-transpose gemv realizes the
  // Hermitian U^H / L^H update. m is the inner dimension (rows of B / A column).
  const auto cgemv = [&](int m, const T *A_mat, const T *x, T *y) -> Status {
    device::hetrs_conj_row<T>(stream, y, ldb, nrhs);
    CLM_TRY(
        wwr::gemv<T>(handle, wwr::WWRBLAS_OP_C, m, nrhs, neg_one, A_mat, ldb, x, 1, one, y, ldb));
    device::hetrs_conj_row<T>(stream, y, ldb, nrhs);
    return wwr::WWRBLAS_STATUS_SUCCESS;
  };

  if (upper) {
    // ── Solve U D X = B (backward) ───────────────────────────────────────
    int k = n - 1;
    while (k >= 0) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        if (k > 0) {
          CLM_TRY(wwr::geru<T, int>(handle, k, nrhs, neg_one, d_A + static_cast<size_t>(k) * lda, 1,
                                    d_B + k, ldb, d_B, ldb));
        }
        device::hetrs_scale_row_real<T>(stream, d_B + k, ldb, nrhs,
                                        d_A + static_cast<size_t>(k) * lda + k);
        k -= 1;
      } else { // 2x2, block (k-1, k)
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k - 1) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + (k - 1), ldb, d_B + kp, ldb));
        }
        if (k - 1 > 0) {
          CLM_TRY(wwr::geru<T, int>(handle, k - 1, nrhs, neg_one,
                                    d_A + static_cast<size_t>(k) * lda, 1, d_B + k, ldb, d_B, ldb));
          CLM_TRY(wwr::geru<T, int>(handle, k - 1, nrhs, neg_one,
                                    d_A + static_cast<size_t>(k - 1) * lda, 1, d_B + (k - 1), ldb,
                                    d_B, ldb));
        }
        device::hetrs_solve_2x2<T>(stream, d_B + (k - 1), d_B + k, ldb, nrhs,
                                   d_A + static_cast<size_t>(k - 1) * lda + (k - 1),
                                   d_A + static_cast<size_t>(k) * lda + (k - 1),
                                   d_A + static_cast<size_t>(k) * lda + k, /*top_conj=*/false);
        k -= 2;
      }
    }

    // ── Solve U^H X = B (forward) ────────────────────────────────────────
    k = 0;
    while (k < n) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1
        if (k > 0) {
          CLM_TRY(cgemv(k, d_B, d_A + static_cast<size_t>(k) * lda, d_B + k));
        }
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k += 1;
      } else { // 2x2, block (k, k+1)
        if (k > 0) {
          CLM_TRY(cgemv(k, d_B, d_A + static_cast<size_t>(k) * lda, d_B + k));
          CLM_TRY(cgemv(k, d_B, d_A + static_cast<size_t>(k + 1) * lda, d_B + (k + 1)));
        }
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k += 2;
      }
    }
  } else {
    // ── Solve L D X = B (forward) ────────────────────────────────────────
    int k = 0;
    while (k < n) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        if (k < n - 1) {
          CLM_TRY(wwr::geru<T, int>(handle, n - k - 1, nrhs, neg_one,
                                    d_A + static_cast<size_t>(k) * lda + (k + 1), 1, d_B + k, ldb,
                                    d_B + (k + 1), ldb));
        }
        device::hetrs_scale_row_real<T>(stream, d_B + k, ldb, nrhs,
                                        d_A + static_cast<size_t>(k) * lda + k);
        k += 1;
      } else { // 2x2, block (k, k+1)
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k + 1) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + (k + 1), ldb, d_B + kp, ldb));
        }
        if (k < n - 2) {
          CLM_TRY(wwr::geru<T, int>(handle, n - k - 2, nrhs, neg_one,
                                    d_A + static_cast<size_t>(k) * lda + (k + 2), 1, d_B + k, ldb,
                                    d_B + (k + 2), ldb));
          CLM_TRY(wwr::geru<T, int>(handle, n - k - 2, nrhs, neg_one,
                                    d_A + static_cast<size_t>(k + 1) * lda + (k + 2), 1,
                                    d_B + (k + 1), ldb, d_B + (k + 2), ldb));
        }
        device::hetrs_solve_2x2<T>(stream, d_B + k, d_B + (k + 1), ldb, nrhs,
                                   d_A + static_cast<size_t>(k) * lda + k,
                                   d_A + static_cast<size_t>(k) * lda + (k + 1),
                                   d_A + static_cast<size_t>(k + 1) * lda + (k + 1),
                                   /*top_conj=*/true);
        k += 2;
      }
    }

    // ── Solve L^H X = B (backward) ───────────────────────────────────────
    k = n - 1;
    while (k >= 0) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1
        if (k < n - 1) {
          CLM_TRY(cgemv(n - k - 1, d_B + (k + 1), d_A + static_cast<size_t>(k) * lda + (k + 1),
                        d_B + k));
        }
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k -= 1;
      } else { // 2x2, block (k-1, k)
        if (k < n - 1) {
          CLM_TRY(cgemv(n - k - 1, d_B + (k + 1), d_A + static_cast<size_t>(k) * lda + (k + 1),
                        d_B + k));
          CLM_TRY(cgemv(n - k - 1, d_B + (k + 1), d_A + static_cast<size_t>(k - 1) * lda + (k + 1),
                        d_B + (k - 1)));
        }
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k -= 2;
      }
    }
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
