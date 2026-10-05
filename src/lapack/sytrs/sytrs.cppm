/**
 * @file sytrs.cppm
 * @brief The calaman.sytrs module -- solve A X = B for a symmetric matrix already
 *        factored by the Bunch-Kaufman method, LAPACK's ?sytrs
 *
 * Given the factor A = U D U^T (Uplo::U) or L D L^T (Uplo::L) and the pivot
 * sequence @p d_ipiv that ?sytrf / wwr::sytrf produced, overwrites the
 * right-hand sides @p d_B with the solution X. Templated over the four element
 * types; COMPLEX IS SYMMETRIC, not Hermitian (no conjugation) -- the Hermitian
 * cousin is calaman.hetrs.
 *
 * Structure follows the reference level-2 ?sytrs exactly: a host loop walks the
 * pivot sequence (read once from device to host), issuing one wrapped BLAS call
 * per step -- ger/geru for the unit-triangular rank-1 updates, gemv for the
 * transpose-solve back-substitution, swap for the row interchanges -- plus the
 * two D^-1 block applies that no BLAS expresses (sytrs.cu). Everything is
 * enqueued on the handle's stream; only the single ipiv read synchronizes.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv and B are
 * caller-provided device pointers; the only host memory is the n-int pivot copy
 * this routine stages to drive its loop. @p d_ipiv is the device pivot array
 * wwr::sytrf wrote (1-based, the Bunch-Kaufman sign convention). Requires the
 * handle's DEFAULT (host) pointer mode -- the BLAS scalars are host constants,
 * as in pstrf.
 */

module;

#include "sytrs_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import supplies.
#include "error_handling/error_macros.h"

export module calaman.sytrs;

import std;               // std::vector, std::max
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_*, wwrblasGetStream
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;       // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas; // ger, geru, gemv, swap
import calaman.common;    // Uplo, usual_fp, real_fp, kOne, kNegativeOne

// export import, not a plain import: sytrs RETURNS calaman::Status, so a consumer
// of `import calaman.sytrs;` must see Status's member functions -- the re-export
// pstrf/laqp2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Named (unexported) namespace, not anonymous: these helpers are named by the
// exported sytrs template's body, instantiated in every importer's TU, so they
// need module linkage reachable from the instantiation yet absent from the
// public surface (the pstrf_detail arrangement).
namespace sytrs_detail {

/// @brief Rank-1 update B <- B + alpha x y^T, dispatching ger (real) / geru (complex)
///
/// The unit-triangular solve's DGER has no single wwr spelling: ger is real-only,
/// geru the unconjugated complex one (gerc would conjugate, which the SYMMETRIC
/// solve must not). One `if constexpr` picks the right name off the element type.
template<typename T>
Status rank1(wwr::wwrblasHandle_t handle, int m, int n, const T *alpha, const T *x, int incx,
             const T *y, int incy, T *A, int lda) {
  if constexpr (real_fp<T>) {
    return wwr::ger<T, int>(handle, m, n, alpha, x, incx, y, incy, A, lda);
  } else {
    return wwr::geru<T, int>(handle, m, n, alpha, x, incx, y, incy, A, lda);
  }
}

} // namespace sytrs_detail

/// @brief Solve A X = B for a Bunch-Kaufman-factored symmetric A (LAPACK ?sytrs)
///
/// Overwrites @p d_B with X, using the factor in @p d_A and the pivots in
/// @p d_ipiv that wwr::sytrf produced for the same @p uplo. Complex is symmetric
/// (no conjugation). The handle must be in host pointer mode and bound to the
/// stream all work is enqueued on.
///
/// @tparam T Element type; one of the four usual_fp types
/// @param handle GPU BLAS handle in host pointer mode; A, d_ipiv and B live on its device
/// @param uplo Which triangle of A holds the factor (as passed to the factorization)
/// @param n Order of the symmetric matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; the Bunch-Kaufman factor, read only
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device pivot array, length n; the 1-based sequence wwr::sytrf wrote
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<usual_fp T>
Status sytrs(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, const int nrhs,
             const T *d_A, const int lda, const int *d_ipiv, T *d_B, const int ldb) {
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0 || nrhs == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Order every stream op on the handle's own stream, so the kernels and BLAS
  // follow the caller's uploads -- the pstrf discipline.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // The pivot sequence drives host control flow, so read it once (the only sync).
  std::vector<int> piv(static_cast<std::size_t>(n));
  CLM_TRY(wwr::wwrMemcpyAsync(piv.data(), d_ipiv, static_cast<std::size_t>(n) * sizeof(int),
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  using std::size_t;
  const T *const neg_one = &kNegativeOne<T>;
  const T *const one = &kOne<T>;
  const bool upper = uplo == Uplo::U;

  if (upper) {
    // ── Solve U D X = B (backward over the columns) ──────────────────────
    int k = n - 1;
    while (k >= 0) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1 pivot
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        if (k > 0) {
          CLM_TRY(sytrs_detail::rank1<T>(handle, k, nrhs, neg_one,
                                         d_A + static_cast<size_t>(k) * lda, 1, d_B + k, ldb, d_B,
                                         ldb));
        }
        device::sytrs_scale_row<T>(stream, d_B + k, ldb, nrhs,
                                   d_A + static_cast<size_t>(k) * lda + k);
        k -= 1;
      } else { // 2x2 pivot, block (k-1, k)
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k - 1) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + (k - 1), ldb, d_B + kp, ldb));
        }
        if (k - 1 > 0) {
          CLM_TRY(sytrs_detail::rank1<T>(handle, k - 1, nrhs, neg_one,
                                         d_A + static_cast<size_t>(k) * lda, 1, d_B + k, ldb, d_B,
                                         ldb));
          CLM_TRY(sytrs_detail::rank1<T>(handle, k - 1, nrhs, neg_one,
                                         d_A + static_cast<size_t>(k - 1) * lda, 1, d_B + (k - 1),
                                         ldb, d_B, ldb));
        }
        device::sytrs_solve_2x2<T>(stream, d_B + (k - 1), d_B + k, ldb, nrhs,
                                   d_A + static_cast<size_t>(k - 1) * lda + (k - 1),
                                   d_A + static_cast<size_t>(k) * lda + (k - 1),
                                   d_A + static_cast<size_t>(k) * lda + k);
        k -= 2;
      }
    }

    // ── Solve U^T X = B (forward over the columns) ───────────────────────
    k = 0;
    while (k < n) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1 pivot
        if (k > 0) {
          CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, k, nrhs, neg_one, d_B, ldb,
                               d_A + static_cast<size_t>(k) * lda, 1, one, d_B + k, ldb));
        }
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k += 1;
      } else { // 2x2 pivot, block (k, k+1)
        if (k > 0) {
          CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, k, nrhs, neg_one, d_B, ldb,
                               d_A + static_cast<size_t>(k) * lda, 1, one, d_B + k, ldb));
          CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, k, nrhs, neg_one, d_B, ldb,
                               d_A + static_cast<size_t>(k + 1) * lda, 1, one, d_B + (k + 1), ldb));
        }
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k += 2;
      }
    }
  } else {
    // ── Solve L D X = B (forward over the columns) ───────────────────────
    int k = 0;
    while (k < n) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1 pivot
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        if (k < n - 1) {
          CLM_TRY(sytrs_detail::rank1<T>(handle, n - k - 1, nrhs, neg_one,
                                         d_A + static_cast<size_t>(k) * lda + (k + 1), 1, d_B + k,
                                         ldb, d_B + (k + 1), ldb));
        }
        device::sytrs_scale_row<T>(stream, d_B + k, ldb, nrhs,
                                   d_A + static_cast<size_t>(k) * lda + k);
        k += 1;
      } else { // 2x2 pivot, block (k, k+1)
        const int kp = -piv[static_cast<size_t>(k)] - 1;
        if (kp != k + 1) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + (k + 1), ldb, d_B + kp, ldb));
        }
        if (k < n - 2) {
          CLM_TRY(sytrs_detail::rank1<T>(handle, n - k - 2, nrhs, neg_one,
                                         d_A + static_cast<size_t>(k) * lda + (k + 2), 1, d_B + k,
                                         ldb, d_B + (k + 2), ldb));
          CLM_TRY(sytrs_detail::rank1<T>(handle, n - k - 2, nrhs, neg_one,
                                         d_A + static_cast<size_t>(k + 1) * lda + (k + 2), 1,
                                         d_B + (k + 1), ldb, d_B + (k + 2), ldb));
        }
        device::sytrs_solve_2x2<T>(stream, d_B + k, d_B + (k + 1), ldb, nrhs,
                                   d_A + static_cast<size_t>(k) * lda + k,
                                   d_A + static_cast<size_t>(k) * lda + (k + 1),
                                   d_A + static_cast<size_t>(k + 1) * lda + (k + 1));
        k += 2;
      }
    }

    // ── Solve L^T X = B (backward over the columns) ──────────────────────
    k = n - 1;
    while (k >= 0) {
      if (piv[static_cast<size_t>(k)] > 0) { // 1x1 pivot
        if (k < n - 1) {
          CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, n - k - 1, nrhs, neg_one, d_B + (k + 1),
                               ldb, d_A + static_cast<size_t>(k) * lda + (k + 1), 1, one, d_B + k,
                               ldb));
        }
        const int kp = piv[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(handle, nrhs, d_B + k, ldb, d_B + kp, ldb));
        }
        k -= 1;
      } else { // 2x2 pivot, block (k-1, k)
        if (k < n - 1) {
          CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, n - k - 1, nrhs, neg_one, d_B + (k + 1),
                               ldb, d_A + static_cast<size_t>(k) * lda + (k + 1), 1, one, d_B + k,
                               ldb));
          CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, n - k - 1, nrhs, neg_one, d_B + (k + 1),
                               ldb, d_A + static_cast<size_t>(k - 1) * lda + (k + 1), 1, one,
                               d_B + (k - 1), ldb));
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
