/**
 * @file pstrf.cppm
 * @brief The calaman.pstrf module -- the blocked, level-3 pivoted Cholesky
 *        factorization of a symmetric positive semidefinite matrix, LAPACK's
 *        ?pstrf
 *
 * Factors P^T A P = U^H U (Uplo::U) or L L^H (Uplo::L) with complete (symmetric)
 * pivoting, overwriting the referenced triangle of A the LAPACK way. Templated
 * over `float` and `double`. See README.md for the blocked recipe.
 *
 * Two paths, split on an internal block size nb (issue #65, the reference
 * ?pstrf structure -- NOT the laqps->laqp2 per-panel delegation):
 *   * nb <= 1 || nb >= n: hand the whole matrix to calaman.pstf2, the unblocked
 *     level-2 panel -- small n or a degenerate block size.
 *   * otherwise: the INLINED blocked path. Each block factors up to nb columns
 *     with a fused per-column pivot look-ahead (gemv + scal + swap over the
 *     running Schur-complement diagonals), then applies one syrk to the trailing
 *     block across the block boundary. The look-ahead is fused into the panel
 *     loop and inlined here, not delegated to pstf2 per panel.
 *
 * RANK-REVEALING, not positive-definite breakdown: the step whose pivot Schur
 * diagonal falls at or below @p tol stops the factorization, reports the number
 * of completed steps in @p rank, zeroes the trailing factor, and returns through
 * @p info = rank + 1 as SUCCESS (the calaman ?pst* convention pstf2 lands,
 * matched here). A NaN pivot aborts the same way. The Status return reports
 * device or BLAS failure only; the numerical outcome rides @p rank / @p info.
 *
 * Allocation-free shipped surface (CLAUDE.md): A and @p work are caller-provided
 * device pointers; @p piv / @p rank / @p info are host (the pivot index is needed
 * host-side each step). Requires the handle's DEFAULT (host) pointer mode, like
 * larfg/laqp2/pstf2. Complex ?pstrf is a deliberate later extension.
 *
 * Usage:
 *   import calaman.pstrf;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   import calaman.common;    // calaman::Uplo
 *   // d_A: n x n device matrix, lda; d_work: length >= n + 3; piv: host int[n]
 *   int rank = 0, info = 0;
 *   const double tol = -1.0; // negative: use the default N*eps*max diagonal
 *   calaman::pstrf<double>(handle, calaman::Uplo::L, n, d_A, lda, piv.data(),
 *                          &rank, &info, tol, d_work);
 */

module;

#include "pstrf_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.pstrf;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_OP_*, WWRBLAS_FILL_MODE_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrMemsetAsync, wwrStreamSynchronize
import wwr.wrappers.blas; // swap, scal, gemv, syrk
import calaman.common;    // calaman::Uplo
import calaman.pstf2;     // calaman::pstf2 -- the unblocked fallback path
import std;               // std::sqrt, std::numeric_limits, std::min

// export import, not a plain import: pstrf RETURNS calaman::Status, so a consumer
// of `import calaman.pstrf;` must see Status's member functions, not just its
// name -- the same re-export laqp2/laqps/pstf2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// pstrf_detail, not an anonymous namespace: these helpers are named by the
// exported pstrf template's body, which is instantiated in every importer's TU.
// A named (unexported) namespace gives them module linkage, reachable by the
// instantiation yet absent from the public surface (as laqp2/laqps/pstf2 do).
// The name is MODULE-SPECIFIC so a consumer importing pstrf alongside pstf2 --
// which defines identically-shaped scalar helpers -- sees no one name on two
// modules.
namespace pstrf_detail {

/// @brief Write one device scalar from a host value on @p stream, blocking
template<typename T>
Status set_device_scalar(wwr::wwrStream_t stream, T *dst, const T &value) {
  CLM_TRY(wwr::wwrMemcpyAsync(dst, &value, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Copy one device scalar to another (device-to-device) on @p stream, blocking
template<typename T>
Status copy_device_scalar(wwr::wwrStream_t stream, T *dst, const T *src) {
  CLM_TRY(wwr::wwrMemcpyAsync(dst, src, sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Swap two single device scalars through a host staging pair, on @p stream
template<typename T>
Status swap_device_scalar(wwr::wwrStream_t stream, T *a, T *b) {
  T ha{};
  T hb{};
  CLM_TRY(wwr::wwrMemcpyAsync(&ha, a, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&hb, b, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  CLM_TRY(wwr::wwrMemcpyAsync(a, &hb, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(b, &ha, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Read the pivot kernel's three-element output back to the host, blocking
///
/// @p out points at the device buffer pstrf_pivot wrote: [max diagonal, index,
/// NaN flag]. The index is stored as T and truncated back to int here.
template<typename T>
Status read_pivot(wwr::wwrStream_t stream, const T *out, T *max_val, int *pos, bool *is_nan) {
  T host[3]{};
  CLM_TRY(wwr::wwrMemcpyAsync(host, out, 3 * sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  *max_val = host[0];
  *pos = static_cast<int>(host[1]);
  *is_nan = host[2] != T{0};
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief The symmetric pivot swap: bring row/column @p pvt to position @p j
///
/// On a triangle-only store this is the reference ?pst* dance -- a diagonal-corner
/// copy plus three strided BLAS swaps that reflect the off-diagonals across the
/// diagonal -- plus the running dot product and the 1-based piv entry. Permutes
/// the ENTIRE stored column/row (prior blocks included), so it is correct inside
/// the blocked panel too. No-op when @p pvt == @p j.
template<typename T>
Status pivot_swap(wwr::wwrblasHandle_t handle, wwr::wwrStream_t stream, bool upper, int n, int j,
                  int pvt, T *A, int lda, T *dots, int *piv) {
  if (pvt == j) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  CLM_TRY(copy_device_scalar<T>(stream, A + static_cast<std::size_t>(pvt) * lda + pvt,
                                A + static_cast<std::size_t>(j) * lda + j));
  if (upper) {
    if (j > 0) {
      CLM_TRY(wwr::swap<T, int>(handle, j, A + static_cast<std::size_t>(j) * lda, 1,
                                A + static_cast<std::size_t>(pvt) * lda, 1));
    }
    if (pvt < n - 1) {
      CLM_TRY(wwr::swap<T, int>(handle, n - 1 - pvt,
                                A + static_cast<std::size_t>(pvt + 1) * lda + j, lda,
                                A + static_cast<std::size_t>(pvt + 1) * lda + pvt, lda));
    }
    if (pvt - j - 1 > 0) {
      CLM_TRY(wwr::swap<T, int>(handle, pvt - j - 1, A + static_cast<std::size_t>(j + 1) * lda + j,
                                lda, A + static_cast<std::size_t>(pvt) * lda + (j + 1), 1));
    }
  } else {
    if (j > 0) {
      CLM_TRY(wwr::swap<T, int>(handle, j, A + j, lda, A + pvt, lda));
    }
    if (pvt < n - 1) {
      CLM_TRY(wwr::swap<T, int>(handle, n - 1 - pvt,
                                A + static_cast<std::size_t>(j) * lda + (pvt + 1), 1,
                                A + static_cast<std::size_t>(pvt) * lda + (pvt + 1), 1));
    }
    if (pvt - j - 1 > 0) {
      CLM_TRY(wwr::swap<T, int>(handle, pvt - j - 1,
                                A + static_cast<std::size_t>(j) * lda + (j + 1), 1,
                                A + static_cast<std::size_t>(j + 1) * lda + pvt, lda));
    }
  }
  CLM_TRY(swap_device_scalar<T>(stream, dots + j, dots + pvt));
  const int tmp = piv[j];
  piv[j] = piv[pvt];
  piv[pvt] = tmp;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace pstrf_detail

// Internal block size for the level-3 path. Small on purpose: large enough to
// turn the trailing update into a real level-3 syrk, small enough that modest
// test shapes span several blocks (issue #65 requires exercising a non-multiple
// of nb). A different nb from the reference LAPACK's ilaenv choice changes only
// the floating-point accumulation order, not the unique pivoted factorization.
inline constexpr int kPstrfBlockSize = 32;

/// @brief Blocked pivoted Cholesky of a symmetric PSD matrix (LAPACK ?pstrf)
///
/// Factors P^T A P = U^H U (@p uplo == Uplo::U) or L L^H (Uplo::L) in place with
/// complete pivoting, overwriting the referenced triangle with the factor and
/// recording the symmetric permutation in @p piv. Rank-revealing: stops at the
/// first pivot Schur diagonal <= the stopping value (@p tol, or N*eps*max_k A(k,k)
/// when @p tol < 0), writing the completed-step count to @p rank and signalling
/// the stop through @p info. A NaN pivot aborts identically. Small n or a
/// degenerate block size falls back to the unblocked calaman.pstf2.
///
/// @p info is 0 on a full-rank factorization (@p rank == n), else @p rank + 1 (a
/// positive success code, not a failure). The returned Status reports only device
/// or BLAS errors, and the factorization stops at the first failing step.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; A and work live on its device
/// @param uplo Which triangle of A is referenced and holds the factor
/// @param n Order of the symmetric matrix A
/// @param A Device matrix, n by n, column-major; its @p uplo triangle is overwritten
/// @param lda Leading dimension of A (>= n)
/// @param piv Host int array, length n; filled with the 1-based symmetric permutation
/// @param rank Host out: the number of completed steps (the computed rank)
/// @param info Host out: 0 if full rank, else @p rank + 1 (a positive success code)
/// @param tol Stopping value; when negative, the default N*eps*max_k A(k,k) is used
/// @param work Device workspace, length >= n + 3 (the Schur diagonals and pivot scratch)
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status pstrf(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, T *A, const int lda,
             int *piv, int *rank, int *info, const T tol, T *work) {
  *rank = 0;
  *info = 0;
  if (n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Unblocked fallback: a degenerate block size or a matrix no larger than one
  // block is the reference's IF( NB <= 1 .OR. NB >= N ) path -- hand it to the
  // level-2 panel wholesale (issue #65).
  constexpr int nb = kPstrfBlockSize;
  if (nb <= 1 || nb >= n) {
    return pstf2<T>(handle, uplo, n, A, lda, piv, rank, info, tol, work);
  }

  // Order every scalar read/write on the handle's own stream, so they follow the
  // caller's uploads and this routine's BLAS work -- the larfg/laqp2/pstf2 discipline.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const bool upper = uplo == Uplo::U;
  using std::size_t;

  // work layout: dots[0:n] the running (within-block) factor dot products,
  // out[n:n+3] the pivot kernel's (max diagonal, index, NaN flag) read back.
  T *dots = work;
  T *out = work + n;
  CLM_TRY(wwr::wwrMemsetAsync(dots, 0, static_cast<size_t>(n) * sizeof(T), stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  // piv starts as the identity permutation (1-based); each pivot swaps two entries.
  for (int i = 0; i < n; ++i) {
    piv[i] = i + 1;
  }

  // Initial pass over every diagonal: the global max sets both the j == 0 pivot
  // and the default stopping value. A non-positive or NaN max means A is not
  // positive (semi)definite -- rank 0, info 1 (== rank + 1), trailing factor zero.
  device::pstrf_pivot<T>(stream, 0, n, /*add_prev=*/false, static_cast<const T *>(nullptr),
                         size_t{0}, A, static_cast<size_t>(lda), dots, out);
  T ajj{};
  int pvt = 0;
  bool is_nan = false;
  CLM_TRY(pstrf_detail::read_pivot<T>(stream, out, &ajj, &pvt, &is_nan));
  if (is_nan || ajj <= T{0}) {
    *rank = 0;
    *info = 1;
    device::pstrf_zero_trailing<T>(stream, upper, 0, n, A, static_cast<size_t>(lda));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // DLAMCH('E') is eps/2, not std::numeric_limits::epsilon(), so the default
  // stopping value matches the reference LAPACK oracle's bit-for-bit.
  const T macheps = std::numeric_limits<T>::epsilon() * T{0.5};
  const T dstop = tol < T{0} ? static_cast<T>(n) * macheps * ajj : tol;

  const T one = T{1};
  const T neg_one = T{-1};

  // Blocked loop: each block factors up to nb columns, then one syrk pushes the
  // block's effect onto the trailing submatrix across the block boundary.
  for (int k = 0; k < n; k += nb) {
    const int jb = std::min(nb, n - k);

    // The running dot products are WITHIN-BLOCK: reset [k, n) so each block
    // accumulates only its own factor rows/columns; the prior blocks' subtraction
    // already lives in A's diagonal via the completed trailing syrks.
    CLM_TRY(wwr::wwrMemsetAsync(dots + k, 0, static_cast<size_t>(n - k) * sizeof(T), stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));

    for (int j = k; j < k + jb; ++j) {
      if (j > 0) {
        // Refresh the (within-block) Schur diagonals with the previous column's
        // factor and pick this step's pivot. add_prev is false on a block's first
        // column (j == k), where there is no prior in-block factor row/column.
        const bool add_prev = j > k;
        const T *prev = upper ? A + (j - 1) : A + static_cast<size_t>(j - 1) * lda;
        const size_t prev_stride = upper ? static_cast<size_t>(lda) : size_t{1};
        device::pstrf_pivot<T>(stream, j, n, add_prev, prev, prev_stride, A,
                               static_cast<size_t>(lda), dots, out);
        CLM_TRY(pstrf_detail::read_pivot<T>(stream, out, &ajj, &pvt, &is_nan));
        if (is_nan || ajj <= dstop) {
          *rank = j;
          *info = j + 1;
          device::pstrf_zero_trailing<T>(stream, upper, j, n, A, static_cast<size_t>(lda));
          CLM_TRY(wwr::wwrStreamSynchronize(stream));
          return wwr::WWRBLAS_STATUS_SUCCESS;
        }
      }

      CLM_TRY(pstrf_detail::pivot_swap<T>(handle, stream, upper, n, j, pvt, A, lda, dots, piv));

      // Take the pivot's square root and write the factor diagonal.
      ajj = std::sqrt(ajj);
      CLM_TRY(pstrf_detail::set_device_scalar<T>(stream, A + static_cast<size_t>(j) * lda + j, ajj));

      // Left-looking update of the pivot row (upper) or column (lower) against the
      // CURRENT BLOCK's already-computed factor (rows/cols [k, j)), then scale by
      // 1/ajj. The inner dimension is j - k, not j: prior blocks were applied by
      // syrk. This is the reference blocked gemv + scal.
      const int trail = n - j - 1;
      if (trail > 0) {
        const T inv = one / ajj;
        if (upper) {
          if (j > k) {
            CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, j - k, trail, &neg_one,
                                 A + static_cast<size_t>(j + 1) * lda + k, lda,
                                 A + static_cast<size_t>(j) * lda + k, 1, &one,
                                 A + static_cast<size_t>(j + 1) * lda + j, lda));
          }
          CLM_TRY(wwr::scal<T, int>(handle, trail, &inv, A + static_cast<size_t>(j + 1) * lda + j,
                                    lda));
        } else {
          if (j > k) {
            CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, trail, j - k, &neg_one,
                                 A + static_cast<size_t>(k) * lda + (j + 1), lda,
                                 A + static_cast<size_t>(k) * lda + j, lda, &one,
                                 A + static_cast<size_t>(j) * lda + (j + 1), 1));
          }
          CLM_TRY(wwr::scal<T, int>(handle, trail, &inv, A + static_cast<size_t>(j) * lda + (j + 1),
                                    1));
        }
      }
    }

    // Trailing-block update across the block boundary: subtract the just-finished
    // block's outer product from A(kb:n, kb:n), the stored triangle only. One
    // level-3 syrk -- the reason ?pstrf blocks at all.
    const int kb = k + jb;
    if (kb < n) {
      const wwr::wwrblasFillMode_t fill =
          upper ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
      if (upper) {
        CLM_TRY(wwr::syrk<T, int>(handle, fill, wwr::WWRBLAS_OP_T, n - kb, jb, &neg_one,
                                  A + static_cast<size_t>(kb) * lda + k, lda, &one,
                                  A + static_cast<size_t>(kb) * lda + kb, lda));
      } else {
        CLM_TRY(wwr::syrk<T, int>(handle, fill, wwr::WWRBLAS_OP_N, n - kb, jb, &neg_one,
                                  A + static_cast<size_t>(k) * lda + kb, lda, &one,
                                  A + static_cast<size_t>(kb) * lda + kb, lda));
      }
    }
  }

  *rank = n;
  *info = 0;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
