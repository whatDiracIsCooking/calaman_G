/**
 * @file lahr2.cppm
 * @brief The calaman.lahr2 module -- reduce the first nb columns of a panel to
 *        upper Hessenberg form, LAPACK's ?lahr2
 *
 * Reduces the first @p nb columns of an n-by-(n-k+1) matrix A so that elements
 * below the @p k-th subdiagonal are zero, by an orthogonal similarity
 * Q^T A Q with Q = H(1) H(2) ... H(nb) a product of nb Householder reflectors.
 * Returns the reflector scalars tau, the triangular factor T (so
 * Q = I - V T V^T), and the auxiliary Y = A V T that ?gehrd needs to apply the
 * block to the unreduced trailing matrix. The panel driver below ?gehrd. This
 * is the LAPACK-3.0+ ?lahr2 (the Quintana-Orti / Van de Geijn variant), not the
 * backward-incompatible ?lahrd it replaced.
 *
 * Pure host over wrapped BLAS -- gemv/trmv (the per-column reflector apply and
 * the T/Y builds), gemm/trmm (the final Y(1:k,:) block) and a geam for the one
 * ?lacpy block copy -- plus calaman.larfg per column for the reflectors. NO
 * custom kernel, the shape of calaman.larft. The handful of single-element
 * writes that set a reflector's unit entry (and restore the Hessenberg
 * subdiagonal afterwards) go through a blocking host->device scalar store, the
 * granularity calaman.laqps already imposes.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/larft: every
 * BLAS scalar is a host address. A, tau, T and Y are caller-owned device
 * pointers; nothing is allocated. Assumes 1 <= nb <= n - k (the panel fits), as
 * ?gehrd guarantees. Templated over float and double; complex is a later
 * extension, the wall larfg/larft document.
 *
 * Usage:
 *   import calaman.lahr2;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   // d_A: n x (n-k+1); d_tau: length nb; d_T: nb x nb (ldt); d_Y: n x nb (ldy)
 *   calaman::lahr2<double>(handle, n, k, nb, d_A, lda, d_tau, d_T, ldt, d_Y, ldy);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.lahr2;

import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_* / _SIDE_* / _FILL_* / _DIAG_*, status
import wwr.runtime_api;   // wwrMemcpyAsync, wwrMemcpyHostToDevice, wwrStreamSynchronize
import wwr.wrappers.blas; // gemv, trmv, gemm, trmm, geam, copy, axpy, scal
import calaman.larfg;     // calaman::larfg -- the per-column reflector

// export import, not a plain import: lahr2 RETURNS calaman::Status, so a consumer
// of `import calaman.lahr2;` must see Status's member functions, not just its
// name -- the same re-export larfg/larft do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// lahr2_detail, not a bare detail: these helpers are named by the exported lahr2
// template's body, instantiated in every importer's TU, so they need module
// linkage with a MODULE-SPECIFIC name -- calaman.laqps defines an identically
// shaped set_device_scalar, and a consumer importing both would otherwise see
// one qualified name attached to two modules (ill-formed), the clash laqps_detail
// documents.
namespace lahr2_detail {

/// @brief Write one device scalar from a host value on @p stream, blocking
template<typename T>
Status set_device_scalar(wwr::wwrStream_t stream, T *dst, const T &value) {
  CLM_TRY(wwr::wwrMemcpyAsync(dst, &value, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace lahr2_detail

/// @brief Reduce the first nb columns of a panel to Hessenberg form (LAPACK ?lahr2)
///
/// Overwrites A's first @p nb columns with the reduced Hessenberg entries on and
/// above the @p k-th subdiagonal and the reflector tails below it, fills @p tau
/// with the nb reflector scalars, @p T with the nb-by-nb triangular factor (so
/// Q = I - V T V^T), and @p Y with the n-by-nb auxiliary A V T. Columns of A
/// after the first nb are left unchanged, as the reference leaves them.
///
/// Short-circuits: the first failing BLAS / larfg / scalar-store Status is
/// returned and the reduction stops there. Returns success and writes nothing
/// when @p n <= 1 or @p nb <= 0.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; all device pointers live on its device
/// @param n Order of the matrix A
/// @param k Offset of the reduction: entries below the @p k-th subdiagonal are zeroed (0 <= k < n)
/// @param nb Number of columns to reduce (1 <= nb <= n - k)
/// @param A Device matrix, n by (n-k+1), column-major; overwritten with the reduction + reflectors
/// @param lda Leading dimension of A (>= max(1, n))
/// @param tau Device array, length >= nb; the reflector scalars (out)
/// @param T_ Device matrix, nb by nb, column-major; the triangular factor (out)
/// @param ldt Leading dimension of T_ (>= nb)
/// @param Y Device matrix, n by nb, column-major; the auxiliary Y = A V T (out)
/// @param ldy Leading dimension of Y (>= max(1, n))
/// @return The Status of the first failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status lahr2(wwr::wwrblasHandle_t handle, const int n, const int k, const int nb, T *A,
             const int lda, T *tau, T *T_, const int ldt, T *Y, const int ldy) {
  // Quick return: the reference bails when n <= 1; nb <= 0 leaves every output
  // untouched (and guards the post-loop subdiagonal restore at A(k+nb, nb)).
  if (n <= 1 || nb <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  const T one = T{1};
  const T zero = T{0};
  const T neg_one = T{-1};

  // t(1,nb) -- the last column of T -- is the reference's length-(i-1) scratch
  // vector w. It is only read as a true T column at the final step (i == nb),
  // after this scratch use, so reusing it is safe for every earlier step.
  T *const wscr = T_ + static_cast<long>(nb - 1) * ldt;

  // ei carries a reflector's leading R entry (beta) from one column to the next:
  // larfg overwrites A(k+i, i) with beta, the body then sets that entry to 1 (the
  // unit v(1) the T/Y builds need), and the restore below writes beta back once
  // the next column no longer reads the whole reflector. The final beta lands
  // after the loop.
  T ei{};

  // fi is the reference's 1-based column index I; the 0-based pointer math
  // subtracts 1. acol = A(:, i), piv = A(k+i, i), ycol = Y(:, i), tcol = T(:, i).
  for (int fi = 1; fi <= nb; ++fi) {
    T *const acol = A + static_cast<long>(fi - 1) * lda;
    T *const piv = acol + (k + fi - 1);
    T *const ycol = Y + static_cast<long>(fi - 1) * ldy;
    T *const tcol = T_ + static_cast<long>(fi - 1) * ldt;

    if (fi > 1) {
      // Update column i of A - Y V^T: A(k+1:n, i) -= Y(k+1:n, 1:i-1) * (row
      // A(k+i-1, 1:i-1))^T. x is that row, read with stride lda.
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, n - k, fi - 1, &neg_one, Y + k, ldy,
                           A + (k + fi - 2), lda, &one, acol + k, 1));

      // Apply I - V T^T V^T to that column b from the left, w = scratch.
      // w := V1^T b1   (V1 unit lower, the first i-1 reflector rows)
      CLM_TRY(wwr::copy<T>(handle, fi - 1, acol + k, 1, wscr, 1));
      CLM_TRY(wwr::trmv<T>(handle, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_T,
                           wwr::WWRBLAS_DIAG_UNIT, fi - 1, A + k, lda, wscr, 1));
      // w += V2^T b2   (V2 the rectangular tail below row k+i-1)
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, n - k - fi + 1, fi - 1, &one,
                           A + (k + fi - 1), lda, piv, 1, &one, wscr, 1));
      // w := T^T w
      CLM_TRY(wwr::trmv<T>(handle, wwr::WWRBLAS_FILL_MODE_UPPER, wwr::WWRBLAS_OP_T,
                           wwr::WWRBLAS_DIAG_NON_UNIT, fi - 1, T_, ldt, wscr, 1));
      // b2 -= V2 w
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, n - k - fi + 1, fi - 1, &neg_one,
                           A + (k + fi - 1), lda, wscr, 1, &one, piv, 1));
      // b1 -= V1 w
      CLM_TRY(wwr::trmv<T>(handle, wwr::WWRBLAS_FILL_MODE_LOWER, wwr::WWRBLAS_OP_N,
                           wwr::WWRBLAS_DIAG_UNIT, fi - 1, A + k, lda, wscr, 1));
      CLM_TRY(wwr::axpy<T>(handle, fi - 1, &neg_one, wscr, 1, acol + k, 1));

      // Restore the previous column's Hessenberg subdiagonal A(k+i-1, i-1) = beta.
      CLM_TRY(lahr2_detail::set_device_scalar<T>(
          stream, A + static_cast<long>(fi - 2) * lda + (k + fi - 2), ei));
    }

    // Generate H(i) to annihilate A(k+i+1:n, i). larfg overwrites A(k+i, i) with
    // beta, writes tau and beta to host scalars; the tail begins at piv + 1 (the
    // next row) and is empty, left untouched, when the reflector has order 1.
    T host_tau{};
    T host_beta{};
    CLM_TRY(larfg<T>(handle, n - k - fi + 1, piv, piv + 1, 1, &host_tau, &host_beta));
    CLM_TRY(lahr2_detail::set_device_scalar<T>(stream, tau + (fi - 1), host_tau));
    ei = host_beta;

    // Set A(k+i, i) = 1 so piv is the full reflector v with v(1) == 1, the form
    // the Y/T builds below need; restored to beta by the next column (or the
    // post-loop write for the last column).
    CLM_TRY(lahr2_detail::set_device_scalar<T>(stream, piv, one));

    // Y(k+1:n, i) = A(k+1:n, i+1:n-k+1) * v.
    CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, n - k, n - k - fi + 1, &one,
                         A + static_cast<long>(fi) * lda + k, lda, piv, 1, &zero, ycol + k, 1));
    // t(1:i-1, i) = V2^T v  (reflector cross-term), then Y -= Y(:,1:i-1) * t.
    CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, n - k - fi + 1, fi - 1, &one, A + (k + fi - 1),
                         lda, piv, 1, &zero, tcol, 1));
    CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, n - k, fi - 1, &neg_one, Y + k, ldy, tcol, 1,
                         &one, ycol + k, 1));
    CLM_TRY(wwr::scal<T>(handle, n - k, &host_tau, ycol + k, 1));

    // T(1:i, i): t(1:i-1, i) = -tau * T(1:i-1, 1:i-1) * t(1:i-1, i); t(i, i) = tau.
    const T neg_tau = -host_tau;
    CLM_TRY(wwr::scal<T>(handle, fi - 1, &neg_tau, tcol, 1));
    CLM_TRY(wwr::trmv<T>(handle, wwr::WWRBLAS_FILL_MODE_UPPER, wwr::WWRBLAS_OP_N,
                         wwr::WWRBLAS_DIAG_NON_UNIT, fi - 1, T_, ldt, tcol, 1));
    CLM_TRY(lahr2_detail::set_device_scalar<T>(stream, tcol + (fi - 1), host_tau));
  }

  // Restore the last column's Hessenberg subdiagonal A(k+nb, nb) = beta.
  CLM_TRY(lahr2_detail::set_device_scalar<T>(
      stream, A + static_cast<long>(nb - 1) * lda + (k + nb - 1), ei));

  // Y(1:k, 1:nb): copy A(1:k, 2:nb+1), scale by V1 (unit lower), add the trailing
  // block A(1:k, nb+2:n-k+1) * V2, and finally right-multiply by T (upper).
  CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, k, nb, &one, A + lda, lda,
                       &zero, Y, ldy, Y, ldy));
  CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                       wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, k, nb, &one, A + k, lda, Y, ldy,
                       Y, ldy));
  if (n > k + nb) {
    CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, k, nb, n - k - nb, &one,
                         A + static_cast<long>(nb + 1) * lda, lda, A + (k + nb), lda, &one, Y,
                         ldy));
  }
  CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                       wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, k, nb, &one, T_, ldt, Y, ldy,
                       Y, ldy));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
