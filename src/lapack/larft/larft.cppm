/**
 * @file larft.cppm
 * @brief The calaman.larft module -- form the triangular factor T of a block
 *        reflector H = I - V T V^T (LAPACK's ?larft)
 *
 * One recursive host routine (LAPACK's Elmroth-Gustavson ?larft) that builds the
 * k-by-k triangular factor T of an order-n block reflector from its k elementary
 * reflectors (columns or rows of V) and their scalars tau. Pure host over
 * wrapped BLAS-3 (trmm, gemm) plus one geam for the block copy that seeds T's
 * off-diagonal corner -- NO custom kernel, the shape of calaman.larf.
 *
 * @p direct fixes T's shape (Direct::F upper, Direct::B lower) and @p storev how
 * V stores each reflector (StoreV::C columnwise, StoreV::R rowwise); the four
 * combinations are the four branches. The diagonal of T is tau, written at the
 * n==1 or k==1 leaves; each node splits k at l = k/2, recurses on the two
 * diagonal sub-factors, and fills the off-diagonal block from the reflector
 * cross-term (see the per-branch comments for the exact block algebra).
 *
 * Requires the handle's DEFAULT (host) pointer mode: the BLAS scalars are host
 * addresses, like calaman.larf. Assumes k <= n. V, tau and T are device pointers
 * the caller owns; nothing is allocated and the stream is not synchronized.
 * Templated over float and double; complex is a later extension.
 *
 * Usage:
 *   import calaman.larft;   // names calaman::larft, calaman::Direct, StoreV
 *   // d_V: n x k reflectors (col-major); d_tau: length k; d_T: k x k, ldt>=k
 *   calaman::larft<double>(handle, calaman::Direct::F, calaman::StoreV::C,
 *                          n, k, d_V, ldv, d_tau, d_T, ldt);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.larft;

import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_* / _SIDE_* / _FILL_* / _DIAG_*, status
import wwr.wrappers.blas; // gemm, trmm, geam
import wwr.runtime_api;   // wwrMemcpyAsync, wwrMemcpyDeviceToDevice, wwrStream_t
import calaman.common;    // Direct, StoreV (:enums)

// export import, not a plain import: larft RETURNS calaman::Status, so a consumer
// of `import calaman.larft;` must see Status's member functions, not just its
// name -- the same re-export diff_norm/larf do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Direct (F / B) and StoreV (C / R) live in calaman.common's :enums partition.
// Re-export them so `import calaman.larft;` alone names them, as the usage
// example relies on -- the same re-export larf does for calaman::Side.
export using calaman::Direct;
export using calaman::StoreV;

/// @brief Form the k-by-k triangular factor T of a block reflector (LAPACK ?larft)
///
/// Builds T so that H = I - V T V^T (StoreV::C) or H = I - V^T T V (StoreV::R),
/// upper triangular for Direct::F and lower for Direct::B, recursively over the
/// k reflectors in @p V with scalars @p tau. Short-circuits: the first failing
/// BLAS/recursive step's Status is returned. Does nothing and returns success
/// when @p n or @p k is 0. Assumes k <= n.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; V, tau and T live on its device
/// @param direct Reflector order / T shape (Direct::F upper, Direct::B lower)
/// @param storev How V stores the reflectors (StoreV::C columnwise, StoreV::R rowwise)
/// @param n Order of the block reflector H (n >= 0)
/// @param k Number of reflectors = order of T (k >= 1 when n > 0)
/// @param V Device matrix of reflectors, column-major; (ldv,k) for C, (ldv,n) for R
/// @param ldv Leading dimension of V (>= max(1,n) for C, >= k for R)
/// @param tau Device array, length k; the reflector scalars (T's diagonal)
/// @param T_ Device matrix, k by k, column-major; the triangular factor (out)
/// @param ldt Leading dimension of T_ (>= k)
/// @return The Status of the first failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status larft(wwr::wwrblasHandle_t handle, const Direct direct, const StoreV storev, const int n,
             const int k, const T *V, const int ldv, const T *tau, T *T_, const int ldt) {
  if (n == 0 || k == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Base case: a single reflector makes T the 1-by-1 scalar tau(0). An on-stream
  // device-to-device copy -- ordered before every caller's later read, so no sync.
  if (n == 1 || k == 1) {
    wwr::wwrStream_t stream{};
    CLM_TRY(wwr::wwrblasGetStream(handle, &stream));
    CLM_TRY(wwr::wwrMemcpyAsync(T_, tau, sizeof(T), wwr::wwrMemcpyDeviceToDevice, stream));
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  const T one = T{1};
  const T zero = T{0};
  const T neg_one = T{-1};
  const int l = k / 2; // the split point; l >= 1 and k - l >= 1 since k > 1

  const bool dirf = direct == Direct::F;
  const bool colv = storev == StoreV::C;

  if (dirf && colv) {
    // QR: forward + columnwise. T is upper triangular; the off-diagonal block is
    // the (l, k-l) corner T_12, built from the cross-term T_12 = -T_11 (V_1^T V_2) T_22.
    // T_11 = T(0:l, 0:l), T_22 = T(l:k, l:k); V split into columns V_1 (cols 0:l)
    // and V_2 (cols l:k), each unit-lower-trapezoidal.
    CLM_TRY(larft<T>(handle, direct, storev, n, l, V, ldv, tau, T_, ldt));
    CLM_TRY(larft<T>(handle, direct, storev, n - l, k - l, V + static_cast<long>(l) * ldv + l, ldv,
                     tau + l, T_ + static_cast<long>(l) * ldt + l, ldt));

    // T_12 = V_{2,1}^T : transpose the (k-l, l) block V(l:k, 0:l) into T(0:l, l:k).
    CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, l, k - l, &one, V + l, ldv,
                         &zero, T_ + static_cast<long>(l) * ldt, ldt,
                         T_ + static_cast<long>(l) * ldt, ldt));
    // T_12 *= V_{2,2} (unit lower, at V(l:k, l:k)).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, l, k - l, &one,
                         V + static_cast<long>(l) * ldv + l, ldv, T_ + static_cast<long>(l) * ldt,
                         ldt, T_ + static_cast<long>(l) * ldt, ldt));
    // T_12 += V_{3,1}^T V_{3,2} (the rectangular tail V(k:n, 0:l) and V(k:n, l:k)).
    if (n > k) {
      CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, l, k - l, n - k, &one,
                           V + k, ldv, V + static_cast<long>(l) * ldv + k, ldv, &one,
                           T_ + static_cast<long>(l) * ldt, ldt));
    }
    // T_12 = -T_11 * T_12 (T_11 upper, non-unit) then T_12 *= T_22 (upper, non-unit).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_UPPER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &neg_one, T_, ldt,
                         T_ + static_cast<long>(l) * ldt, ldt, T_ + static_cast<long>(l) * ldt,
                         ldt));
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &one,
                         T_ + static_cast<long>(l) * ldt + l, ldt, T_ + static_cast<long>(l) * ldt,
                         ldt, T_ + static_cast<long>(l) * ldt, ldt));
  } else if (dirf && !colv) {
    // LQ: forward + rowwise. T upper triangular; off-diagonal block T_12 = (0:l, l:k).
    // V split into rows V_1 (rows 0:l) and V_2 (rows l:k), each unit-upper-trapezoidal.
    CLM_TRY(larft<T>(handle, direct, storev, n, l, V, ldv, tau, T_, ldt));
    CLM_TRY(larft<T>(handle, direct, storev, n - l, k - l, V + static_cast<long>(l) * ldv + l, ldv,
                     tau + l, T_ + static_cast<long>(l) * ldt + l, ldt));

    // T_12 = V_{1,2} : straight copy of V(0:l, l:k) into T(0:l, l:k).
    CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, l, k - l, &one,
                         V + static_cast<long>(l) * ldv, ldv, &zero,
                         T_ + static_cast<long>(l) * ldt, ldt, T_ + static_cast<long>(l) * ldt,
                         ldt));
    // T_12 *= V_{2,2}^T (V_{2,2} unit upper at V(l:k, l:k)).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                         wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, l, k - l, &one,
                         V + static_cast<long>(l) * ldv + l, ldv, T_ + static_cast<long>(l) * ldt,
                         ldt, T_ + static_cast<long>(l) * ldt, ldt));
    // T_12 += V_{1,3} V_{2,3}^T (the rectangular tail V(0:l, k:n) and V(l:k, k:n)).
    if (n > k) {
      CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, l, k - l, n - k, &one,
                           V + static_cast<long>(k) * ldv, ldv, V + static_cast<long>(k) * ldv + l,
                           ldv, &one, T_ + static_cast<long>(l) * ldt, ldt));
    }
    // T_12 = -T_11 * T_12 then T_12 *= T_22 (both upper, non-unit).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_UPPER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &neg_one, T_, ldt,
                         T_ + static_cast<long>(l) * ldt, ldt, T_ + static_cast<long>(l) * ldt,
                         ldt));
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &one,
                         T_ + static_cast<long>(l) * ldt + l, ldt, T_ + static_cast<long>(l) * ldt,
                         ldt, T_ + static_cast<long>(l) * ldt, ldt));
  } else if (!dirf && colv) {
    // QL: backward + columnwise. T lower triangular; off-diagonal block is the
    // (l, k-l) corner T_21 = T(k-l:k, 0:k-l). T_11 = T(0:k-l, 0:k-l) (k-l square),
    // T_22 = T(k-l:k, k-l:k) (l square).
    CLM_TRY(larft<T>(handle, direct, storev, n - l, k - l, V, ldv, tau, T_, ldt));
    CLM_TRY(larft<T>(handle, direct, storev, n, l, V + static_cast<long>(k - l) * ldv, ldv,
                     tau + (k - l), T_ + static_cast<long>(k - l) * ldt + (k - l), ldt));

    // T_21 = V_{2,2}^T : transpose V(n-k:n-l, k-l:k) into T(k-l:k, 0:k-l).
    CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, l, k - l, &one,
                         V + static_cast<long>(k - l) * ldv + (n - k), ldv, &zero, T_ + (k - l),
                         ldt, T_ + (k - l), ldt));
    // T_21 *= V_{2,1} (unit upper at V(n-k:n-l, 0:k-l)).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, l, k - l, &one, V + (n - k),
                         ldv, T_ + (k - l), ldt, T_ + (k - l), ldt));
    // T_21 += V_{1,2}^T V_{1,1} (the rectangular top V(0:n-k, k-l:k) and V(0:n-k, 0:k-l)).
    if (n > k) {
      CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, l, k - l, n - k, &one,
                           V + static_cast<long>(k - l) * ldv, ldv, V, ldv, &one, T_ + (k - l),
                           ldt));
    }
    // T_21 = -T_22 * T_21 (T_22 lower, non-unit) then T_21 *= T_11 (lower, non-unit).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &neg_one,
                         T_ + static_cast<long>(k - l) * ldt + (k - l), ldt, T_ + (k - l), ldt,
                         T_ + (k - l), ldt));
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &one, T_, ldt,
                         T_ + (k - l), ldt, T_ + (k - l), ldt));
  } else {
    // RQ: backward + rowwise. T lower triangular; off-diagonal block T_21 = (k-l:k, 0:k-l).
    // V split into rows V_1 (rows 0:k-l) and V_2 (rows k-l:k), each unit-lower-trapezoidal.
    CLM_TRY(larft<T>(handle, direct, storev, n - l, k - l, V, ldv, tau, T_, ldt));
    CLM_TRY(larft<T>(handle, direct, storev, n, l, V + (k - l), ldv, tau + (k - l),
                     T_ + static_cast<long>(k - l) * ldt + (k - l), ldt));

    // T_21 = V_{2,2} : straight copy of V(k-l:k, n-k:n) into T(k-l:k, 0:k-l).
    CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, l, k - l, &one,
                         V + static_cast<long>(n - k) * ldv + (k - l), ldv, &zero, T_ + (k - l),
                         ldt, T_ + (k - l), ldt));
    // T_21 *= V_{1,2}^T (V_{1,2} unit lower at V(0:k-l, n-k:n)).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                         wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, l, k - l, &one,
                         V + static_cast<long>(n - k) * ldv, ldv, T_ + (k - l), ldt, T_ + (k - l),
                         ldt));
    // T_21 += V_{2,1} V_{1,1}^T (the rectangular left V(k-l:k, 0:n-k) and V(0:k-l, 0:n-k)).
    if (n > k) {
      CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, l, k - l, n - k, &one,
                           V + (k - l), ldv, V, ldv, &one, T_ + (k - l), ldt));
    }
    // T_21 = -T_22 * T_21 (T_22 lower, non-unit) then T_21 *= T_11 (lower, non-unit).
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &neg_one,
                         T_ + static_cast<long>(k - l) * ldt + (k - l), ldt, T_ + (k - l), ldt,
                         T_ + (k - l), ldt));
    CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                         wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_NON_UNIT, l, k - l, &one, T_, ldt,
                         T_ + (k - l), ldt, T_ + (k - l), ldt));
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
