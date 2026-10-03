/**
 * @file larfb.cppm
 * @brief The calaman.larfb module -- apply a block reflector or its transpose
 *        to a general matrix (LAPACK's ?larfb)
 *
 * One host routine that applies the order-k block reflector H = I - V T V^T
 * (StoreV::C) or H = I - V^T T V (StoreV::R) -- or its transpose H^T -- to an
 * m-by-n matrix C in place, from the left (C := H C / H^T C) or the right
 * (C := C H / C H^T). This is LAPACK's ?larfb, composed entirely from wrapped
 * BLAS-3 (trmm, gemm) plus geam for the block copies and the final rank-k
 * subtraction -- NO custom kernel, the pure-host shape of calaman.larft.
 *
 * The eight branches are the (storev, direct, side) combinations. Each forms the
 * intermediate W (n-by-k for side L, m-by-k for side R) in the caller's
 * workspace, multiplies by the triangular factor T (op per @p trans), and
 * subtracts the rank-k correction back into C. V's trapezoidal halves (V1 unit
 * triangular, V2 the rectangular tail) follow ?larfb's Further Details layout;
 * see the per-branch comments for the exact block algebra. The geam that seeds W
 * replaces ?larfb's dcopy loop, and the geam that closes each branch replaces its
 * explicit C1 -= W (or C -= W^T) loop.
 *
 * Requires the handle's DEFAULT (host) pointer mode: the BLAS scalars are host
 * addresses, like calaman.larft. V, T, C and W are device pointers the caller
 * owns; nothing is allocated and the stream is not synchronized. T carries the
 * factor calaman.larft produces. Templated over float and double; complex is a
 * later extension.
 *
 * Usage:
 *   import calaman.larfb;    // names calaman::larfb, Side, Trans, Direct, StoreV
 *   // d_V, d_T: the block reflector; d_C: m x n (col-major); d_W: ldwork x k
 *   calaman::larfb<double>(handle, Side::L, Trans::N, Direct::F, StoreV::C,
 *                          m, n, k, d_V, ldv, d_T, ldt, d_C, ldc, d_W, ldwork);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.larfb;

import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_* / _SIDE_* / _FILL_* / _DIAG_*, status
import wwr.wrappers.blas; // gemm, trmm, geam
import calaman.common;    // Side, Trans, Direct, StoreV (:enums)

// export import, not a plain import: larfb RETURNS calaman::Status, so a consumer
// of `import calaman.larfb;` must see Status's member functions, not just its
// name -- the same re-export larft does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Side/Trans/Direct/StoreV live in calaman.common's :enums partition. Re-export
// them so `import calaman.larfb;` alone names them, as the usage example relies
// on -- the same re-export larft does for Direct/StoreV.
export using calaman::Direct;
export using calaman::Side;
export using calaman::StoreV;
export using calaman::Trans;

/// @brief Apply a block reflector H (or H^T) to C in place (LAPACK ?larfb)
///
/// Side::L computes C := H C or H^T C; Side::R computes C := C H or C H^T, where
/// H = I - V T V^T (StoreV::C) or I - V^T T V (StoreV::R), of the shape @p direct
/// fixes. @p trans selects H (Trans::N) or H^T (Trans::T). Short-circuits: the
/// first failing BLAS step's Status is returned. Does nothing and returns success
/// when @p m, @p n or @p k is 0.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; V, T_, C, W live on its device
/// @param side Side::L for C := H C / H^T C, Side::R for C := C H / C H^T
/// @param trans Trans::N applies H, Trans::T applies H^T
/// @param direct Reflector order / T shape (Direct::F forward/upper, Direct::B backward/lower)
/// @param storev How V stores the reflectors (StoreV::C columnwise, StoreV::R rowwise)
/// @param m Row count of C
/// @param n Column count of C
/// @param k Order of T = number of elementary reflectors (k <= m for L, k <= n for R)
/// @param V Device matrix of reflectors, column-major (see ?larfb Further Details for shape)
/// @param ldv Leading dimension of V
/// @param T_ Device k-by-k triangular factor, column-major
/// @param ldt Leading dimension of T_ (>= k)
/// @param C Device matrix, m by n, column-major, updated in place
/// @param ldc Leading dimension of C (>= max(1,m))
/// @param W Device workspace, ldwork by k; overwritten with the intermediate
/// @param ldwork Leading dimension of W (>= max(1,n) for L, >= max(1,m) for R)
/// @return The Status of the first failing BLAS step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status larfb(wwr::wwrblasHandle_t handle, const Side side, const Trans trans, const Direct direct,
             const StoreV storev, const int m, const int n, const int k, const T *V, const int ldv,
             const T *T_, const int ldt, T *C, const int ldc, T *W, const int ldwork) {
  if (m == 0 || n == 0 || k == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  const T one = T{1};
  const T zero = T{0};
  const T neg_one = T{-1};

  // op applied to T: trans selects H (op = N) or H^T (op = T); the W := W * T
  // steps use op_trans, matching ?larfb's TRANS, and the side-L form (which
  // builds W from C^T) uses op_transt = the opposite, matching ?larfb's TRANST.
  const wwr::wwrblasOperation_t op_trans =
      trans == Trans::N ? wwr::WWRBLAS_OP_N : wwr::WWRBLAS_OP_T;
  const wwr::wwrblasOperation_t op_transt =
      trans == Trans::N ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N;

  const bool left = side == Side::L;
  const bool dirf = direct == Direct::F;
  const bool colv = storev == StoreV::C;

  if (colv && dirf) {
    // STOREV='C', DIRECT='F': V = (V1; V2), V1 unit lower triangular (first k rows).
    if (left) {
      // C = (C1; C2), C1 the top k rows. W (n x k) := C1^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, k, &one, C, ldc, &zero,
                           W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, n, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      if (m > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, k, m - k, &one, C + k,
                             ldc, V + k, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER, op_transt,
                           wwr::WWRBLAS_DIAG_NON_UNIT, n, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (m > k) {
        // C2 := C2 - V2 * W^T.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m - k, n, k, &neg_one,
                             V + k, ldv, W, ldwork, &one, C + k, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, n, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      // C1 := C1 - W^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, k, n, &neg_one, W, ldwork,
                           &one, C, ldc, C, ldc));
    } else {
      // C = (C1 C2), C1 the left k columns. W (m x k) := C1.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &one, C, ldc, &zero,
                           W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, m, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      if (n > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, n - k, &one,
                             C + static_cast<long>(k) * ldc, ldc, V + k, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER, op_trans,
                           wwr::WWRBLAS_DIAG_NON_UNIT, m, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (n > k) {
        // C2 := C2 - W * V2^T.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m, n - k, k, &neg_one, W,
                             ldwork, V + k, ldv, &one, C + static_cast<long>(k) * ldc, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, m, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      // C1 := C1 - W.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &neg_one, W, ldwork,
                           &one, C, ldc, C, ldc));
    }
  } else if (colv && !dirf) {
    // STOREV='C', DIRECT='B': V = (V1; V2), V2 unit upper triangular (last k rows).
    if (left) {
      // C = (C1; C2), C2 the bottom k rows at row m-k. W (n x k) := C2^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, k, &one, C + (m - k),
                           ldc, &zero, W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, n, k, &one, V + (m - k), ldv,
                           W, ldwork, W, ldwork));
      if (m > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, k, m - k, &one, C,
                             ldc, V, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, op_transt,
                           wwr::WWRBLAS_DIAG_NON_UNIT, n, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (m > k) {
        // C1 := C1 - V1 * W^T.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m - k, n, k, &neg_one, V,
                             ldv, W, ldwork, &one, C, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, n, k, &one, V + (m - k), ldv,
                           W, ldwork, W, ldwork));
      // C2 := C2 - W^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, k, n, &neg_one, W, ldwork,
                           &one, C + (m - k), ldc, C + (m - k), ldc));
    } else {
      // C = (C1 C2), C2 the right k columns at column n-k. W (m x k) := C2.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &one,
                           C + static_cast<long>(n - k) * ldc, ldc, &zero, W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, m, k, &one, V + (n - k), ldv,
                           W, ldwork, W, ldwork));
      if (n > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, n - k, &one, C,
                             ldc, V, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, op_trans,
                           wwr::WWRBLAS_DIAG_NON_UNIT, m, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (n > k) {
        // C1 := C1 - W * V1^T.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m, n - k, k, &neg_one, W,
                             ldwork, V, ldv, &one, C, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, m, k, &one, V + (n - k), ldv,
                           W, ldwork, W, ldwork));
      // C2 := C2 - W.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &neg_one, W, ldwork,
                           &one, C + static_cast<long>(n - k) * ldc, ldc,
                           C + static_cast<long>(n - k) * ldc, ldc));
    }
  } else if (!colv && dirf) {
    // STOREV='R', DIRECT='F': V = (V1 V2), V1 unit upper triangular (first k columns).
    if (left) {
      // C = (C1; C2), C1 the top k rows. W (n x k) := C1^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, k, &one, C, ldc, &zero,
                           W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, n, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      if (m > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_T, n, k, m - k, &one, C + k,
                             ldc, V + static_cast<long>(k) * ldv, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER, op_transt,
                           wwr::WWRBLAS_DIAG_NON_UNIT, n, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (m > k) {
        // C2 := C2 - V2^T * W^T.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_T, m - k, n, k, &neg_one,
                             V + static_cast<long>(k) * ldv, ldv, W, ldwork, &one, C + k, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, n, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      // C1 := C1 - W^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, k, n, &neg_one, W, ldwork,
                           &one, C, ldc, C, ldc));
    } else {
      // C = (C1 C2), C1 the left k columns. W (m x k) := C1.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &one, C, ldc, &zero,
                           W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, m, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      if (n > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m, k, n - k, &one,
                             C + static_cast<long>(k) * ldc, ldc, V + static_cast<long>(k) * ldv,
                             ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER, op_trans,
                           wwr::WWRBLAS_DIAG_NON_UNIT, m, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (n > k) {
        // C2 := C2 - W * V2.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n - k, k, &neg_one, W,
                             ldwork, V + static_cast<long>(k) * ldv, ldv, &one,
                             C + static_cast<long>(k) * ldc, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_UPPER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, m, k, &one, V, ldv, W, ldwork,
                           W, ldwork));
      // C1 := C1 - W.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &neg_one, W, ldwork,
                           &one, C, ldc, C, ldc));
    }
  } else {
    // STOREV='R', DIRECT='B': V = (V1 V2), V2 unit lower triangular (last k columns).
    if (left) {
      // C = (C1; C2), C2 the bottom k rows at row m-k. W (n x k) := C2^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, n, k, &one, C + (m - k),
                           ldc, &zero, W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, n, k, &one,
                           V + static_cast<long>(m - k) * ldv, ldv, W, ldwork, W, ldwork));
      if (m > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_T, n, k, m - k, &one, C,
                             ldc, V, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, op_transt,
                           wwr::WWRBLAS_DIAG_NON_UNIT, n, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (m > k) {
        // C1 := C1 - V1^T * W^T.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_T, m - k, n, k, &neg_one, V,
                             ldv, W, ldwork, &one, C, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, n, k, &one,
                           V + static_cast<long>(m - k) * ldv, ldv, W, ldwork, W, ldwork));
      // C2 := C2 - W^T.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, k, n, &neg_one, W, ldwork,
                           &one, C + (m - k), ldc, C + (m - k), ldc));
    } else {
      // C = (C1 C2), C2 the right k columns at column n-k. W (m x k) := C2.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &one,
                           C + static_cast<long>(n - k) * ldc, ldc, &zero, W, ldwork, W, ldwork));
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, m, k, &one,
                           V + static_cast<long>(n - k) * ldv, ldv, W, ldwork, W, ldwork));
      if (n > k) {
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, m, k, n - k, &one, C,
                             ldc, V, ldv, &one, W, ldwork));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER, op_trans,
                           wwr::WWRBLAS_DIAG_NON_UNIT, m, k, &one, T_, ldt, W, ldwork, W, ldwork));
      if (n > k) {
        // C1 := C1 - W * V1.
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n - k, k, &neg_one, W,
                             ldwork, V, ldv, &one, C, ldc));
      }
      CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                           wwr::WWRBLAS_OP_N, wwr::WWRBLAS_DIAG_UNIT, m, k, &one,
                           V + static_cast<long>(n - k) * ldv, ldv, W, ldwork, W, ldwork));
      // C2 := C2 - W.
      CLM_TRY(wwr::geam<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, k, &neg_one, W, ldwork,
                           &one, C + static_cast<long>(n - k) * ldc, ldc,
                           C + static_cast<long>(n - k) * ldc, ldc));
    }
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
