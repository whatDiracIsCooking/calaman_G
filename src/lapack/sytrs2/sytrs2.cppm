/**
 * @file sytrs2.cppm
 * @brief The calaman.sytrs2 module -- the level-3 solve of A X = B for a symmetric
 *        matrix already Bunch-Kaufman factored, LAPACK's ?sytrs2
 *
 * Same result as calaman.sytrs, but level-3: it pulls the pivot interchanges to
 * the ends so the triangular solves are two clean trsm calls over all right-hand
 * sides at once, rather than the per-column ger/gemv walk. Templated over the four
 * element types; COMPLEX IS SYMMETRIC, not Hermitian -- the Hermitian cousin is
 * calaman.hetrs2.
 *
 * The recipe mirrors the reference ?sytrs2 exactly:
 *   1. ?syconv CONVERT -- move each 2x2 block's D off-diagonal out of the factor
 *      into @p d_work (VALUE, a device kernel) and permute the factor's
 *      off-diagonal triangle (PERMUTATIONS, host-driven wwr::swap), leaving a pure
 *      unit-triangular factor for trsm.
 *   2. P^T B, trsm (unit triangular), D^-1 (the block applies, off-diagonals read
 *      from @p d_work), trsm (transpose), P B.
 *   3. ?syconv REVERT -- undo (1), restoring the factor exactly.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv, B and @p d_work are
 * caller-provided device pointers; @p d_work is the n-element E vector (the one
 * extra buffer over the level-2 path). @p d_A is restored on return. Requires the
 * handle's DEFAULT (host) pointer mode. The only host memory is the n-int pivot
 * copy this routine stages to drive its loops.
 */

module;

#include "sytrs2_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

export module calaman.sytrs2;

import std;               // std::vector, std::max
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_* / _SIDE_* / _FILL_* / _DIAG_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;       // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas; // trsm, swap
import calaman.common;    // Uplo, usual_fp, kOne

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace sytrs2_detail {

using std::size_t;

/// @brief ?syconv CONVERT PERMUTATIONS: permute the factor's off-diagonal triangle
template<typename T>
Status perm_A_convert(wwr::wwrblasHandle_t h, bool upper, int n, T *A, int lda,
                      const std::vector<int> &p) {
  if (upper) {
    int i = n - 1;
    while (i >= 0) {
      if (p[static_cast<size_t>(i)] > 0) {
        const int ip = p[static_cast<size_t>(i)] - 1;
        if (i < n - 1) {
          CLM_TRY(wwr::swap<T, int>(h, n - i - 1, A + static_cast<size_t>(i + 1) * lda + ip, lda,
                                    A + static_cast<size_t>(i + 1) * lda + i, lda));
        }
      } else {
        const int ip = -p[static_cast<size_t>(i)] - 1;
        if (i < n - 1) {
          CLM_TRY(wwr::swap<T, int>(h, n - i - 1, A + static_cast<size_t>(i + 1) * lda + ip, lda,
                                    A + static_cast<size_t>(i + 1) * lda + (i - 1), lda));
        }
        --i;
      }
      --i;
    }
  } else {
    int i = 0;
    while (i < n) {
      if (p[static_cast<size_t>(i)] > 0) {
        const int ip = p[static_cast<size_t>(i)] - 1;
        if (i > 0) {
          CLM_TRY(wwr::swap<T, int>(h, i, A + ip, lda, A + i, lda));
        }
      } else {
        const int ip = -p[static_cast<size_t>(i)] - 1;
        if (i > 0) {
          CLM_TRY(wwr::swap<T, int>(h, i, A + ip, lda, A + (i + 1), lda));
        }
        ++i;
      }
      ++i;
    }
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief ?syconv REVERT PERMUTATIONS: undo perm_A_convert (reverse order)
template<typename T>
Status perm_A_revert(wwr::wwrblasHandle_t h, bool upper, int n, T *A, int lda,
                     const std::vector<int> &p) {
  if (upper) {
    int i = 0;
    while (i < n) {
      if (p[static_cast<size_t>(i)] > 0) {
        const int ip = p[static_cast<size_t>(i)] - 1;
        if (i < n - 1) {
          CLM_TRY(wwr::swap<T, int>(h, n - i - 1, A + static_cast<size_t>(i + 1) * lda + ip, lda,
                                    A + static_cast<size_t>(i + 1) * lda + i, lda));
        }
      } else {
        const int ip = -p[static_cast<size_t>(i)] - 1;
        ++i;
        if (i < n - 1) {
          CLM_TRY(wwr::swap<T, int>(h, n - i - 1, A + static_cast<size_t>(i + 1) * lda + ip, lda,
                                    A + static_cast<size_t>(i + 1) * lda + (i - 1), lda));
        }
      }
      ++i;
    }
  } else {
    int i = n - 1;
    while (i >= 0) {
      if (p[static_cast<size_t>(i)] > 0) {
        const int ip = p[static_cast<size_t>(i)] - 1;
        if (i > 0) {
          CLM_TRY(wwr::swap<T, int>(h, i, A + i, lda, A + ip, lda));
        }
      } else {
        const int ip = -p[static_cast<size_t>(i)] - 1;
        --i;
        if (i > 0) {
          CLM_TRY(wwr::swap<T, int>(h, i, A + (i + 1), lda, A + ip, lda));
        }
      }
      --i;
    }
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief P^T B: apply the pivot interchanges to the right-hand sides (pre-solve)
template<typename T>
Status perm_B_ptb(wwr::wwrblasHandle_t h, bool upper, int n, int nrhs, T *B, int ldb,
                  const std::vector<int> &p) {
  if (upper) {
    int k = n - 1;
    while (k >= 0) {
      if (p[static_cast<size_t>(k)] > 0) {
        const int kp = p[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + k, ldb, B + kp, ldb));
        }
        --k;
      } else {
        const int kp = -p[static_cast<size_t>(k)] - 1;
        if (p[static_cast<size_t>(k)] == p[static_cast<size_t>(k - 1)] && kp != k - 1) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + (k - 1), ldb, B + kp, ldb));
        }
        k -= 2;
      }
    }
  } else {
    int k = 0;
    while (k < n) {
      if (p[static_cast<size_t>(k)] > 0) {
        const int kp = p[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + k, ldb, B + kp, ldb));
        }
        ++k;
      } else {
        const int kp = -p[static_cast<size_t>(k + 1)] - 1;
        if (p[static_cast<size_t>(k + 1)] == p[static_cast<size_t>(k)] && kp != k + 1) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + (k + 1), ldb, B + kp, ldb));
        }
        k += 2;
      }
    }
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief P B: apply the pivot interchanges to the right-hand sides (post-solve)
template<typename T>
Status perm_B_pb(wwr::wwrblasHandle_t h, bool upper, int n, int nrhs, T *B, int ldb,
                 const std::vector<int> &p) {
  if (upper) {
    int k = 0;
    while (k < n) {
      if (p[static_cast<size_t>(k)] > 0) {
        const int kp = p[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + k, ldb, B + kp, ldb));
        }
        ++k;
      } else {
        const int kp = -p[static_cast<size_t>(k)] - 1;
        if (k < n - 1 && p[static_cast<size_t>(k)] == p[static_cast<size_t>(k + 1)] && kp != k) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + k, ldb, B + kp, ldb));
        }
        k += 2;
      }
    }
  } else {
    int k = n - 1;
    while (k >= 0) {
      if (p[static_cast<size_t>(k)] > 0) {
        const int kp = p[static_cast<size_t>(k)] - 1;
        if (kp != k) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + k, ldb, B + kp, ldb));
        }
        --k;
      } else {
        const int kp = -p[static_cast<size_t>(k)] - 1;
        if (k > 0 && p[static_cast<size_t>(k)] == p[static_cast<size_t>(k - 1)] && kp != k) {
          CLM_TRY(wwr::swap<T, int>(h, nrhs, B + k, ldb, B + kp, ldb));
        }
        k -= 2;
      }
    }
  }
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace sytrs2_detail

/// @brief Device workspace, in elements of T, required by sytrs2() (= n)
///
/// The level-3 solve needs one n-element E vector (the 2x2 off-diagonals ?syconv
/// pulls out) over the level-2 path's zero extra workspace.
export template<usual_fp T>
Status sytrs2_bufferSize(const int n, int *lwork) {
  if (n < 0) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  *lwork = n;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Level-3 solve A X = B for a Bunch-Kaufman-factored symmetric A (?sytrs2)
///
/// Overwrites @p d_B with X, using the factor in @p d_A and the pivots in
/// @p d_ipiv that wwr::sytrf produced for the same @p uplo. Complex is symmetric
/// (no conjugation). @p d_A is converted in place and restored before return. The
/// handle must be in host pointer mode and bound to the stream all work runs on.
///
/// @tparam T Element type; one of the four usual_fp types
/// @param handle GPU BLAS handle in host pointer mode; all arrays live on its device
/// @param uplo Which triangle of A holds the factor (as passed to the factorization)
/// @param n Order of the symmetric matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; the factor, restored on return
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device pivot array, length n; the 1-based sequence wwr::sytrf wrote
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @param d_work Device workspace, >= n elements of T (the E vector)
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<usual_fp T>
Status sytrs2(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, const int nrhs, T *d_A,
              const int lda, const int *d_ipiv, T *d_B, const int ldb, T *d_work) {
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
  const bool upper = uplo == Uplo::U;
  const T *const one = &kOne<T>;
  const wwr::wwrblasFillMode_t fill =
      upper ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
  T *const E = d_work;

  // ── ?syconv CONVERT: VALUE (kernel) then PERMUTATIONS (host-driven swaps) ───
  device::syconv_value<T>(stream, upper, /*convert=*/true, n, d_A, static_cast<size_t>(lda), d_ipiv,
                          E);
  CLM_TRY(sytrs2_detail::perm_A_convert<T>(handle, upper, n, d_A, lda, piv));

  // ── P^T B ──────────────────────────────────────────────────────────────────
  CLM_TRY(sytrs2_detail::perm_B_ptb<T>(handle, upper, n, nrhs, d_B, ldb, piv));

  // ── trsm 1: (U or L) \ B, unit triangular ──────────────────────────────────
  CLM_TRY(wwr::trsm<T, int>(handle, wwr::WWRBLAS_SIDE_LEFT, fill, wwr::WWRBLAS_OP_N,
                            wwr::WWRBLAS_DIAG_UNIT, n, nrhs, one, d_A, lda, d_B, ldb));

  // ── D^-1 B (block-diagonal; 2x2 off-diagonals read from E) ──────────────────
  if (upper) {
    int i = n - 1;
    while (i >= 0) {
      if (piv[static_cast<size_t>(i)] > 0) {
        device::sytrs2_scale_row<T>(stream, d_B + i, ldb, nrhs,
                                    d_A + static_cast<size_t>(i) * lda + i);
      } else if (i > 0 && piv[static_cast<size_t>(i - 1)] == piv[static_cast<size_t>(i)]) {
        device::sytrs2_solve_2x2<T>(stream, d_B + (i - 1), d_B + i, ldb, nrhs,
                                    d_A + static_cast<size_t>(i - 1) * lda + (i - 1), E + i,
                                    d_A + static_cast<size_t>(i) * lda + i);
        --i;
      }
      --i;
    }
  } else {
    int i = 0;
    while (i < n) {
      if (piv[static_cast<size_t>(i)] > 0) {
        device::sytrs2_scale_row<T>(stream, d_B + i, ldb, nrhs,
                                    d_A + static_cast<size_t>(i) * lda + i);
      } else {
        device::sytrs2_solve_2x2<T>(stream, d_B + i, d_B + (i + 1), ldb, nrhs,
                                    d_A + static_cast<size_t>(i) * lda + i, E + i,
                                    d_A + static_cast<size_t>(i + 1) * lda + (i + 1));
        ++i;
      }
      ++i;
    }
  }

  // ── trsm 2: (U or L)^T \ B, unit triangular ─────────────────────────────────
  CLM_TRY(wwr::trsm<T, int>(handle, wwr::WWRBLAS_SIDE_LEFT, fill, wwr::WWRBLAS_OP_T,
                            wwr::WWRBLAS_DIAG_UNIT, n, nrhs, one, d_A, lda, d_B, ldb));

  // ── P B ────────────────────────────────────────────────────────────────────
  CLM_TRY(sytrs2_detail::perm_B_pb<T>(handle, upper, n, nrhs, d_B, ldb, piv));

  // ── ?syconv REVERT: PERMUTATIONS then VALUE -- restore d_A exactly ──────────
  CLM_TRY(sytrs2_detail::perm_A_revert<T>(handle, upper, n, d_A, lda, piv));
  device::syconv_value<T>(stream, upper, /*convert=*/false, n, d_A, static_cast<size_t>(lda),
                          d_ipiv, E);
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
