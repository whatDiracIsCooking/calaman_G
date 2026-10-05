/**
 * @file hetrs2.cppm
 * @brief The calaman.hetrs2 module -- the level-3 solve of A X = B for a Hermitian
 *        matrix already Bunch-Kaufman factored, LAPACK's ?hetrs2
 *
 * Same result as calaman.hetrs, but level-3: the pivot interchanges are batched to
 * the ends so the triangular solves are two trsm calls. COMPLEX ONLY (c/z); a real
 * Hermitian matrix is symmetric (calaman.sytrs2).
 *
 * The recipe mirrors the reference ?hetrs2, which reuses the SYMMETRIC ?syconv (the
 * VALUE move is unconjugated; the conjugation lives in the conjugate-transpose trsm
 * and the Hermitian D^-1). So the ?syconv convert/revert (VALUE kernel + host-driven
 * PERMUTATION swaps) and the P^T B / P B pivot loops are identical to
 * calaman.sytrs2; hetrs2 differs only in the second trsm (OP_C, not OP_T) and the
 * Hermitian D^-1 (real 1x1, conjugated 2x2).
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p d_ipiv, B and @p d_work are
 * caller-provided device pointers; @p d_work is the n-element E vector. @p d_A is
 * restored on return. Requires the handle's DEFAULT (host) pointer mode.
 */

module;

#include "hetrs2_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment.
#include "error_handling/error_macros.h"

export module calaman.hetrs2;

import std;               // std::vector, std::max
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_OP_* / _SIDE_* / _FILL_* / _DIAG_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize
import wwr.complex;       // wwrFloatComplex, wwrDoubleComplex
import wwr.wrappers.blas; // trsm, swap
import calaman.common;    // Uplo, complex_fp, kOne

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

namespace hetrs2_detail {

using std::size_t;

// The ?syconv PERMUTATION and P^T B / P B loops are identical to calaman.sytrs2's
// (?hetrs2 reuses ?syconv, and its pivot-swap conditions match ?sytrs2's exactly).

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

} // namespace hetrs2_detail

/// @brief Device workspace, in elements of T, required by hetrs2() (= n)
export template<complex_fp T>
Status hetrs2_bufferSize(const int n, int *lwork) {
  if (n < 0) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  *lwork = n;
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Level-3 solve A X = B for a Bunch-Kaufman-factored Hermitian A (?hetrs2)
///
/// Overwrites @p d_B with X, using the factor in @p d_A and the pivots in
/// @p d_ipiv that ?hetrf produced for the same @p uplo. Complex only. @p d_A is
/// converted in place and restored before return. The handle must be in host
/// pointer mode and bound to the stream all work runs on.
///
/// @tparam T Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param handle GPU BLAS handle in host pointer mode; all arrays live on its device
/// @param uplo Which triangle of A holds the factor (as passed to the factorization)
/// @param n Order of the Hermitian matrix A
/// @param nrhs Number of right-hand sides (columns of B)
/// @param d_A Device matrix, n by n, column-major; the factor, restored on return
/// @param lda Leading dimension of A (>= max(1, n))
/// @param d_ipiv Device pivot array, length n; the 1-based sequence ?hetrf wrote
/// @param d_B Device matrix, n by nrhs, column-major; overwritten in place with X
/// @param ldb Leading dimension of B (>= max(1, n))
/// @param d_work Device workspace, >= n elements of T (the E vector)
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<complex_fp T>
Status hetrs2(wwr::wwrblasHandle_t handle, const Uplo uplo, const int n, const int nrhs, T *d_A,
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
  device::hetrs2_syconv_value<T>(stream, upper, /*convert=*/true, n, d_A, static_cast<size_t>(lda),
                                 d_ipiv, E);
  CLM_TRY(hetrs2_detail::perm_A_convert<T>(handle, upper, n, d_A, lda, piv));

  // ── P^T B ──────────────────────────────────────────────────────────────────
  CLM_TRY(hetrs2_detail::perm_B_ptb<T>(handle, upper, n, nrhs, d_B, ldb, piv));

  // ── trsm 1: (U or L) \ B, unit triangular, no transpose ─────────────────────
  CLM_TRY(wwr::trsm<T, int>(handle, wwr::WWRBLAS_SIDE_LEFT, fill, wwr::WWRBLAS_OP_N,
                            wwr::WWRBLAS_DIAG_UNIT, n, nrhs, one, d_A, lda, d_B, ldb));

  // ── D^-1 B (Hermitian: real 1x1, conjugated 2x2; off-diagonals from E) ──────
  if (upper) {
    int i = n - 1;
    while (i >= 0) {
      if (piv[static_cast<size_t>(i)] > 0) {
        device::hetrs2_scale_row_real<T>(stream, d_B + i, ldb, nrhs,
                                         d_A + static_cast<size_t>(i) * lda + i);
      } else if (i > 0 && piv[static_cast<size_t>(i - 1)] == piv[static_cast<size_t>(i)]) {
        device::hetrs2_solve_2x2<T>(stream, d_B + (i - 1), d_B + i, ldb, nrhs,
                                    d_A + static_cast<size_t>(i - 1) * lda + (i - 1), E + i,
                                    d_A + static_cast<size_t>(i) * lda + i, /*top_conj=*/false);
        --i;
      }
      --i;
    }
  } else {
    int i = 0;
    while (i < n) {
      if (piv[static_cast<size_t>(i)] > 0) {
        device::hetrs2_scale_row_real<T>(stream, d_B + i, ldb, nrhs,
                                         d_A + static_cast<size_t>(i) * lda + i);
      } else {
        device::hetrs2_solve_2x2<T>(stream, d_B + i, d_B + (i + 1), ldb, nrhs,
                                    d_A + static_cast<size_t>(i) * lda + i, E + i,
                                    d_A + static_cast<size_t>(i + 1) * lda + (i + 1),
                                    /*top_conj=*/true);
        ++i;
      }
      ++i;
    }
  }

  // ── trsm 2: (U or L)^H \ B, unit triangular, CONJUGATE transpose ────────────
  CLM_TRY(wwr::trsm<T, int>(handle, wwr::WWRBLAS_SIDE_LEFT, fill, wwr::WWRBLAS_OP_C,
                            wwr::WWRBLAS_DIAG_UNIT, n, nrhs, one, d_A, lda, d_B, ldb));

  // ── P B ────────────────────────────────────────────────────────────────────
  CLM_TRY(hetrs2_detail::perm_B_pb<T>(handle, upper, n, nrhs, d_B, ldb, piv));

  // ── ?syconv REVERT: PERMUTATIONS then VALUE -- restore d_A exactly ──────────
  CLM_TRY(hetrs2_detail::perm_A_revert<T>(handle, upper, n, d_A, lda, piv));
  device::hetrs2_syconv_value<T>(stream, upper, /*convert=*/false, n, d_A, static_cast<size_t>(lda),
                                 d_ipiv, E);
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
