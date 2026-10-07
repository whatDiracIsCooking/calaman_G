/**
 * @file hfrk.cppm
 * @brief calaman.hfrk -- rank-k update of a Hermitian/symmetric matrix held in
 *        Rectangular Full Packed storage, LAPACK's ?hfrk (c/z) and ?sfrk (s/d)
 *
 * C := alpha * A * A**H + beta * C (trans N) or alpha * A**H * A + beta * C
 * (trans C, or T for a real T), C n-by-n in RFP, alpha and beta real. One
 * module for all four precisions: for a real T it is ?sfrk, and `sfrk` names
 * the same routine. Two herk/syrk calls on the diagonal blocks and one gemm on
 * the off-diagonal block, placed by calaman.rfp_blocks. No device code of its
 * own and no workspace.
 *
 * Requires the handle's DEFAULT (host) pointer mode: the scalars are host
 * addresses. Enqueued on the handle's stream; returns without synchronizing.
 *
 * | ?HFRK / ?SFRK      | calaman::hfrk                                  |
 * |--------------------|------------------------------------------------|
 * | CHARACTER TRANSR   | Trans (N; T real, C complex)                   |
 * | CHARACTER UPLO     | Uplo                                           |
 * | CHARACTER TRANS    | Trans (N; T real, C complex)                   |
 * | s/d/c/z variants   | one template over T (usual_fp)                 |
 * | INFO               | returned Status (invalid-value for bad args)   |
 *
 * Usage:
 *   import calaman.hfrk;   // also re-exports Trans, Uplo and Status
 *   calaman::hfrk<wwr::wwrDoubleComplex>(blas, Trans::N, Uplo::L, Trans::N, n, k,
 *                                        alpha, d_a, lda, beta, d_c);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment;
// needs calaman::Status visible at expansion (the export import below).
#include "error_handling/error_macros.h"

export module calaman.hfrk;

import std;
import wwr.blas;           // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_OP_*, FILL_MODE_*
import wwr.runtime_api;    // wwrStream_t, wwrMemsetAsync, wwrSuccess
import wwr.complex;        // wwrFloatComplex, wwrDoubleComplex, make_wwr*Complex
import wwr.wrappers.blas;  // gemm, herk, syrk
import calaman.common;     // Trans, Uplo, usual_fp, complex_fp, ComplexToRealType
import calaman.rfp_blocks; // rfp_blocks -- the RFP sub-block geometry

// export import: hfrk RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

export using calaman::Trans;
export using calaman::Uplo;

namespace hfrk_detail {

template<usual_fp T>
T as_element(const ComplexToRealType<T> x) {
  if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::make_wwrFloatComplex(x, 0.0f);
  } else if constexpr (std::is_same_v<T, wwr::wwrDoubleComplex>) {
    return wwr::make_wwrDoubleComplex(x, 0.0);
  } else {
    return x;
  }
}

/// @brief herk for a complex T, syrk for a real one -- the same update on C
template<usual_fp T>
Status rank_k(const wwr::wwrblasHandle_t handle, const Uplo uplo, const wwr::wwrblasOperation_t op,
              const int n, const int k, const ComplexToRealType<T> *alpha, const T *a,
              const int lda, const ComplexToRealType<T> *beta, T *c, const int ldc) {
  const auto fill = uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
  if constexpr (complex_fp<T>) {
    return wwr::herk<T>(handle, fill, op, n, k, alpha, a, lda, beta, c, ldc);
  } else {
    return wwr::syrk<T>(handle, fill, op, n, k, alpha, a, lda, beta, c, ldc);
  }
}

} // namespace hfrk_detail

/// @brief Rank-k update of the n-by-n Hermitian (symmetric, real T) C in RFP
///
/// Quick return (C untouched) when n is 0, or alpha or k is 0 and beta is 1;
/// alpha = beta = 0 zeroes all n(n+1)/2 elements. Returns invalid-value for a
/// bad transr/trans, a negative n or k, lda below max(1, nrowa), or a null C.
///
/// @tparam T      Element type (usual_fp); a real T is ?sfrk
/// @param handle  BLAS handle in host pointer mode; its stream carries every op
/// @param transr  RFP layout of C: Trans::N, or Trans::T (real) / Trans::C (complex)
/// @param uplo    Which triangle the RFP array holds
/// @param trans   Trans::N: C += A A**H, A n-by-k; Trans::C (T, real): A**H A, A k-by-n
/// @param n       Order of C
/// @param k       Columns (trans N) or rows (otherwise) of A
/// @param alpha   Real scalar on A A**H
/// @param d_a     Device A, column-major, leading dimension lda
/// @param lda     Leading dimension of A; >= max(1, n) for trans N, else max(1, k)
/// @param beta    Real scalar on C
/// @param d_c     Device RFP array of n(n+1)/2 elements; overwritten
/// @return Success, invalid-value, a herk/syrk/gemm status, or a memset error
export template<usual_fp T>
Status hfrk(const wwr::wwrblasHandle_t handle, const Trans transr, const Uplo uplo,
            const Trans trans, const int n, const int k, const ComplexToRealType<T> alpha,
            const T *const d_a, const int lda, const ComplexToRealType<T> beta, T *const d_c) {
  using R = ComplexToRealType<T>;
  constexpr Trans kTransposed = complex_fp<T> ? Trans::C : Trans::T;
  const bool notrans = trans == Trans::N;
  const int nrowa = notrans ? n : k;
  if ((transr != Trans::N && transr != kTransposed) || (!notrans && trans != kTransposed) ||
      n < 0 || k < 0 || lda < std::max(1, nrowa)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (n == 0 || ((alpha == R{0} || k == 0) && beta == R{1})) {
    return wwr::wwrSuccess;
  }
  if (d_c == nullptr || (d_a == nullptr && k > 0 && alpha != R{0})) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  if (alpha == R{0} && beta == R{0}) {
    wwr::wwrStream_t stream{};
    CLM_TRY(wwr::wwrblasGetStream(handle, &stream));
    const auto un = static_cast<std::size_t>(n);
    return wwr::wwrMemsetAsync(d_c, 0, un * (un + 1) / 2 * sizeof(T), stream);
  }

  const RfpBlocks b = rfp_blocks(transr != Trans::N, uplo, static_cast<std::size_t>(n));
  const auto n1 = static_cast<int>(b.n1);
  const auto n2 = static_cast<int>(b.n2);
  // A1 / A2: the rows (trans N) or columns of A that A11 / A22 are built from.
  const T *const a1 = d_a;
  const T *const a2 =
      d_a == nullptr ? nullptr : d_a + (notrans ? b.n1 : b.n1 * static_cast<std::size_t>(lda));
  const auto op_a =
      notrans ? wwr::WWRBLAS_OP_N : (complex_fp<T> ? wwr::WWRBLAS_OP_C : wwr::WWRBLAS_OP_T);
  const auto op_h = complex_fp<T> ? wwr::WWRBLAS_OP_C : wwr::WWRBLAS_OP_T;

  if (n1 > 0) {
    CLM_TRY(hfrk_detail::rank_k<T>(handle, b.a11.uplo, op_a, n1, k, &alpha, a1, lda, &beta,
                                   d_c + b.a11.offset, static_cast<int>(b.a11.ld)));
  }
  if (n2 > 0) {
    CLM_TRY(hfrk_detail::rank_k<T>(handle, b.a22.uplo, op_a, n2, k, &alpha, a2, lda, &beta,
                                   d_c + b.a22.offset, static_cast<int>(b.a22.ld)));
  }
  if (n1 > 0 && n2 > 0) {
    // The stored block is A21 = A2 A1**H when it holds rows of the trailing
    // block, else A12 = A1 A2**H; trans N/C picks op(A) = A or A**H.
    const bool stores_21 = (uplo == Uplo::L) != b.off.transposed;
    const T *const left = stores_21 ? a2 : a1;
    const T *const right = stores_21 ? a1 : a2;
    const T calpha = hfrk_detail::as_element<T>(alpha);
    const T cbeta = hfrk_detail::as_element<T>(beta);
    CLM_TRY(wwr::gemm<T>(handle, notrans ? wwr::WWRBLAS_OP_N : op_h,
                         notrans ? op_h : wwr::WWRBLAS_OP_N, static_cast<int>(b.off.rows),
                         static_cast<int>(b.off.cols), k, &calpha, left, lda, right, lda, &cbeta,
                         d_c + b.off.offset, static_cast<int>(b.off.ld)));
  }
  return wwr::wwrSuccess;
}

/// @brief ?sfrk: the real-T spelling of hfrk -- C := alpha A A**T + beta C in RFP
export template<real_fp T>
Status sfrk(const wwr::wwrblasHandle_t handle, const Trans transr, const Uplo uplo,
            const Trans trans, const int n, const int k, const T alpha, const T *const d_a,
            const int lda, const T beta, T *const d_c) {
  return hfrk<T>(handle, transr, uplo, trans, n, k, alpha, d_a, lda, beta, d_c);
}

} // namespace calaman
