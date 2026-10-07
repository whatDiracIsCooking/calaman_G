/**
 * @file tfsm.cppm
 * @brief calaman.tfsm -- triangular solve with the triangular operand held in
 *        Rectangular Full Packed storage, LAPACK's ?tfsm (s/d/c/z)
 *
 * B := alpha * op(A)**-1 * B (side L) or alpha * B * op(A)**-1 (side R), A
 * triangular of order m (side L) or n (side R) in RFP, B m-by-n full. op(A)
 * is block triangular over A11 / A22 as calaman.rfp_blocks places them, so
 * every branch is one trsm on the block of B solved first, one gemm folding
 * it into the other block of B, and one trsm on that. No device code of its
 * own and no workspace.
 *
 * Requires the handle's DEFAULT (host) pointer mode: the scalars are host
 * addresses. Enqueued on the handle's stream; returns without synchronizing.
 *
 * | ?TFSM                | calaman::tfsm                                |
 * |----------------------|----------------------------------------------|
 * | CHARACTER TRANSR     | Trans (N; T real, C complex)                 |
 * | CHARACTER SIDE       | Side                                         |
 * | CHARACTER UPLO       | Uplo                                         |
 * | CHARACTER TRANS      | Trans (N; T real, C complex)                 |
 * | CHARACTER DIAG       | Diag                                         |
 * | s/d/c/z variants     | one template over T (usual_fp)               |
 * | INFO                 | returned Status (invalid-value for bad args) |
 *
 * Usage:
 *   import calaman.tfsm;   // also re-exports Trans, Uplo, Side, Diag, Status
 *   calaman::tfsm<double>(blas, Trans::N, Side::L, Uplo::L, Trans::T, Diag::N,
 *                         m, n, alpha, d_arf, d_b, ldb);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment;
// needs calaman::Status visible at expansion (the export import below).
#include "error_handling/error_macros.h"

export module calaman.tfsm;

import std;
import wwr.blas;           // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_* enums
import wwr.runtime_api;    // wwrStream_t, wwrMemset2DAsync, wwrSuccess
import wwr.complex;        // wwrFloatComplex, wwrDoubleComplex, component access
import wwr.wrappers.blas;  // gemm, trsm
import calaman.common;     // Trans, Uplo, Side, Diag, usual_fp, complex_fp
import calaman.rfp_blocks; // rfp_blocks -- the RFP sub-block geometry

// export import: tfsm RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

export using calaman::Diag;
export using calaman::Side;
export using calaman::Trans;
export using calaman::Uplo;

namespace tfsm_detail {

template<usual_fp T>
bool is_zero(const T x) {
  if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::wwrCrealf(x) == 0.0f && wwr::wwrCimagf(x) == 0.0f;
  } else if constexpr (std::is_same_v<T, wwr::wwrDoubleComplex>) {
    return wwr::wwrCreal(x) == 0.0 && wwr::wwrCimag(x) == 0.0;
  } else {
    return x == T{0};
  }
}

template<usual_fp T>
T real_element(const double x) {
  if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::make_wwrFloatComplex(static_cast<float>(x), 0.0f);
  } else if constexpr (std::is_same_v<T, wwr::wwrDoubleComplex>) {
    return wwr::make_wwrDoubleComplex(x, 0.0);
  } else {
    return static_cast<T>(x);
  }
}

} // namespace tfsm_detail

/// @brief Solve op(A) X = alpha B or X op(A) = alpha B for X, A triangular in RFP
///
/// Quick return (B untouched) when m or n is 0; alpha = 0 zeroes B's m-by-n
/// block. Returns invalid-value for a bad transr/trans, a negative m or n,
/// ldb below max(1, m), or a null A or B.
///
/// @tparam T      Element type (usual_fp)
/// @param handle  BLAS handle in host pointer mode; its stream carries every op
/// @param transr  RFP layout of A: Trans::N, or Trans::T (real) / Trans::C (complex)
/// @param side    Side::L: op(A) X = alpha B, A m-by-m; Side::R: X op(A), A n-by-n
/// @param uplo    Which triangle A is
/// @param trans   op(A): Trans::N, or Trans::T (real) / Trans::C (complex)
/// @param diag    Diag::U: A is unit triangular, its stored diagonal is not read
/// @param m       Rows of B
/// @param n       Columns of B
/// @param alpha   Scalar on B
/// @param d_a     Device RFP array of A, order(order+1)/2 elements
/// @param d_b     Device B, column-major, leading dimension ldb; overwritten by X
/// @param ldb     Leading dimension of B; >= max(1, m)
/// @return Success, invalid-value, a trsm/gemm status, or a memset error
export template<usual_fp T>
Status tfsm(const wwr::wwrblasHandle_t handle, const Trans transr, const Side side,
            const Uplo uplo, const Trans trans, const Diag diag, const int m, const int n,
            const T alpha, const T *const d_a, T *const d_b, const int ldb) {
  constexpr Trans kTransposed = complex_fp<T> ? Trans::C : Trans::T;
  constexpr auto kOpH = complex_fp<T> ? wwr::WWRBLAS_OP_C : wwr::WWRBLAS_OP_T;
  if ((transr != Trans::N && transr != kTransposed) ||
      (trans != Trans::N && trans != kTransposed) || m < 0 || n < 0 || ldb < std::max(1, m)) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  if (d_b == nullptr || (d_a == nullptr && !tfsm_detail::is_zero(alpha))) {
    return wwr::WWRBLAS_STATUS_INVALID_VALUE;
  }

  if (tfsm_detail::is_zero(alpha)) {
    wwr::wwrStream_t stream{};
    CLM_TRY(wwr::wwrblasGetStream(handle, &stream));
    return wwr::wwrMemset2DAsync(d_b, static_cast<std::size_t>(ldb) * sizeof(T), 0,
                                 static_cast<std::size_t>(m) * sizeof(T),
                                 static_cast<std::size_t>(n), stream);
  }

  const bool left = side == Side::L;
  const bool notrans = trans == Trans::N;
  const RfpBlocks b = rfp_blocks(transr != Trans::N, uplo, static_cast<std::size_t>(left ? m : n));
  // op(A) is lower block triangular iff (L, N) or (U, T/C). Solving from the
  // left starts at the top block when it is lower; from the right, the bottom.
  const bool lower = (uplo == Uplo::L) == notrans;
  const bool first_is_11 = left == lower;
  const RfpTriangle &first = first_is_11 ? b.a11 : b.a22;
  const RfpTriangle &second = first_is_11 ? b.a22 : b.a11;
  // Row (side L) or column (side R) block of B that pairs with each triangle.
  const auto block_of_b = [&](const bool is_11) {
    const std::size_t start = is_11 ? 0 : b.n1;
    return d_b + (left ? start : start * static_cast<std::size_t>(ldb));
  };
  T *const b_first = block_of_b(first_is_11);
  T *const b_second = block_of_b(!first_is_11);

  // A block stored as the (conjugate) transpose of the logical one flips op.
  const auto op_of = [&](const bool transposed) {
    return transposed != !notrans ? kOpH : wwr::WWRBLAS_OP_N;
  };
  const auto blas_side = left ? wwr::WWRBLAS_SIDE_LEFT : wwr::WWRBLAS_SIDE_RIGHT;
  const auto blas_diag = diag == Diag::U ? wwr::WWRBLAS_DIAG_UNIT : wwr::WWRBLAS_DIAG_NON_UNIT;
  const auto solve = [&](const RfpTriangle &t, const T *scale, T *const d_x) -> Status {
    const auto fill =
        t.uplo == Uplo::U ? wwr::WWRBLAS_FILL_MODE_UPPER : wwr::WWRBLAS_FILL_MODE_LOWER;
    const int order = static_cast<int>(t.order);
    return wwr::trsm<T>(handle, blas_side, fill, op_of(t.transposed), blas_diag,
                        left ? order : m, left ? n : order, scale, d_a + t.offset,
                        static_cast<int>(t.ld), d_x, ldb);
  };

  const auto nf = static_cast<int>(first.order);
  const auto ns = static_cast<int>(second.order);
  if (nf > 0) {
    CLM_TRY(solve(first, &alpha, b_first));
  }
  if (nf > 0 && ns > 0) {
    // B_second := alpha B_second - P X_first (side L) or - X_first P (side R),
    // P the off-diagonal block of op(A); then B_second is already scaled.
    const T neg = tfsm_detail::real_element<T>(-1.0);
    const T *const off = d_a + b.off.offset;
    const auto op_off = op_of(b.off.transposed);
    const auto ldo = static_cast<int>(b.off.ld);
    if (left) {
      CLM_TRY(wwr::gemm<T>(handle, op_off, wwr::WWRBLAS_OP_N, ns, n, nf, &neg, off, ldo, b_first,
                           ldb, &alpha, b_second, ldb));
    } else {
      CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, op_off, m, ns, nf, &neg, b_first, ldb, off,
                           ldo, &alpha, b_second, ldb));
    }
  }
  if (ns > 0) {
    // alpha went in with the gemm; with an empty first block it goes in here.
    const T scale = nf > 0 ? tfsm_detail::real_element<T>(1.0) : alpha;
    CLM_TRY(solve(second, &scale, b_second));
  }
  return wwr::wwrSuccess;
}

} // namespace calaman
