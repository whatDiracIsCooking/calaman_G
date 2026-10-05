/**
 * @file hetrs_bridge.h
 * @brief Device-launcher declarations shared between calaman.hetrs and its
 *        device-compiled translation unit
 *
 * Included by hetrs.cppm in its GLOBAL MODULE FRAGMENT and by hetrs.cu directly,
 * the split sytrs uses. hetrs()'s numerical bulk is wrapped BLAS (geru/gemv/swap);
 * what stays here is what the HERMITIAN solve needs beyond them: the 1x1 apply by
 * the REAL reciprocal of D(k,k), the Hermitian 2x2 block apply (the off-diagonal
 * conjugated on one row), and the row conjugation that wraps each conjugate-
 * transpose gemv (the reference ?hetrs's ZLACGV + ZGEMV('C') + ZLACGV).
 *
 * Complex only (c/z): a real Hermitian matrix is symmetric, which is calaman.sytrs.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief 1x1 pivot: scale one B row by 1/Re(D(k,k)) (the Hermitian diagonal is real)
///
/// Multiplies @p b_row[j*ldb] by 1 / real(*@p akk) for every j in [0, @p nrhs) --
/// the ZDSCAL(1/DBLE(A(K,K))) of the reference ?hetrs 1x1 branch.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetrs_scale_row_real(wwr::wwrStream_t stream, T *b_row, int ldb, int nrhs, const T *akk);

/// @brief 2x2 pivot: apply the inverse of one Hermitian 2x2 D block to two B rows
///
/// Solves D * [x_top; x_bot] = [b_top; b_bot] per column, where D's off-diagonal is
/// *@p off and its conjugate -- the reference ?hetrs's DCONJG(AKM1K) dance. When
/// @p top_conj the top row divides by conj(off) and the bottom by off (the LOWER
/// case); otherwise top by off, bottom by conj(off) (the UPPER case). @p top_diag /
/// @p bot_diag are the stored (real-valued) D diagonals.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetrs_solve_2x2(wwr::wwrStream_t stream, T *b_top, T *b_bot, int ldb, int nrhs,
                     const T *top_diag, const T *off, const T *bot_diag, bool top_conj);

/// @brief Conjugate one B row in place: b[j*ldb] <- conj(b[j*ldb]), j in [0, nrhs)
///
/// The ZLACGV that brackets each conjugate-transpose gemv in the reference ?hetrs
/// back-substitution, so a plain gemv('C') realizes the Hermitian update without
/// touching (or needing scratch for) the conjugate of a factor column.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetrs_conj_row(wwr::wwrStream_t stream, T *b_row, int ldb, int nrhs);

} // namespace calaman::device
