/**
 * @file hetrs2_bridge.h
 * @brief Device-launcher declarations shared between calaman.hetrs2 and its
 *        device-compiled translation unit
 *
 * Included by hetrs2.cppm in its GLOBAL MODULE FRAGMENT and by hetrs2.cu directly.
 * hetrs2() is the Hermitian level-3 solve; like the reference ?hetrs2 it reuses the
 * SYMMETRIC ?syconv (the VALUE move is unconjugated -- the off-diagonal stored in E
 * is the raw D entry, and the conjugation lives in the conjugate-transpose trsm and
 * the Hermitian D^-1). What stays here is the ?syconv VALUE move plus the two
 * Hermitian D^-1 applies (real 1x1, conjugated 2x2). Complex only (c/z).
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief ?syconv VALUE: move the 2x2 D off-diagonals between the factor and E
///
/// Identical to the symmetric path (?hetrs2 calls ?syconv, not a Hermitian conv):
/// @p convert true pulls each 2x2 block's off-diagonal out of @p A into @p E (and
/// zeroes it, leaving a unit-triangular factor for trsm) and zeroes @p E at 1x1
/// positions; @p convert false restores them. No conjugation here.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetrs2_syconv_value(wwr::wwrStream_t stream, bool upper, bool convert, int n, T *A,
                         std::size_t lda, const int *d_ipiv, T *E);

/// @brief 1x1 pivot: scale one B row by 1/Re(D(k,k)) (the Hermitian diagonal is real)
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetrs2_scale_row_real(wwr::wwrStream_t stream, T *b_row, int ldb, int nrhs, const T *akk);

/// @brief 2x2 pivot: apply the inverse of one Hermitian 2x2 D block to two B rows
///
/// @p off is the raw off-diagonal E entry; @p top_conj picks which row divides by
/// its conjugate (true for LOWER, false for UPPER -- the reference's DCONJG(AKM1K)).
/// @p top_diag / @p bot_diag are the stored (real-valued) D diagonals.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetrs2_solve_2x2(wwr::wwrStream_t stream, T *b_top, T *b_bot, int ldb, int nrhs,
                      const T *top_diag, const T *off, const T *bot_diag, bool top_conj);

} // namespace calaman::device
