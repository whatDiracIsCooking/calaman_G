/**
 * @file sytrs2_bridge.h
 * @brief Device-launcher declarations shared between calaman.sytrs2 and its
 *        device-compiled translation unit
 *
 * Included by sytrs2.cppm in its GLOBAL MODULE FRAGMENT and by sytrs2.cu
 * directly, the split pstrf/sytrs use. sytrs2()'s numerical bulk is wrapped
 * BLAS-3 (trsm) plus swaps; what stays here is what no BLAS expresses: the
 * ?syconv VALUE move (the block-diagonal D off-diagonals between the stored
 * factor and a separate E vector), and the two D^-1 block applies to the
 * right-hand sides (structurally calaman.sytrs's, kept separate on the module
 * boundary as pstrf keeps pstf2's).
 *
 * The ?syconv PERMUTATION step is NOT here -- it is a sequence of strided row
 * swaps over a column range, which the module drives host-side through wwr::swap.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief ?syconv VALUE: move the 2x2 D off-diagonals between the factor and E
///
/// @p convert true pulls each 2x2 block's off-diagonal out of the stored factor
/// @p A into @p E (and zeroes it in A, leaving a pure unit-triangular factor for
/// trsm) and sets @p E to zero at 1x1 positions; @p convert false restores them.
/// Reads the pivot signs from the device @p d_ipiv. Single-threaded: O(n) trivial
/// work, dwarfed by the trsm it precedes.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex, wwrDoubleComplex
template<typename T>
void syconv_value(wwr::wwrStream_t stream, bool upper, bool convert, int n, T *A, std::size_t lda,
                  const int *d_ipiv, T *E);

/// @brief 1x1 pivot: scale one right-hand-side row by 1/D(k,k)
///
/// Divides @p b_row[j*ldb] by *@p akk for every j in [0, @p nrhs) -- the
/// DSCAL(1/A(I,I)) of the reference ?sytrs2 1x1 branch.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex, wwrDoubleComplex
template<typename T>
void sytrs2_scale_row(wwr::wwrStream_t stream, T *b_row, int ldb, int nrhs, const T *akk);

/// @brief 2x2 pivot: apply the inverse of one symmetric 2x2 D block to two B rows
///
/// Solves D * [x_top; x_bot] = [b_top; b_bot] per column, where D is the symmetric
/// 2x2 [[*@p top_diag, *@p off], [*@p off, *@p bot_diag]] -- here @p off is the E
/// entry ?syconv pulled out, @p top_diag / @p bot_diag the stored D diagonals. The
/// AKM1K/AKM1/AK/DENOM arithmetic of the reference ?sytrs2 2x2 branch.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex, wwrDoubleComplex
template<typename T>
void sytrs2_solve_2x2(wwr::wwrStream_t stream, T *b_top, T *b_bot, int ldb, int nrhs,
                      const T *top_diag, const T *off, const T *bot_diag);

} // namespace calaman::device
