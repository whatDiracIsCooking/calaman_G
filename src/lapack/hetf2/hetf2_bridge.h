/**
 * @file hetf2_bridge.h
 * @brief Device-launcher declarations shared between calaman.hetf2 and its
 *        device-compiled translation unit
 *
 * Included by hetf2.cppm in its GLOBAL MODULE FRAGMENT and by hetf2.cu directly.
 * hetf2()'s pivot search and decisions run host-side over wrapped BLAS (iamax,
 * her, scal, swap) with per-step scalar reads; what stays here is what no BLAS
 * expresses: the Hermitian interchange of a pivot pair (a conjugating row/column
 * dance), and the rank-2 trailing update for a 2x2 pivot block.
 *
 * Complex only (c/z): a real Hermitian matrix is symmetric, which is calaman.sytf2
 * territory (the vendor's sytrf). The 2x2 scalars d_a/d_b/dfac arrive as double and
 * are narrowed to the element's real type in the kernel.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Hermitian interchange of the pivot pair, plus making pivot diagonals real
///
/// One step of ?hetf2's "Interchange rows and columns KK and KP" block (both the
/// @p kp != @p kk swap-with-conjugation dance and the @p kp == @p kk real-diagonal
/// fixup), single-threaded over the O(n) affected entries. All indices 0-based;
/// @p kk is the pivot's leading column, @p kp the row/column it swaps with.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetf2_interchange(wwr::wwrStream_t stream, bool upper, int n, T *A, std::size_t lda, int k,
                       int kk, int kp, int kstep);

/// @brief Rank-2 trailing update for a 2x2 pivot block (?hetf2's DO 40 / DO 80)
///
/// Forms the two factor columns W into @p wa (column k) and @p wb (its partner,
/// k-1 upper / k+1 lower) from the inverse-D scalars, subtracts W inv(D) W^H from
/// the trailing triangle, stores W back into A's two columns, and keeps the touched
/// diagonals real. @p d_a / @p d_b are the real inv-D coefficients, @p doff the
/// (scaled) off-diagonal, @p dfac the shared real factor. @p wa / @p wb are n-length
/// device scratch.
///
/// @tparam T Complex element type; instantiated for wwrFloatComplex, wwrDoubleComplex
template<typename T>
void hetf2_rank2(wwr::wwrStream_t stream, bool upper, int n, T *A, std::size_t lda, int k,
                 double d_a, double d_b, T doff, double dfac, T *wa, T *wb);

} // namespace calaman::device
