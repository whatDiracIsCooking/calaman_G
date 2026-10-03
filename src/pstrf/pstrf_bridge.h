/**
 * @file pstrf_bridge.h
 * @brief Device-launcher declarations shared between calaman.pstrf and its
 *        device-compiled translation unit
 *
 * Included by pstrf.cppm in its GLOBAL MODULE FRAGMENT and by pstrf.cu directly,
 * the split larfg/laqp2/laqps/pstf2 use: the declarations live in the GMF, not
 * the module purview, so a purview name's module linkage cannot stop them
 * binding to the definitions compiled in the plain .cu translation unit.
 *
 * pstrf()'s numerical bulk is wrapped BLAS (syrk/gemv/scal/swap) plus the
 * calaman.pstf2 unblocked fallback. What stays here is the two pieces no BLAS
 * call expresses -- structurally the calaman.pstf2 kernels, kept separate on the
 * module boundary (the issue #65 call structure): the per-step diagonal
 * maintenance fused with the pivot search over the trailing range, and zeroing
 * the trailing factor after a rank-revealing stop.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * handle's stream straight through. Reading the backend define that header needs
 * is why the module links wwr_backend PRIVATE -- see this directory's
 * CMakeLists.txt, the calaman.pstf2 arrangement.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief Maintain the Schur-complement diagonals over [@p j, @p n) and pick the pivot
///
/// One step of pivoted Cholesky's diagonal bookkeeping, fused into a single-block
/// reduction. For each i in [@p j, @p n): when @p add_prev, folds the just-computed
/// factor entry into the running dot product (@p dots[i] += prev[i]^2, @p prev
/// strided by @p prev_stride); then forms the Schur diagonal A(i,i) - dots[i].
/// Writes to the three-element device buffer @p out: out[0] the maximum Schur
/// diagonal, out[1] its index i (as a value), out[2] a 0/1 flag set when any
/// diagonal in the range is NaN. In the blocked driver @p dots is the WITHIN-BLOCK
/// partial (reset each block), the prior blocks' contribution already in A's
/// diagonal via the trailing syrk; @p add_prev is false on a block's first column.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void pstrf_pivot(wwr::wwrStream_t stream, int j, int n, bool add_prev, const T *prev,
                 std::size_t prev_stride, const T *A, std::size_t lda, T *dots, T *out);

/// @brief Zero the trailing factor outside the computed leading @p rank columns
///
/// Sets the stored triangle of the trailing block A(@p rank:@p n, @p rank:@p n) to
/// zero -- the upper triangle when @p upper, else the lower -- so a rank-revealing
/// stop leaves a clean factor whose uncomputed tail is exactly zero. Launches
/// nothing when @p rank >= @p n.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void pstrf_zero_trailing(wwr::wwrStream_t stream, bool upper, int rank, int n, T *A,
                         std::size_t lda);

} // namespace calaman::device
