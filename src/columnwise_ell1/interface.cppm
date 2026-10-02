/**
 * @file interface.cppm
 * @brief Primary interface for calaman.columnwise_ell1 -- the L1 norm of each
 *        column of a column-major matrix
 *
 * For each column j of an m-by-n column-major matrix A, computes
 *
 *     d_result[j] = sum_{i=0}^{rows-1} |A(i, j)|
 *
 * the L1 (sum-of-magnitudes) norm of that column, in one kernel over all
 * columns at once. This is the per-column counterpart of what BLAS asum does for
 * a single vector (calaman.diff_norm's Norm::l1): asum would need one call per
 * column, whereas this reduces every column in a single launch.
 *
 * The computation IS a per-column reduction, so it owns no kernel of its own:
 * columnwise_ell1.cu defers to calaman.reduce_columns (a header-only, no-Thrust
 * segmented reduce) with |.| as the pre-transform and + as the fold. The
 * transform path pre-transforms every element including the seed, so a
 * single-row column correctly reports |a|, not a.
 *
 * A stream, not a device handle, is the whole requirement: this enqueues one
 * kernel and allocates nothing, so it needs no device index and no memory pool.
 * Taking a `wwrStream_t` also keeps a concrete handle type out of calaman's
 * shipped surface, matching calaman.lacpy and calaman.diff_norm. Unlike the
 * reference this was ported from, it needs no caller-supplied key/scratch buffer
 * -- that was a Thrust reduce_by_key artifact the hand-written kernel removes.
 *
 * Templated over float and double. The L1 norm of a COMPLEX column is
 * real-valued (sum of moduli sqrt(re^2 + im^2)), so a complex variant is a
 * distinct T -> real reduction rather than this T -> T one -- a deliberate later
 * extension, as calaman.diff_norm notes for the same reason. reduce_columns is
 * already generic over that (ValT may differ from T); only this wrapper is
 * type-fixed.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.columnwise_ell1;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_A: rows-by-cols device matrix, column-major, leading dimension lda
 *   // d_norms: device vector of length cols
 *   calaman::columnwise_ell1(stream, rows, cols, d_A, lda, d_norms);
 */

module;

#include "columnwise_ell1_bridge.h"

export module calaman.columnwise_ell1;

import std;
import wwr.runtime_api; // wwrStream_t (the type the launcher takes)

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declaration below sits in the plain namespace.

/// @brief Per-column L1 norm of a column-major matrix on @p stream
///
/// Writes `d_result[j] = sum_i |A(i, j)|` for each of the @p cols columns.
/// Enqueues one kernel and returns without synchronizing, like a BLAS call; the
/// caller synchronizes when it needs the result. Enqueues nothing (and leaves
/// @p d_result untouched) when @p rows or @p cols is 0. A and result must live
/// on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream   Stream the kernel is enqueued on; A and result live on its device
/// @param rows     Number of rows reduced per column
/// @param cols     Number of columns, and the length of @p d_result
/// @param d_A      Source device matrix, column-major, leading dimension @p lda
/// @param lda      Leading dimension of @p d_A; lda >= rows (so columns do not overlap)
/// @param d_result Device output of length @p cols; `d_result[j]` is column j's L1 norm
export template<typename T>
void columnwise_ell1(const wwr::wwrStream_t stream, const std::size_t rows, const std::size_t cols,
                     const T *const d_A, const std::size_t lda, T *const d_result) {
  if (rows == 0 || cols == 0) {
    return;
  }
  device::columnwise_ell1(stream, rows, cols, d_A, lda, d_result);
}

extern template void columnwise_ell1<float>(wwr::wwrStream_t, std::size_t, std::size_t,
                                            const float *, std::size_t, float *);
extern template void columnwise_ell1<double>(wwr::wwrStream_t, std::size_t, std::size_t,
                                             const double *, std::size_t, double *);

} // namespace calaman
