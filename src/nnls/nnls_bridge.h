/**
 * @file nnls_bridge.h
 * @brief Device-launcher declarations shared between calaman.nnls and its
 *        device-compiled translation unit
 *
 * Included by nnls.cppm in its GLOBAL MODULE FRAGMENT and by nnls.cu directly,
 * the split larfg/laqp2/laqps use: the declarations live in the GMF, not the
 * module purview, so a purview name's module linkage cannot stop them binding
 * to the definitions compiled in the plain .cu translation unit.
 *
 * nnls()'s numerical bulk is wrapped BLAS (gemv/trsv/nrm2), calaman::geqp3 for
 * the passive-set pivoted QR, and wwr::ormqr to apply Q. What stays here is the
 * P/Z active-set bookkeeping no BLAS call expresses: the masked argmax that
 * picks the entering column, the Lawson-Hanson min-ratio step test, the
 * compaction that moves zeroed entries back to Z, plus the gather that forms the
 * passive submatrix and the rank read off the pivoted R diagonal.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * handle's stream straight through. Reading the backend define that header needs
 * is why the module links wwr_backend PRIVATE -- see this directory's
 * CMakeLists.txt, the calaman.laqps arrangement.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Seed the bookkeeping: order = iota(n), x = 0 (both length @p n)
template<typename T>
void nnls_init(wwr::wwrStream_t stream, int *order, T *x, int n);

/// @brief Swap order[@p a] and order[@p b] in one single-thread launch
void nnls_swap_order(wwr::wwrStream_t stream, int *order, int a, int b);

/// @brief Position in [@p p, @p n) maximizing w[order[k]]
///
/// Single-block reduction; writes the max value to @p max_val and its position
/// within @p order (not the column index) to @p arg_pos. Not called when p == n.
template<typename T>
void nnls_masked_argmax(wwr::wwrStream_t stream, const T *w, const int *order, int p, int n,
                        T *max_val, int *arg_pos);

/// @brief Gather the @p p passive columns of A into a packed m-by-p matrix
///
/// AP[:, c] = A[:, order[c]] for c in [0, @p p), column-major, AP leading
/// dimension @p m. The gather map is the outer P/Z partition used directly.
template<typename T>
void nnls_gather_columns(wwr::wwrStream_t stream, T *AP, int m, const T *A, int lda,
                         const int *order, int p);

/// @brief Numerical rank off a pivoted R diagonal
///
/// Counts the leading diagonal entries of the @p k-by-@p k upper triangle @p R
/// (leading dimension @p ldr) with |R[i,i]| > @p rel_tol * |R[0,0]|; writes it to
/// @p rank. Valid because column pivoting orders |R[i,i]| non-increasing.
template<typename T>
void nnls_rank(wwr::wwrStream_t stream, const T *R, int ldr, int k, T rel_tol, int *rank);

/// @brief Scatter a passive-set solution into the full n-length candidate
///
/// z[order[jpvt0[j]]] = qb[j] for j in [0, @p p) -- composes geqp3's pivot
/// (@p jpvt0, 0-based) with the outer P/Z partition. @p z is caller-pre-zeroed.
template<typename T>
void nnls_scatter_z(wwr::wwrStream_t stream, T *z, const T *qb, const int *order, const int *jpvt0,
                    int p);

/// @brief The Lawson-Hanson step-length test over order[0, @p p)
///
/// Scans the passive entries for z[order[k]] <= 0, tracking the minimum of
/// x[order[k]] / (x[order[k]] - z[order[k]]). @p infeasible is 1 if any such
/// entry exists (then @p alpha holds the step), else 0.
template<typename T>
void nnls_masked_min_ratio(wwr::wwrStream_t stream, const T *x, const T *z, const int *order, int p,
                           T *alpha, int *infeasible);

/// @brief x[i] += alpha * (z[i] - x[i]) for i in [0, @p n)
template<typename T>
void nnls_line_step(wwr::wwrStream_t stream, T *x, const T *z, T alpha, int n);

/// @brief Move the just-zeroed entries of P back into Z
///
/// Stable-partitions order[0, @p p): entries with |x[order[k]]| > @p eps stay in
/// P (front), the rest rejoin Z. @p scratch is caller workspace of >= @p p ints;
/// the new passive-set size is written to @p new_p.
template<typename T>
void nnls_compact_zeros(wwr::wwrStream_t stream, int *order, int *scratch, const T *x, int p, T eps,
                        int *new_p);

} // namespace calaman::device
