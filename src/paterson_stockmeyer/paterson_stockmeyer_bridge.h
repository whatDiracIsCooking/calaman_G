/**
 * @file paterson_stockmeyer_bridge.h
 * @brief Device-launcher declaration shared between calaman.paterson_stockmeyer's
 *        interface unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by
 * paterson_stockmeyer.cu directly -- the same split horner_bridge.h /
 * laqps_bridge.h use: the declaration lives in the GMF, not the module purview,
 * so a purview name's module linkage cannot stop it binding to the definition
 * compiled in the plain .cu translation unit.
 *
 * Paterson-Stockmeyer is a host composition of wrapped BLAS gemms (the power
 * bank and the outer Horner steps, via wwr::gemm) plus exactly one genuinely
 * device-elementwise stage: building each block
 *
 *     B_j = c_(js) I + c_(js+1) A + ... + c_(js+m-1) A^(m-1)
 *
 * from the precomputed powers in a single fused pass. With BLAS that would be m
 * separate axpy launches, each a full read-modify-write over n^2; this does it
 * in one pass. That launcher is this bridge, run through
 * wwr.extension.parallel_for from paterson_stockmeyer.cu.
 *
 * The coefficients are read from a *device* pointer, which is what keeps the
 * whole evaluation on the handle's stream without a host round-trip: a
 * coefficient produced by an earlier kernel is fed straight in. This is
 * independent of the BLAS handle's pointer mode -- the kernel reads device
 * memory directly, while the gemm scalars are the host constants kOne/kZero.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather than an
 * `import`, since a GMF cannot import; it is the SAME type wwr.runtime_api
 * exports, so the module passes its handle's stream straight through. Reading
 * the backend define that header needs is why the module links wwr_backend
 * PRIVATE -- see this directory's CMakeLists.txt.
 *
 * Templated over float and double only. Complex p(A) is a deliberate later
 * extension, for the reason calaman.common's constants.h documents: there is no
 * portable constexpr spelling of the kOne/kZero gemm scalars in complex.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief Build one Paterson-Stockmeyer block into @p d_B on @p stream
///
/// Writes, over the n-by-n column-major block at @p d_B,
///
///     B = d_coeffs[0] * I
///       + d_coeffs[1] * A
///       + sum_{k=2}^{num_terms-1} d_coeffs[k] * A^k
///
/// in a single fused pass -- one thread per element walks the @p num_terms
/// terms and writes once. A^0 is the identity and is never materialized; A^1 is
/// the caller's matrix, with its own leading dimension; A^2 and up come from the
/// precomputed power bank, which is uniform -- leading dimension @p ld_powers,
/// @p stride elements between consecutive powers, A^2 first. Enqueued on
/// @p stream; returns without synchronizing. Launches nothing when @p n is 0 or
/// @p num_terms is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream    Stream the launch is enqueued on
/// @param d_B       Device output, n-by-n column-major, leading dimension @p ldb
/// @param ldb       Leading dimension of @p d_B; ldb >= n
/// @param n         Matrix order
/// @param d_A       A itself (the k = 1 term), leading dimension @p lda
/// @param lda       Leading dimension of @p d_A; lda >= n
/// @param d_powers  Bank of A^2, A^3, ... ; may be null when num_terms <= 2
/// @param ld_powers Leading dimension of each power in the bank
/// @param stride    Elements between consecutive powers in the bank
/// @param d_coeffs  Device pointer to this block's @p num_terms coefficients
/// @param num_terms Number of terms, 1 to s; term k multiplies A^k
template<typename T>
void paterson_stockmeyer_build_block(wwr::wwrStream_t stream, T *d_B, int ldb, int n, const T *d_A,
                                     int lda, const T *d_powers, int ld_powers, std::size_t stride,
                                     const T *d_coeffs, int num_terms);

extern template void paterson_stockmeyer_build_block<float>(wwr::wwrStream_t, float *, int, int,
                                                            const float *, int, const float *, int,
                                                            std::size_t, const float *, int);
extern template void paterson_stockmeyer_build_block<double>(wwr::wwrStream_t, double *, int, int,
                                                             const double *, int, const double *,
                                                             int, std::size_t, const double *, int);

} // namespace calaman::device
