/**
 * @file columnwise_ell2_bridge.h
 * @brief Device-launcher declaration shared between calaman.columnwise_ell2's
 *        interface unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by
 * columnwise_ell2.cu directly -- the same split lacpy_bridge.h / horner_bridge.h
 * and the sibling columnwise_ell1_bridge.h use. The declaration must live in the
 * GMF, not the module purview: a purview name gets module linkage and can never
 * bind to a definition compiled in a plain TU, which is what the .cu is.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather than an
 * `import`, since a GMF cannot import. It is the SAME type wwr.runtime_api
 * exports, so the module passes its stream straight through. Reading the backend
 * define selected_backend.h needs is why the module links wwr_backend PRIVATE --
 * see this directory's CMakeLists.txt.
 *
 * Templated over float and double only, matching the rest of calaman. The L2
 * norm of a complex column is real-valued (a sum of squared moduli under a
 * sqrt), so a complex variant is a separate (T -> real) reduction, a deliberate
 * later extension -- see interface.cppm.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the per-column L2 (Euclidean) norm of a column-major matrix on
///        @p stream
///
/// Writes `d_result[j] = sqrt(sum_i A(i, j)^2)` for each of the @p cols columns,
/// as two passes on @p stream: a per-column sum of squares, then an element-wise
/// sqrt over the @p cols results. Enqueued on @p stream and returns without
/// synchronizing. Launches nothing when @p rows or @p cols is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream   Stream the launches are enqueued on; A and result live on its device
/// @param rows     Rows per column
/// @param cols     Number of columns
/// @param d_A      Column-major matrix; column j starts at `d_A + j*lda`
/// @param lda      Leading dimension (column stride); lda >= rows
/// @param d_result Device output of length @p cols
template<typename T>
void columnwise_ell2(wwr::wwrStream_t stream, std::size_t rows, std::size_t cols, const T *d_A,
                     std::size_t lda, T *d_result);

} // namespace calaman::device
