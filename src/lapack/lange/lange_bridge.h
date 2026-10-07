/**
 * @file lange_bridge.h
 * @brief Device-launcher declaration shared between calaman.lange's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lange.cu
 * directly -- the same split lacpy_bridge.h and the columnwise_*_bridge.h use.
 * The declaration must live in the GMF, not the module purview: a purview name
 * gets module linkage and can never bind to a definition compiled in a plain TU,
 * which is what the .cu is.
 *
 * wwrStream_t arrives from runtime.h and `which` is calaman::MatrixNorm
 * (common/enums.h) -- both by #include, not import, since a GMF cannot import.
 * wwrStream_t is the SAME type wwr.runtime_api exports and MatrixNorm the same
 * plain-header enum calaman.common re-exports, so the module passes both straight
 * through. Reading the backend define selected_backend.h needs is why the module
 * links wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 *
 * The caller owns @p d_scratch (the per-column or per-row intermediate the
 * two-stage reduction writes); the host wrapper allocates it. R is T's real
 * component type: scratch and result are R, so a complex matrix has a real norm.
 * It is a second parameter, not ComplexToRealType<T>, so this GMF header needs
 * no fp_types.h.
 */

#pragma once

#include "common/enums.h"
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?lange matrix norm @p which of a column-major matrix
///
/// Reduces A to a per-column (or, for MatrixNorm::inf, per-row) intermediate in
/// @p d_scratch, then folds that to the single scalar @p d_result. Enqueued on
/// @p stream and returns without synchronizing. Assumes @p m and @p n are both
/// nonzero (the host wrapper handles the empty case).
///
/// @tparam T Element type; float, double, wwrFloatComplex, wwrDoubleComplex
/// @tparam R Real component type of T -- the type of the scratch and the result
/// @param stream    Stream the launches are enqueued on; all pointers live on its device
/// @param which     Which matrix norm to compute
/// @param m         Rows of A
/// @param n         Columns of A
/// @param d_A       Column-major matrix; column j starts at `d_A + j*lda`
/// @param lda       Leading dimension (column stride); lda >= m
/// @param d_result  Device scalar receiving the norm
/// @param d_scratch Device scratch of length n (or m when which == inf)
template<typename T, typename R>
void lange(wwr::wwrStream_t stream, MatrixNorm which, std::size_t m, std::size_t n, const T *d_A,
           std::size_t lda, R *d_result, R *d_scratch);

} // namespace calaman::device
