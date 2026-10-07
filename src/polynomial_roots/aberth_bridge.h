/**
 * @file aberth_bridge.h
 * @brief Device-launcher declaration shared between calaman.polynomial_roots's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its global module fragment and by aberth.cu
 * directly, so the declaration binds to the definition compiled in the plain .cu
 * TU (lartg_bridge.h's split). wwrStream_t arrives from runtime.h, which is why
 * the module links wwr_backend PRIVATE.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// Dynamic shared memory one block may claim without an opt-in, on both backends.
inline constexpr std::size_t kAberthMaxSharedBytes = std::size_t{48} * 1024;

/// @brief Every root of @p batch degree-@p n polynomials, one warp each
///
/// Polynomial b is coeffs[b*(n+1) + k] = a_k (ascending); its roots land in
/// roots[b*n, b*n + n) and its iteration count, or -1, in iters[b]. Enqueued on
/// @p stream; returns without synchronizing. Requires n >= 1, batch >= 1, and
/// 2n * sizeof(CT) <= kAberthMaxSharedBytes. Unconstrained: a requires-clause
/// enters the mangled name, which nvcc and clang spell differently.
template<typename T, typename CT, typename R>
void aberth(wwr::wwrStream_t stream, int n, int batch, const T *coeffs, CT *roots, int *iters,
            R tol, int max_iter);

} // namespace calaman::device
