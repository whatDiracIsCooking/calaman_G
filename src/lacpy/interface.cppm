/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lacpy -- copy all or part of a matrix
 *
 * A single device routine: copy the m-by-n column-major matrix A to B, either
 * whole or restricted to its upper or lower triangle. The copy is enqueued on
 * the given stream and returns WITHOUT synchronizing, like a BLAS call; the
 * caller synchronizes when it needs B. A and B are device pointers the caller
 * owns; nothing is allocated here, and B outside the copied region is untouched.
 *
 * A stream, not a device handle, is the whole requirement: this routine enqueues
 * one kernel and allocates nothing, so it needs no device index and no memory
 * pool. Taking a `wwrStream_t` also keeps a concrete handle type -- and the error
 * policy such a type hard-codes -- out of calaman's shipped surface, matching
 * calaman.diff_norm's bare wwrblasHandle_t. See test/shared/README.md.
 *
 * Mapping from LAPACK's DLACPY (docs/architecture.md §4 -- keep the name, drop
 * the Fortran calling convention):
 *
 * | DLACPY           | calaman::lacpy                          |
 * |------------------|-----------------------------------------|
 * | CHARACTER UPLO   | copy_region enum (no lsame char)        |
 * | s/d/c/z variants | one template over T (float, double)     |
 * | INTEGER extents  | std::size_t                             |
 * | LDA / LDB        | kept: column-major leading dimensions   |
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lacpy;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_a, d_b: device matrices, column-major, leading dims lda/ldb
 *   calaman::lacpy(stream, calaman::copy_region::upper, m, n, d_a, lda, d_b, ldb);
 */

module;

#include "lacpy_bridge.h"

export module calaman.lacpy;

import std;
import wwr.runtime_api;

namespace calaman {

/// @brief Which part of A to copy -- the typed replacement for DLACPY's UPLO
///
/// Enumerator values are the integer contract launch_lacpy() takes (see
/// lacpy_bridge.h); keep them in step.
export enum class copy_region : int {
  full = 0,  ///< the whole m-by-n matrix
  upper = 1, ///< the upper triangle/trapezoid (diagonal and above)
  lower = 2, ///< the lower triangle/trapezoid (diagonal and below)
};

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Copy the m-by-n column-major matrix A to B on @p stream
///
/// Enqueues the copy of the region @p region selects and returns without
/// synchronizing; B outside that region is left untouched. Enqueues nothing
/// when @p m or @p n is 0. A and B must live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the copy is enqueued on; A and B live on its device
/// @param region Which part of A to copy (full, upper, lower)
/// @param m Number of rows of A and B
/// @param n Number of columns of A and B
/// @param d_a Source device matrix, column-major, leading dimension @p lda
/// @param lda Leading dimension of A; lda >= m
/// @param d_b Destination device matrix, column-major, leading dimension @p ldb
/// @param ldb Leading dimension of B; ldb >= m
export template<typename T>
void lacpy(const wwr::wwrStream_t stream, const copy_region region, const std::size_t m,
           const std::size_t n, const T *d_a, const std::size_t lda, T *d_b,
           const std::size_t ldb) {
  if (m == 0 || n == 0) {
    return;
  }
  device::launch_lacpy(stream, static_cast<int>(region), m, n, d_a, lda, d_b, ldb);
}

extern template void lacpy<float>(wwr::wwrStream_t, copy_region, std::size_t, std::size_t,
                                  const float *, std::size_t, float *, std::size_t);
extern template void lacpy<double>(wwr::wwrStream_t, copy_region, std::size_t, std::size_t,
                                   const double *, std::size_t, double *, std::size_t);

} // namespace calaman
