/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laset -- set all or part of a matrix to
 *        constants
 *
 * A single device routine: set the m-by-n column-major matrix A to BETA on its
 * diagonal and ALPHA off it, either whole or restricted to its upper or lower
 * triangle. The fill is enqueued on the given stream and returns WITHOUT
 * synchronizing, like a BLAS call; the caller synchronizes when it needs A. A is
 * a device pointer the caller owns; nothing is allocated here, and A outside the
 * set region is untouched.
 *
 * A stream, not a device handle, is the whole requirement: this routine enqueues
 * one kernel and allocates nothing, so it needs no device index and no memory
 * pool. Taking a `wwrStream_t` also keeps a concrete handle type -- and the error
 * policy such a type hard-codes -- out of calaman's shipped surface, matching
 * calaman.lacpy's bare wwrStream_t. See test/shared/README.md.
 *
 * Mapping from LAPACK's DLASET (docs/architecture.md §4 -- keep the name, drop
 * the Fortran calling convention):
 *
 * | DLASET           | calaman::laset                          |
 * |------------------|-----------------------------------------|
 * | CHARACTER UPLO   | calaman::Region enum (no lsame char)    |
 * | s/d/c/z variants | one template over T (float, double)     |
 * | INTEGER extents  | std::size_t                             |
 * | ALPHA / BETA     | kept: off-diagonal / diagonal fill      |
 * | LDA              | kept: column-major leading dimension    |
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.laset;      // also re-exports calaman::Region and Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_a: device matrix, column-major, leading dimension lda
 *   // identity: ALPHA = 0 off the diagonal, BETA = 1 on it
 *   calaman::laset(stream, calaman::Region::A, m, n, T{0}, T{1}, d_a, lda);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include "laset_bridge.h"

export module calaman.laset;

import std;
import wwr.runtime_api;
import calaman.common; // Region (:enums) -- the typed replacement for DLASET UPLO

// export import, not a plain import: laset RETURNS calaman::Status, so a consumer
// of `import calaman.laset;` must see Status's member functions, not just its
// name -- the same re-export diff_norm does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Region (U / L / A) lives in calaman.common's :enums partition, shared with the
// device .cu through common/enums.h. Re-export it so `import calaman.laset;`
// alone still names calaman::Region, as the usage example and tests expect.
export using calaman::Region;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Set the m-by-n column-major matrix A to constants on @p stream
///
/// Enqueues the fill of the region @p region selects -- @p beta on the diagonal
/// (A(i,i), 0 <= i < min(m,n)) and @p alpha off it -- and returns without
/// synchronizing; A outside that region is left untouched. Enqueues nothing when
/// @p m or @p n is 0. A must live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the fill is enqueued on; A lives on its device
/// @param region Which part of A to set (Region::U, Region::L, Region::A)
/// @param m Number of rows of A
/// @param n Number of columns of A
/// @param alpha The constant written to the off-diagonal elements
/// @param beta The constant written to the diagonal elements
/// @param d_a Device matrix to set, column-major, leading dimension @p lda
/// @param lda Leading dimension of A; lda >= m
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status laset(const wwr::wwrStream_t stream, const Region region, const std::size_t m,
             const std::size_t n, const T alpha, const T beta, T *d_a, const std::size_t lda) {
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  device::laset(stream, region, m, n, alpha, beta, d_a, lda);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as gebal does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laset<float>(wwr::wwrStream_t, Region, std::size_t, std::size_t, float,
                                    float, float *, std::size_t);
extern template Status laset<double>(wwr::wwrStream_t, Region, std::size_t, std::size_t, double,
                                     double, double *, std::size_t);

} // namespace calaman
