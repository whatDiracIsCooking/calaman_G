/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lasrt -- sort a 1-D array into
 *        increasing or decreasing order, LAPACK's ?lasrt
 *
 * A single device routine: sort the length-@p n device array @p d in place, into
 * increasing (SortDir::I) or decreasing (SortDir::D) order. The sort is enqueued
 * on the given stream and returns WITHOUT synchronizing, like a BLAS call and
 * like calaman.laset; the caller synchronizes when it needs @p d. @p d is a
 * device pointer the caller owns; nothing is allocated here.
 *
 * A stream, not a device handle, is the whole requirement: this routine enqueues
 * one kernel and allocates nothing, so it needs no device index and no memory
 * pool, matching calaman.laset / calaman.lacpy. See test/shared/README.md.
 *
 * Mapping from LAPACK's DLASRT (docs/architecture.md §4 -- keep the name, drop
 * the Fortran calling convention):
 *
 * | DLASRT           | calaman::lasrt                          |
 * |------------------|-----------------------------------------|
 * | CHARACTER ID     | calaman::SortDir enum (no lsame char)   |
 * | s/d variants     | one template over T (float, double)     |
 * | INTEGER N        | kept: int (the sort's explicit stack)   |
 * | INTEGER INFO     | dropped: a calaman::Status return       |
 *
 * There is no INFO out-parameter: a typed SortDir cannot be an illegal ID
 * (DLASRT's INFO = -1 is unreachable), and a bad N returns a Status rather than
 * writing INFO = -2 -- the same trade the typed selectors make across src/.
 * Real only, like calaman.larfg: complex has no total order, so LAPACK ships no
 * c/z ?lasrt.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lasrt;      // also re-exports calaman::SortDir and Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_d: device array of n elements
 *   calaman::lasrt(stream, calaman::SortDir::I, n, d_d);
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the global
// module fragment, not by import. Resolved root-relative via the src/ root
// calaman.error_handling exports; needs calaman::Status visible at expansion,
// which the import below (export import) supplies.
#include "error_handling/error_macros.h"

#include "lasrt_bridge.h"

export module calaman.lasrt;

import std;
import wwr.runtime_api;
import calaman.common; // SortDir (:enums) -- the typed replacement for DLASRT ID

// export import, not a plain import: lasrt RETURNS calaman::Status, so a consumer
// of `import calaman.lasrt;` must see Status's member functions, not just its
// name -- the same re-export laset / diff_norm do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// SortDir (I / D) lives in calaman.common's :enums partition, shared with the
// device .cu through common/enums.h. Re-export it so `import calaman.lasrt;`
// alone still names calaman::SortDir, as the usage example and tests expect.
export using calaman::SortDir;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Sort the length-@p n device array @p d in place on @p stream (?lasrt)
///
/// Enqueues the sort of @p d into increasing (SortDir::I) or decreasing
/// (SortDir::D) order and returns without synchronizing; @p d must live on
/// @p stream's device. Returns an invalid-value Status when @p n < 0. Enqueues
/// nothing and succeeds when @p n <= 1 (already sorted); @p d is then left
/// untouched and may be null.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the sort is enqueued on; @p d lives on its device
/// @param id Sort direction (SortDir::I increasing, SortDir::D decreasing)
/// @param n Length of @p d; the sort's explicit stack bounds it as LAPACK does
/// @param d Device array of @p n elements, sorted in place
/// @return Success, or the runtime error the kernel launch reported; an
///         invalid-value Status when @p n < 0 or @p d is null with @p n > 1
export template<typename T>
Status lasrt(const wwr::wwrStream_t stream, const SortDir id, const int n, T *d) {
  CLM_REQUIRE(n >= 0, wwr::wwrErrorInvalidValue);
  if (n <= 1) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d != nullptr, wwr::wwrErrorInvalidValue);
  device::lasrt(stream, id, n, d);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laset does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lasrt<float>(wwr::wwrStream_t, SortDir, int, float *);
extern template Status lasrt<double>(wwr::wwrStream_t, SortDir, int, double *);

} // namespace calaman
