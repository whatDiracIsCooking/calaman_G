/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lascl2 -- diagonal row-scaling of a
 *        matrix, x <-- D * x
 *
 * A single device routine: scale the m-by-n column-major matrix X in place by a
 * diagonal matrix D held as a length-m vector, so X(i,j) <-- d(i) * X(i,j). The
 * scaling is enqueued on the given stream and returns WITHOUT synchronizing,
 * like a BLAS call; the caller synchronizes when it needs X. D and X are device
 * pointers the caller owns; nothing is allocated here.
 *
 * A stream, not a device handle, is the whole requirement: this routine
 * enqueues one kernel and allocates nothing, so it needs no device index and no
 * memory pool. Taking a `wwrStream_t` also keeps a concrete handle type -- and
 * the error policy such a type hard-codes -- out of calaman's shipped surface,
 * matching calaman.laset's bare wwrStream_t. See test/shared/README.md.
 *
 * Mapping from LAPACK's DLASCL2 (docs/architecture.md §4 -- keep the name, drop
 * the Fortran calling convention): the s/d variants become one template over T,
 * the INTEGER extents become std::size_t, and LDX is kept -- a fact of
 * column-major storage, not a Fortran accommodation. There is no INFO; DLASCL2
 * reports none.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lascl2;    // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_d: length-m device vector; d_x: column-major device matrix, ld ldx
 *   calaman::lascl2(stream, m, n, d_d, d_x, ldx);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "lascl2_bridge.h"

export module calaman.lascl2;

import std;
import wwr.runtime_api;

// export import, not a plain import: lascl2 RETURNS calaman::Status, so a
// consumer of `import calaman.lascl2;` must see Status's member functions, not
// just its name -- the same re-export laset does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Scale the m-by-n column-major matrix X by a diagonal D on @p stream
///
/// Enqueues the in-place scaling X(i,j) <-- d(i) * X(i,j) and returns without
/// synchronizing. Enqueues nothing when @p m or @p n is 0. D and X must live on
/// @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the scaling is enqueued on; D and X live on its device
/// @param m Number of rows of X and length of D
/// @param n Number of columns of X
/// @param d_d Device diagonal vector, length m (per-row scale factors)
/// @param d_x Device matrix to scale in place, column-major, leading dim @p ldx
/// @param ldx Leading dimension of X; ldx >= m
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status lascl2(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n,
              const T *d_d, T *d_x, const std::size_t ldx) {
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  device::lascl2(stream, m, n, d_d, d_x, ldx);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as laset does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lascl2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                     float *, std::size_t);
extern template Status lascl2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                      double *, std::size_t);

} // namespace calaman
