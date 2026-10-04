/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lange -- the norm of a general matrix,
 *        LAPACK's ?lange
 *
 * Returns the 1-norm, infinity-norm, Frobenius norm, or largest absolute element
 * of an m-by-n column-major matrix A, selected at runtime by a MatrixNorm (as
 * ?lange selects on its NORM char). The result is written to a device scalar and
 * the call returns WITHOUT synchronizing, like a BLAS call; the caller
 * synchronizes when it needs the value. An empty matrix (m or n zero) writes 0,
 * matching DLANGE.
 *
 * The computation is a two-stage reduction (lange.cu): A to a per-column (or
 * per-row) intermediate, then that to one scalar. The intermediate needs device
 * scratch, so -- unlike the stream-only, allocate-nothing columnwise_* modules --
 * this allocates it on @p stream (wwrMallocAsync/FreeAsync, stream-ordered) and
 * therefore returns a calaman::Status, the way calaman.gebal does for its own
 * scratch. A bare wwrStream_t, not a device handle, is still the whole handle
 * requirement: no device index and no memory pool beyond that one buffer.
 *
 * Mapping from DLANGE (docs/architecture.md §4 -- keep the name, drop the Fortran
 * calling convention): CHARACTER NORM becomes the MatrixNorm enum; the s/d/c/z
 * variants become one template over T (float, double); the WORK array DLANGE asks
 * for the infinity norm is the internally-allocated scratch. The Frobenius norm
 * uses a plain sum of squares (not DLANGE's scaled DLASSQ), matching
 * calaman.columnwise_ell2 -- well-scaled inputs only, a deliberate simplification.
 *
 * Templated over float and double; the complex matrix norm is a T -> real
 * reduction, a deliberate later extension, as calaman.diff_norm notes.
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lange;      // also re-exports calaman::MatrixNorm and Status
 *   import wwr.runtime_api;    // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_A: m-by-n device matrix, column-major, leading dimension lda
 *   // d_result: device scalar
 *   calaman::lange<double>(stream, calaman::MatrixNorm::one, m, n, d_A, lda, d_result);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include "lange_bridge.h"

export module calaman.lange;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMallocAsync/FreeAsync/MemsetAsync, wwrGetLastError
import calaman.common;  // MatrixNorm (:enums) -- the typed replacement for DLANGE NORM

// export import, not a plain import: lange RETURNS calaman::Status, so a consumer
// of `import calaman.lange;` must see Status's member functions, not just its
// name -- the same re-export diff_norm and lacpy do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// MatrixNorm (max_abs / one / inf / frobenius) lives in calaman.common's :enums
// partition, shared with the device .cu through common/enums.h. Re-export it so
// `import calaman.lange;` alone still names calaman::MatrixNorm, as lacpy does for
// Region.
export using calaman::MatrixNorm;

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declaration below sits in the plain namespace.

/// @brief Norm of the m-by-n column-major matrix A on @p stream (?lange)
///
/// Writes the @p which norm of A to @p d_result and returns without
/// synchronizing. Writes 0 when @p m or @p n is 0. Allocates and frees a
/// per-column/per-row scratch buffer on @p stream, so the return carries any
/// allocation or launch failure. A and result must live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream   Stream the launches are enqueued on; A and result live on its device
/// @param which    Which matrix norm to compute (max_abs, one, inf, frobenius)
/// @param m        Number of rows of A
/// @param n        Number of columns of A
/// @param d_A      Source device matrix, column-major, leading dimension @p lda
/// @param lda      Leading dimension of @p d_A; lda >= m
/// @param d_result Device scalar receiving the norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lange(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t m,
             const std::size_t n, const T *const d_A, const std::size_t lda, T *const d_result) {
  if (m == 0 || n == 0) {
    // DLANGE is 0 for an empty matrix; a zeroed scalar is that in IEEE.
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(T), stream);
  }

  // The infinity norm reduces along rows, so its intermediate is one value per
  // row; the other three reduce along columns, one value per column.
  const std::size_t scratch_len = (which == MatrixNorm::inf) ? m : n;
  T *d_scratch = nullptr;
  CLM_TRY(
      wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), scratch_len * sizeof(T), stream));

  device::lange<T>(stream, which, m, n, d_A, lda, d_result, d_scratch);
  // The launcher returns void; its sticky launch error is the only way to catch a
  // bad launch. Capture it BEFORE the free so a free failure cannot mask it, then
  // free unconditionally -- the free is stream-ordered after the kernels, and a
  // cleanup-time failure is not fatal, so it is discarded, as diff_norm does.
  const wwr::wwrError_t launch_status = wwr::wwrGetLastError();
  static_cast<void>(wwr::wwrFreeAsync(d_scratch, stream));
  return launch_status;
}

extern template Status lange<float>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                    const float *, std::size_t, float *);
extern template Status lange<double>(wwr::wwrStream_t, MatrixNorm, std::size_t, std::size_t,
                                     const double *, std::size_t, double *);

} // namespace calaman
