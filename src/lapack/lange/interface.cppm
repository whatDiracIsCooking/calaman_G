/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lange -- the norm of a general matrix,
 *        LAPACK's ?lange
 *
 * Returns the 1-norm, infinity-norm, Frobenius norm, or largest absolute element
 * of an m-by-n column-major matrix A, selected by a MatrixNorm (as ?lange selects
 * on its NORM char). The result is written to a device scalar and the call
 * returns WITHOUT synchronizing; an empty matrix writes 0, matching DLANGE.
 *
 * A two-stage reduction (lange.cu) whose intermediate is device scratch this
 * allocates on @p stream (stream-ordered), hence the calaman::Status return.
 *
 * Mapping from DLANGE (docs/architecture.md §4): NORM becomes MatrixNorm; s/d/c/z
 * become one template over T, whose norm is always real (ComplexToRealType<T>);
 * WORK is the internal scratch. Divergence: the Frobenius norm is a plain sum of
 * squares, not DLANGE's scaled ?lassq -- correct for well-scaled inputs.
 *
 * `extern template` pairs with instantiations.cpp, so an importer never
 * re-instantiates a body that names the GMF-declared .cu launcher.
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
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // MatrixNorm (:enums), ComplexToRealType

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
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream   Stream the launches are enqueued on; A and result live on its device
/// @param which    Which matrix norm to compute (max_abs, one, inf, frobenius)
/// @param m        Number of rows of A
/// @param n        Number of columns of A
/// @param d_A      Source device matrix, column-major, leading dimension @p lda
/// @param lda      Leading dimension of @p d_A; lda >= m
/// @param d_result Device scalar receiving the (real) norm
/// @return Success, or the first failing step's error (allocation or launch)
export template<typename T>
Status lange(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t m,
             const std::size_t n, const T *const d_A, const std::size_t lda,
             ComplexToRealType<T> *const d_result) {
  using R = ComplexToRealType<T>;
  if (m == 0 || n == 0) {
    // DLANGE is 0 for an empty matrix; a zeroed scalar is that in IEEE.
    return wwr::wwrMemsetAsync(d_result, 0, sizeof(R), stream);
  }

  // The infinity norm reduces along rows, so its intermediate is one value per
  // row; the other three reduce along columns, one value per column.
  const std::size_t scratch_len = (which == MatrixNorm::inf) ? m : n;
  R *d_scratch = nullptr;
  CLM_TRY(
      wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_scratch), scratch_len * sizeof(R), stream));

  device::lange<T, R>(stream, which, m, n, d_A, lda, d_result, d_scratch);
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
extern template Status lange<wwr::wwrFloatComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   std::size_t, const wwr::wwrFloatComplex *,
                                                   std::size_t, float *);
extern template Status lange<wwr::wwrDoubleComplex>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                    std::size_t, const wwr::wwrDoubleComplex *,
                                                    std::size_t, double *);

} // namespace calaman
