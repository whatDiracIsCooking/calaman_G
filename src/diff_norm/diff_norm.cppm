/**
 * @file diff_norm.cppm
 * @brief Primary interface for calaman.diff_norm -- a chosen norm of y - x
 *
 * One host routine composed from BLAS level-1 calls on the device: an axpy that
 * overwrites y with y - x, then a reduction that reports a norm of that
 * difference. The reduction is selected by a Norm enum -- ell_1 (sum of
 * magnitudes), ell_2 (Euclidean), or ell_inf (largest magnitude). The result is
 * ||y - x|| in the chosen norm, the distance between two vectors. The axpy scalar
 * is fixed at -1, taken from calaman.common as kNegativeOne<T>, so
 * there is no alpha to pass.
 *
 * The three norms are NOT symmetric in BLAS, and ell_inf is the odd one. ell_1 is
 * asum and ell_2 is nrm2 -- each a single call that writes the norm to `result`.
 * ell_inf's BLAS primitive is iamax, which writes the INDEX of the
 * largest-magnitude element, so this reads that one element back from the device
 * and takes its magnitude. That costs a synchronizing device->host copy and the
 * wwr.runtime_api dependency, and it crosses error domains: iamax reports a BLAS
 * status but the copy reports a runtime error, which -- WarpWraps exposing no
 * neutral "internal error" BLAS code -- is surfaced as WWRBLAS_STATUS_NOT_INITIALIZED.
 *
 * No device code lives here. asum/nrm2/iamax/axpy are stock BLAS, so this module
 * reaches for the OUTERMOST WarpWraps layer that does the job -- the type-safe
 * dispatch wrappers in `wwr.wrappers.blas` (see CLAUDE.md, "prefer the outermost
 * layer"). There is nothing backend-specific to write: no .cu, no launcher
 * bridge, no explicit-instantiation unit, unlike calaman.lacpy which owns a kernel.
 *
 * Requires the handle's DEFAULT (host) cuBLAS/hipBLAS pointer mode. x and y are
 * device pointers the caller owns; the -1 scalar is passed as &kNegativeOne<T>,
 * a HOST address, which is only correct in host pointer mode -- so @p result is
 * a host pointer too, and the reduction blocks until it is written. Either way y
 * is left holding y - x.
 *
 * Templated over `float` and `double`. Complex would add `wwrFloatComplex` /
 * `wwrDoubleComplex`, whose asum/nrm2 return the real element type -- the reason
 * the result is spelled T here rather than reused as the source type.
 *
 * Usage:
 *   import calaman.diff_norm;
 *   import wwr.blas;   // wwrblasHandle_t, wwrblasCreate, WWRBLAS_STATUS_SUCCESS
 *   wwr::wwrblasHandle_t handle{};
 *   wwr::wwrblasCreate(&handle);
 *   // d_x, d_y: device vectors of length n; norm: a host scalar
 *   float norm = 0.0f;
 *   calaman::diff_norm(handle, calaman::Norm::inf, n, d_x, 1, d_y, 1, &norm);
 */

export module calaman.diff_norm;

import wwr.blas;               // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_STATUS_*
import wwr.runtime_api;        // wwrMemcpy, wwrMemcpyDeviceToHost, wwrSuccess (ell_inf fetch)
import wwr.wrappers.blas;      // axpy, asum, nrm2, iamax
import calaman.common;         // kNegativeOne<T> (:constants), Norm (:enums)

namespace calaman {

// Norm (ell_1 / ell_2 / ell_inf) now lives in calaman.common's :enums partition.
// Re-export it so `import calaman.diff_norm;` alone still names calaman::Norm, as
// the usage example above relies on.
export using calaman::Norm;

/// @brief Overwrite y with y - x, then report the chosen norm of the difference
///
/// Enqueues an axpy (y := -x + y, the scalar fixed at kNegativeOne<T>) followed
/// by the reduction @p which selects: asum for l1, nrm2 for l2, or iamax plus a
/// device read for inf. Writes the norm to @p result. Short-circuits: if the axpy
/// (or, for inf, the iamax) does not succeed its status is returned and no norm is
/// written. Does nothing and returns success when @p n is 0. y always ends up
/// holding y - x, whether or not the caller wanted the norm.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; x and y live on its device
/// @param which Which norm to report (l1, l2, inf)
/// @param n Length of x and y
/// @param x Source device vector, stride @p incx
/// @param incx Stride between elements of x
/// @param y Device vector, updated in place to y - x, stride @p incy
/// @param incy Stride between elements of y
/// @param result Host scalar; ||y - x|| in norm @p which is written here (blocks)
/// @return The BLAS status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
///         (for inf, WWRBLAS_STATUS_NOT_INITIALIZED if the device read failed)
export template<typename T>
wwr::wwrblasStatus_t diff_norm(wwr::wwrblasHandle_t handle, const Norm which, const int n,
                               const T *x, const int incx, T *y, const int incy, T *result) {
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  const auto status = wwr::axpy<T>(handle, n, &kNegativeOne<T>, x, incx, y, incy);
  if (status != wwr::WWRBLAS_STATUS_SUCCESS) {
    return status;
  }

  // if/else so the function always returns without inventing a status for an
  // out-of-enum value -- l2 is the fallthrough. WarpWraps exposes no neutral
  // "invalid argument" code.
  if (which == Norm::l1) {
    return wwr::asum<T>(handle, n, y, incy, result);
  }
  if (which == Norm::inf) {
    // iamax gives the 1-based index of the largest-magnitude element, not the
    // value, so read that one element back (host pointer mode has already synced
    // it) and take its magnitude.
    int idx = 0;
    const auto imax_status = wwr::iamax<T>(handle, n, y, incy, &idx);
    if (imax_status != wwr::WWRBLAS_STATUS_SUCCESS) {
      return imax_status;
    }
    T elem{};
    if (wwr::wwrMemcpy(&elem, y + (idx - 1) * incy, sizeof(T), wwr::wwrMemcpyDeviceToHost) !=
        wwr::wwrSuccess) {
      return wwr::WWRBLAS_STATUS_NOT_INITIALIZED; // the only neutral non-success code exposed
    }
    *result = elem < T{0} ? -elem : elem;
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  return wwr::nrm2<T>(handle, n, y, incy, result); // Norm::l2
}

} // namespace calaman
