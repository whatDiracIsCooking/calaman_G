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
 * largest-magnitude element, not the value, so the value has to be fetched from
 * that index -- and how depends on the handle's pointer mode (below).
 *
 * WORKS IN EITHER POINTER MODE. The axpy scalar &kNegativeOne<T> is a HOST
 * address, correct only in host pointer mode, so this forces host mode for the
 * axpy with a wwr::extension::ScopedPointerMode guard and restores the caller's
 * mode when the guard leaves scope -- exactly as calaman.expm does for its
 * host-scalar gemms. The reductions then run in the CALLER's mode, so asum / nrm2
 * write `result` to wherever that mode dictates: host memory in host mode, device
 * memory in device mode. The guard's original_mode() is how the ell_inf path
 * learns which it is.
 *
 * ell_inf then splits on that mode:
 *   - host mode:   iamax's index comes back on the host (synced), so one element
 *                  is copied back device->host on the handle's stream
 *                  (wwrMemcpyAsync + a StreamSynchronize that keeps the blocking
 *                  contract) and its magnitude is taken on the host.
 *   - device mode: iamax writes its index to device memory and `result` is a
 *                  device pointer, so there is no host read -- a device int for
 *                  the index is allocated on the handle's own stream (via
 *                  wwrblasGetStream), iamax fills it, and calaman.set_element's
 *                  set_element_abs turns that device index into the magnitude at
 *                  `result`, entirely on the device.
 *
 * It crosses error domains: iamax reports a BLAS status, while the stream query,
 * the allocation and the copy report runtime errors. Every step returns a
 * calaman::Status (calaman.error_handling) via CLM_TRY, which carries each code
 * in its OWN domain -- so a runtime failure is reported as a runtime error, not
 * disguised as the stand-in BLAS code a wwrblasStatus_t return once forced. No
 * device code lives here -- set_element_abs is the kernel-owning module; this
 * stays a host composition over the OUTERMOST WarpWraps layer (wwr.wrappers.blas)
 * plus it.
 *
 * Templated over `float` and `double`. Complex would add `wwrFloatComplex` /
 * `wwrDoubleComplex`, whose asum/nrm2 return the real element type -- the reason
 * the result is spelled T here rather than reused as the source type.
 *
 * Usage:
 *   import calaman.diff_norm;   // names calaman::diff_norm, Norm and Status
 *   import wwr.blas;            // wwrblasHandle_t, wwrblasCreate
 *   wwr::wwrblasHandle_t handle{};
 *   wwr::wwrblasCreate(&handle);
 *   // d_x, d_y: device vectors of length n; norm: a host scalar (host mode)
 *   float norm = 0.0f;
 *   const calaman::Status s =
 *       calaman::diff_norm(handle, calaman::Norm::inf, n, d_x, 1, d_y, 1, &norm);
 *   if (!s.ok()) { ... report s.name() and s.message() ... }
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.diff_norm;

import std;               // std::source_location (the pointer-mode policy)
import wwr.blas;          // wwrblasHandle_t, WWRBLAS_STATUS_*, pointer mode, GetStream
import wwr.runtime_api;   // wwrStream_t, wwrMemcpyAsync/StreamSynchronize, wwrMallocAsync/FreeAsync
import wwr.wrappers.blas; // axpy, asum, nrm2, iamax
import wwr.extension.blas; // ScopedPointerMode (forces host mode for the axpy)
import calaman.common;    // kNegativeOne<T> (:constants), Norm (:enums)
import calaman.set_element; // set_element_abs (ell_inf's device-mode magnitude)

// export import, not a plain import: diff_norm RETURNS calaman::Status, and
// Status carries member functions (ok/name/message). A consumer of
// `import calaman.diff_norm;` must see those definitions, not just the type
// name, so the whole module is re-exported -- unlike Norm below, a plain enum
// whose name alone (export using) is enough.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Norm (ell_1 / ell_2 / ell_inf) now lives in calaman.common's :enums partition.
// Re-export it so `import calaman.diff_norm;` alone still names calaman::Norm, as
// the usage example above relies on.
export using calaman::Norm;

// A pointer-mode error policy for the ScopedPointerMode guard that RECORDS the
// first failing status rather than aborting or throwing: calaman ships no abort
// policy (test/shared owns that -- see the handle/policy boundary), and a
// get/set/restore failure must surface as diff_norm's own return value. Not
// exported -- module linkage is enough for the diff_norm template body to name
// it. handle_error is noexcept so the guard's restore, which runs from a
// destructor, cannot terminate.
struct PointerModeStatus {
  using error_type = wwr::wwrblasStatus_t;
  wwr::wwrblasStatus_t *first_error;
  void handle_error(const wwr::wwrblasStatus_t status,
                    const std::source_location = std::source_location::current()) noexcept {
    if (status != wwr::WWRBLAS_STATUS_SUCCESS && *first_error == wwr::WWRBLAS_STATUS_SUCCESS) {
      *first_error = status;
    }
  }
};

/// @brief Overwrite y with y - x, then report the chosen norm of the difference
///
/// Enqueues an axpy (y := -x + y, the scalar fixed at kNegativeOne<T>) followed
/// by the reduction @p which selects: asum for l1, nrm2 for l2, or iamax plus a
/// fetch of that element for inf. Writes the norm to @p result. Works in either
/// pointer mode: the axpy runs under a forced host mode (its scalar is a host
/// constant) and is restored, then the reduction runs in the caller's mode, so
/// @p result is written to host memory in host mode and device memory in device
/// mode. Short-circuits on a failing step and writes no norm. Does nothing and
/// returns success when @p n is 0. y always ends up holding y - x, whether or
/// not the caller wanted the norm.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle; x and y live on its device, in either pointer mode
/// @param which Which norm to report (l1, l2, inf)
/// @param n Length of x and y
/// @param x Source device vector, stride @p incx
/// @param incx Stride between elements of x
/// @param y Device vector, updated in place to y - x, stride @p incy
/// @param incy Stride between elements of y
/// @param result ||y - x|| in norm @p which; a host pointer in host pointer mode
///        (blocks), a device pointer in device pointer mode (async)
/// @return A Status: success, or the status of the failing step -- the BLAS
///         status for axpy/asum/nrm2/iamax, or the runtime error for the
///         stream/alloc/copy calls, each carried in its own domain
export template<typename T>
Status diff_norm(wwr::wwrblasHandle_t handle, const Norm which, const int n, const T *x,
                 const int incx, T *y, const int incy, T *result) {
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // axpy (y := y - x). Its alpha is &kNegativeOne<T>, a HOST address, so force
  // host pointer mode for it no matter how the caller left the handle; the guard
  // records the caller's mode and restores it when this block ends. In host
  // pointer mode the scalar is consumed during the call, so restoring it
  // afterwards is safe.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  wwr::wwrblasStatus_t axpy_status = wwr::WWRBLAS_STATUS_SUCCESS;
  wwr::wwrblasPointerMode_t caller_mode{};
  {
    wwr::extension::ScopedPointerMode scope{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                            PointerModeStatus{&pm_status}};
    caller_mode = scope.original_mode();
    axpy_status = wwr::axpy<T>(handle, n, &kNegativeOne<T>, x, incx, y, incy);
  }
  CLM_TRY(pm_status);   // the guard's get/set/restore
  CLM_TRY(axpy_status);

  // The reduction runs in the caller's restored mode, so asum/nrm2 write `result`
  // where that mode wants it. if/else so an out-of-enum value falls through to l2
  // rather than inventing a status -- WarpWraps exposes no neutral "invalid
  // argument" code.
  if (which == Norm::l1) {
    return wwr::asum<T>(handle, n, y, incy, result);
  }
  if (which == Norm::inf) {
    // Both ell_inf paths work on the handle's own stream -- the device path to
    // allocate and launch on it, the host path to order its copy-back against the
    // iamax that produced the index.
    wwr::wwrStream_t stream{};
    CLM_TRY(wwr::wwrblasGetStream(handle, &stream));
    if (caller_mode == wwr::WWRBLAS_POINTER_MODE_DEVICE) {
      // Device mode: iamax writes its 1-based index to the device and `result` is
      // a device pointer. Allocate a device int for the index on that stream, run
      // iamax, and let set_element_abs read the index and write the magnitude to
      // `result` -- all on the device, no host round trip. iamax is NOT wrapped in
      // CLM_TRY: d_idx must be freed first, so its status is captured and returned
      // after the free.
      int *d_idx = nullptr;
      CLM_TRY(wwr::wwrMallocAsync(reinterpret_cast<void **>(&d_idx), sizeof(int), stream));
      const auto imax_status = wwr::iamax<T>(handle, n, y, incy, d_idx);
      if (imax_status == wwr::WWRBLAS_STATUS_SUCCESS) {
        set_element_abs<T>(stream, y, incy, d_idx, result);
      }
      // Free is stream-ordered after the kernel; a cleanup-time failure is not
      // fatal and would only mask the status we already have, so discard it.
      static_cast<void>(wwr::wwrFreeAsync(d_idx, stream));
      return imax_status;
    }
    // Host mode: `result` is a host pointer and iamax's index came back on the
    // host. Copy that one element back on the handle's stream, then synchronize
    // before reading it on the host -- the sync is what keeps the host-mode
    // contract that `result` is populated on return.
    int idx = 0;
    CLM_TRY(wwr::iamax<T>(handle, n, y, incy, &idx));
    T elem{};
    CLM_TRY(wwr::wwrMemcpyAsync(&elem, y + (idx - 1) * incy, sizeof(T),
                                wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));
    *result = elem < T{0} ? -elem : elem;
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  return wwr::nrm2<T>(handle, n, y, incy, result); // Norm::l2
}

} // namespace calaman
