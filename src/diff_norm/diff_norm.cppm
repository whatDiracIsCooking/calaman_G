/**
 * @file diff_norm.cppm
 * @brief Primary interface for calaman.diff_norm -- a chosen norm of y - x
 *
 * One host routine over BLAS level-1 calls: an axpy overwrites y with y - x, then
 * a reduction reports ||y - x|| in the Norm the template argument selects (ell_1 =
 * asum, ell_2 = nrm2, ell_inf = iamax plus a fetch of that element). It returns a
 * calaman::Status (calaman.error_handling), which carries each step's BLAS or
 * runtime code in its own domain. No device code lives here -- the ell_inf device
 * path calls calaman.set_element; this is a host composition over the outermost
 * WarpWraps layer (wwr.wrappers.blas).
 *
 * Constraints:
 *   - The axpy scalar is &kNegativeOne<T>, a host address, so the axpy is forced
 *     into host pointer mode and the caller's mode restored before the reduction.
 *   - The reduction then runs in the CALLER's mode, so `result` is a host pointer
 *     in host mode (blocks) and a device pointer in device mode (async).
 *   - `result` is spelled T, not the source type: for complex, asum/nrm2 return
 *     the real element type.
 *
 * Usage:
 *   import calaman.diff_norm;   // names calaman::diff_norm, Norm and Status
 *   import wwr.blas;            // wwrblasHandle_t, wwrblasCreate
 *   wwr::wwrblasHandle_t handle{};
 *   wwr::wwrblasCreate(&handle);
 *   // d_x, d_y: device vectors of length n; norm: a host scalar (host mode)
 *   float norm = 0.0f;
 *   const calaman::Status s =
 *       calaman::diff_norm<calaman::Norm::inf>(handle, n, d_x, 1, d_y, 1, &norm);
 *   if (!s.ok()) { ... report s.name() and s.message() ... }
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.diff_norm;

import wwr.blas;          // wwrblasHandle_t, WWRBLAS_STATUS_*, pointer mode, GetStream
import wwr.runtime_api;   // wwrStream_t, wwrMemcpyAsync/StreamSynchronize, wwrMallocAsync/FreeAsync
import wwr.wrappers.blas; // axpy, asum, nrm2, iamax
import wwr.extension.blas; // ScopedPointerMode (forces host mode for the axpy)
import calaman.common;    // kNegativeOne<T> (:constants), Norm (:enums)
import calaman.set_element; // set_element_abs (ell_inf's device-mode magnitude)

// export import, not plain import: consumers of this module must see Status's
// member definitions (ok/name/message), not just its name. Also re-exports
// PointerModeStatus (the :pointer_mode_policy partition) for the axpy guard below.
export import calaman.error_handling; // Status + PointerModeStatus

namespace calaman {

// Norm (ell_1 / ell_2 / ell_inf) now lives in calaman.common's :enums partition.
// Re-export it so `import calaman.diff_norm;` alone still names calaman::Norm, as
// the usage example above relies on.
export using calaman::Norm;

/// @brief Overwrite y with y - x, then report the chosen norm of the difference
///
/// Enqueues an axpy (scalar fixed at kNegativeOne<T>) then the reduction @p which
/// selects, writing ||y - x|| to @p result. Does nothing and returns success when
/// @p n is 0; short-circuits on a failing step, writing no norm. y always ends up
/// holding y - x, whether or not the norm was wanted.
///
/// @tparam which Norm to report (l1, l2, inf); non-deducible, so spell diff_norm<Norm::inf>(...)
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle; x and y live on its device, in either pointer mode
/// @param n Length of x and y
/// @param x Source device vector, stride @p incx
/// @param incx Stride between elements of x
/// @param y Device vector, updated in place to y - x, stride @p incy
/// @param incy Stride between elements of y
/// @param result ||y - x||; a host pointer in host pointer mode (blocks), a device
///        pointer in device pointer mode (async)
/// @return Status: success, or the first failing step's code, each in its own domain
export template<Norm which, typename T>
Status diff_norm(wwr::wwrblasHandle_t handle, const int n, const T *x, const int incx, T *y,
                 const int incy, T *result) {
  if (n == 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // The axpy scalar &kNegativeOne<T> is a host address, so force host pointer mode
  // for it. The braces are load-bearing: the guard restores the caller's mode on
  // block exit, and the reduction below must run in THAT mode, not host mode.
  wwr::wwrblasStatus_t pm_status = wwr::WWRBLAS_STATUS_SUCCESS;
  [[maybe_unused]] wwr::wwrblasPointerMode_t caller_mode{};
  {
    wwr::extension::ScopedPointerMode scope{handle, wwr::WWRBLAS_POINTER_MODE_HOST,
                                            PointerModeStatus{&pm_status}};
    caller_mode = scope.original_mode();
    // A failing axpy early-returns here; the guard still restores on the way out.
    CLM_TRY(wwr::axpy<T>(handle, n, &kNegativeOne<T>, x, incx, y, incy));
  }
  CLM_TRY(pm_status);   // the guard's get/set/restore, checked after it has run

  if constexpr (which == Norm::l1) {
    return wwr::asum<T>(handle, n, y, incy, result);
  } else if constexpr (which == Norm::l2) {
    return wwr::nrm2<T>(handle, n, y, incy, result);
  } else {
    // Both ell_inf paths need the handle's stream (device: alloc/launch; host: copy-back).
    wwr::wwrStream_t stream{};
    CLM_TRY(wwr::wwrblasGetStream(handle, &stream));
    if (caller_mode == wwr::WWRBLAS_POINTER_MODE_DEVICE) {
      // d_idx must be freed before any return, so iamax is NOT wrapped in the
      // early-returning CLM_TRY -- that would leak it. Its status is captured and
      // returned after the free. set_element_abs returns void (a kernel launch),
      // so there is nothing to check there.
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
}

} // namespace calaman
