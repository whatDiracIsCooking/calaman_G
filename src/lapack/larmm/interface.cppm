/**
 * @file interface.cppm
 * @brief Primary interface for calaman.larmm -- batched device ?larmm, the
 *        overflow-avoiding scale factor for C - A*B
 *
 * The scalar helper is device-only and header-only (larmm_scalar in
 * "lapack/larmm/larmm.cuh", reached by #include from a kernel, not by import).
 * This module exports the batched driver that exercises it on a card: n
 * independent ?larmm calls, enqueued on a stream without synchronizing. All
 * arrays are caller-owned device pointers; nothing is allocated.
 *
 * Usage:
 *   import calaman.larmm;     // also re-exports calaman::Status
 *   import wwr.runtime_api;
 *   calaman::larmm(stream, n, d_anorm, d_bnorm, d_cnorm, d_s);
 */

module;

// CLM_TRY arrives by #include (a macro); needs calaman::Status, which the export
// import below supplies.
#include "error_handling/error_macros.h"

#include "larmm_bridge.h"

export module calaman.larmm;

import std;
import wwr.runtime_api;

export import calaman.error_handling; // Status -- the return type

namespace calaman {

/// @brief Batch @p n ?larmm calls on @p stream: s = larmm(anorm, bnorm, cnorm)
///
/// For each k in [0, n) writes the scale s[k] in (0, 1] that keeps
/// s * (cnorm + anorm * bnorm) from overflowing. Returns success without
/// launching when @p n is 0. s may alias an input. All arrays live on
/// @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status larmm(const wwr::wwrStream_t stream, const std::size_t n, const T *anorm, const T *bnorm,
             const T *cnorm, T *s) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::larmm(stream, n, anorm, bnorm, cnorm, s);
  CLM_TRY(wwr::wwrGetLastError()); // the launcher is void; a bad launch is sticky
  return wwr::wwrSuccess;
}

extern template Status larmm<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                                    const float *, float *);
extern template Status larmm<double>(wwr::wwrStream_t, std::size_t, const double *,
                                     const double *, const double *, double *);

} // namespace calaman
