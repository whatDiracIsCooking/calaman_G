/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lapy3 -- batched device ?lapy3, the
 *        overflow-safe sqrt(x^2 + y^2 + z^2)
 *
 * The scalar helper is device-only and header-only (lapy3_scalar in
 * "lapack/lapy3/lapy3.cuh", reached by #include from a kernel, not by import).
 * This module exports the batched driver that exercises it on a card: n
 * independent ?lapy3 calls, enqueued on a stream without synchronizing. All
 * arrays are caller-owned device pointers; nothing is allocated.
 *
 * Usage:
 *   import calaman.lapy3;     // also re-exports calaman::Status
 *   import wwr.runtime_api;
 *   calaman::lapy3(stream, n, d_x, d_y, d_z, d_r);
 */

module;

// CLM_TRY arrives by #include (a macro); needs calaman::Status, which the export
// import below supplies.
#include "error_handling/error_macros.h"

#include "lapy3_bridge.h"

export module calaman.lapy3;

import std;
import wwr.runtime_api;

export import calaman.error_handling; // Status -- the return type

namespace calaman {

/// @brief Batch @p n ?lapy3 calls on @p stream: r = sqrt(x^2 + y^2 + z^2)
///
/// For each k in [0, n) writes r[k] without spurious overflow; a NaN input
/// yields NaN. Returns success without launching when @p n is 0. r may alias
/// x, y or z. All arrays live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @return Success, or the runtime error the kernel launch reported
export template<typename T>
Status lapy3(const wwr::wwrStream_t stream, const std::size_t n, const T *x, const T *y,
             const T *z, T *r) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::lapy3(stream, n, x, y, z, r);
  CLM_TRY(wwr::wwrGetLastError()); // the launcher is void; a bad launch is sticky
  return wwr::wwrSuccess;
}

extern template Status lapy3<float>(wwr::wwrStream_t, std::size_t, const float *, const float *,
                                    const float *, float *);
extern template Status lapy3<double>(wwr::wwrStream_t, std::size_t, const double *,
                                     const double *, const double *, double *);

} // namespace calaman
