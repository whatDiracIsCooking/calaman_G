/**
 * @file interface.cppm
 * @brief Primary interface for calaman.laruv -- LAPACK's uniform (0,1)
 *        generator, ?laruv, bit for bit
 *
 * Draws n <= 128 uniform (0,1) values from the 48-bit multiplicative
 * congruential generator of slaruv/dlaruv and advances the seed, reproducing
 * the reference's output and updated ISEED bitwise. ISEED is a device array of
 * 4 ints the caller owns (each in [0, 4095], ISEED(4) odd). Enqueued on the
 * stream; returns WITHOUT synchronizing. Allocates nothing: one block of 128
 * threads, the multiplier table in constant memory (laruv.cuh).
 *
 * Mapping from DLARUV (docs/architecture.md §4): s/d become one template over
 * T; N > 128, which the reference documents as a precondition, is an
 * invalid-value Status; N = 0 enqueues nothing and leaves ISEED unchanged.
 *
 * Usage:
 *   import calaman.laruv;     // also re-exports calaman::Status
 *   // d_iseed: 4 device ints; d_x: n device values
 *   calaman::laruv<double>(stream, d_iseed, n, d_x);
 */

module;

// CLM_TRY / CLM_REQUIRE arrive by #include in the GMF; laruv_bridge.h declares
// the .cu launcher (and kLaruvMaxN) there too.
#include "error_handling/error_macros.h"

#include "laruv_bridge.h"

export module calaman.laruv;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError, wwrSuccess
import calaman.common;  // real_fp

// export import: laruv RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

/// @brief @p n uniform (0,1) draws from LAPACK's stream into @p d_x (?laruv)
///
/// Enqueues one kernel on @p stream and returns without synchronizing; reads
/// and then advances the device seed @p d_iseed exactly as the reference does.
///
/// @tparam T Element type (float, double)
/// @param stream Stream the launch is enqueued on; both pointers live on its device
/// @param d_iseed Device int[4]: the seed, overwritten by the advanced seed
/// @param n Number of draws, 0 <= n <= 128
/// @param d_x Device output, length n
/// @return Success, an invalid-value Status, or the launch's runtime error
export template<calaman::real_fp T>
Status laruv(const wwr::wwrStream_t stream, int *const d_iseed, const int n, T *const d_x) {
  CLM_REQUIRE(n >= 0 && n <= device::kLaruvMaxN, wwr::wwrErrorInvalidValue);
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d_iseed != nullptr && d_x != nullptr, wwr::wwrErrorInvalidValue);
  device::laruv(stream, d_iseed, n, d_x);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status laruv<float>(wwr::wwrStream_t, int *, int, float *);
extern template Status laruv<double>(wwr::wwrStream_t, int *, int, double *);

} // namespace calaman
