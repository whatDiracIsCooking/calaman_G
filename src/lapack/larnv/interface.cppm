/**
 * @file interface.cppm
 * @brief Primary interface for calaman.larnv -- a random vector from LAPACK's
 *        stream, ?larnv
 *
 * Fills n device values from distribution IDIST -- 1 uniform (0,1), 2 uniform
 * (-1,1), 3 normal (0,1) by Box-Muller; complex adds 4 uniform on the unit disk
 * and 5 uniform on the unit circle -- and advances ISEED, reproducing the
 * reference's chunking: IDIST 1/2 and ISEED bit for bit, IDIST 3/4/5 to device
 * log/sqrt/cos/sin accuracy. ISEED is a device int[4] the caller owns (each in
 * [0, 4095], ISEED(4) odd). Enqueued on the stream; returns WITHOUT
 * synchronizing; allocates nothing. Not wwrrandGenerate*: that is faster but
 * is not LAPACK's sequence.
 *
 * Mapping from DLARNV (docs/architecture.md §4): s/d/c/z become one template
 * over T; an IDIST outside the type's range is an invalid-value Status, as is
 * an n of 2^37 or more (2^31 chunks).
 *
 * Usage:
 *   import calaman.larnv;     // also re-exports calaman::Status
 *   // d_iseed: 4 device ints; d_x: n device values
 *   calaman::larnv<double>(stream, 3, d_iseed, n, d_x);
 */

module;

// CLM_TRY / CLM_REQUIRE arrive by #include in the GMF; larnv_bridge.h declares
// the .cu launcher there too.
#include "error_handling/error_macros.h"

#include "larnv_bridge.h"

export module calaman.larnv;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError, wwrSuccess
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // usual_fp, complex_fp

// export import: larnv RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

/// @brief @p n values of distribution @p idist from LAPACK's stream (?larnv)
///
/// Enqueues the work on @p stream and returns without synchronizing; reads and
/// then advances the device seed @p d_iseed exactly as the reference does.
///
/// @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex)
/// @param stream Stream the work is enqueued on; both pointers live on its device
/// @param idist 1..3 for a real T, 1..5 for a complex T (see the file header)
/// @param d_iseed Device int[4]: the seed, overwritten by the advanced seed
/// @param n Number of values; n == 0 enqueues nothing and leaves the seed
/// @param d_x Device output, length n
/// @return Success, an invalid-value Status, or the launch's runtime error
export template<calaman::usual_fp T>
Status larnv(const wwr::wwrStream_t stream, const int idist, int *const d_iseed,
             const std::size_t n, T *const d_x) {
  CLM_REQUIRE(idist >= 1 && idist <= (calaman::complex_fp<T> ? 5 : 3),
              wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(n < (std::size_t{1} << 37), wwr::wwrErrorInvalidValue);
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d_iseed != nullptr && d_x != nullptr, wwr::wwrErrorInvalidValue);
  device::larnv(stream, idist, d_iseed, n, d_x);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status larnv<float>(wwr::wwrStream_t, int, int *, std::size_t, float *);
extern template Status larnv<double>(wwr::wwrStream_t, int, int *, std::size_t, double *);
extern template Status larnv<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int *, std::size_t,
                                                   wwr::wwrFloatComplex *);
extern template Status larnv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int *, std::size_t,
                                                    wwr::wwrDoubleComplex *);

} // namespace calaman
