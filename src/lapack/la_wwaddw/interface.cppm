/**
 * @file interface.cppm
 * @brief Primary interface for calaman.la_wwaddw -- double-word accumulate,
 *        (x, y) <-- (x, y) + w
 *
 * Adds w into the double-word accumulator carried as the pair (x, y), x the
 * high word and y the low word, elementwise over length n: the GPU counterpart
 * of ?LA_WWADDW. Enqueued on a stream; returns WITHOUT synchronizing. Takes a
 * stream, not a handle, and allocates nothing (test/shared/README.md).
 *
 * The result is bitwise the reference loop's: every add and subtract is
 * rounded on its own, never contracted or re-associated (la_wwaddw.cu).
 *
 * Mapping from ?LA_WWADDW (docs/architecture.md §4): s/d/c/z become one
 * template over T, N becomes std::size_t. No INFO, as the reference has none.
 *
 * `extern template` pairs with instantiations.cpp, so an importer never
 * re-instantiates a body that names the GMF-declared .cu launcher.
 *
 * Usage:
 *   import calaman.la_wwaddw;   // also re-exports calaman::Status
 *   // d_x, d_y: device high/low words, length n; d_w: device addends
 *   calaman::la_wwaddw<double>(stream, n, d_x, d_y, d_w);
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the global
// module fragment; they need calaman::Status visible (the export import below).
#include "error_handling/error_macros.h"

#include "la_wwaddw_bridge.h"

export module calaman.la_wwaddw;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Add w into the double-word accumulator (x, y) on @p stream
///
/// Per element, s = x + w; s = (s + s) - s; y = ((x - s) + w) + y; x = s.
/// Enqueues nothing when @p n is 0; returns without synchronizing.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the update is enqueued on; x, y and w live on its device
/// @param n Number of elements
/// @param d_x Device high words, length n; updated in place
/// @param d_y Device low words, length n; updated in place
/// @param d_w Device addends, length n
/// @return Success, wwrErrorInvalidValue for a null pointer when n > 0, or the
///         runtime error the kernel launch reported
export template<typename T>
Status la_wwaddw(const wwr::wwrStream_t stream, const std::size_t n, T *const d_x, T *const d_y,
                 const T *const d_w) {
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(d_x != nullptr && d_y != nullptr && d_w != nullptr, wwr::wwrErrorInvalidValue);
  device::la_wwaddw<T>(stream, n, d_x, d_y, d_w);
  // The launcher returns void; the sticky launch error is the only report.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status la_wwaddw<float>(wwr::wwrStream_t, std::size_t, float *, float *,
                                        const float *);
extern template Status la_wwaddw<double>(wwr::wwrStream_t, std::size_t, double *, double *,
                                         const double *);
extern template Status la_wwaddw<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t,
                                                       wwr::wwrFloatComplex *,
                                                       wwr::wwrFloatComplex *,
                                                       const wwr::wwrFloatComplex *);
extern template Status la_wwaddw<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t,
                                                        wwr::wwrDoubleComplex *,
                                                        wwr::wwrDoubleComplex *,
                                                        const wwr::wwrDoubleComplex *);

} // namespace calaman
