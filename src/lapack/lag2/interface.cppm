/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lag2 -- general matrix, double <->
 *        single, LAPACK's ?lag2?
 *
 * Converts the m-by-n column-major A into SA in the other precision: dlag2s,
 * slag2d, zlag2c and clag2z as one template over the (From, To) pair. Narrowing
 * reproduces the reference's INFO: 1 if any entry (any real or imaginary part)
 * lies outside +-SLAMCH('O') = FLT_MAX, else 0; on INFO = 1 the content of SA
 * is unspecified, as in the reference. Widening always gives 0. Enqueued on
 * the stream; returns WITHOUT synchronizing; allocates nothing.
 *
 * INFO convention: a device int the caller owns and reads back, as calaman.sterf
 * and calaman.lahqr report theirs; a single int needs no named status block
 * (#211's fixed_struct). Status carries only argument and runtime errors.
 *
 * | DLAG2S / ZLAG2C   | calaman::lag2                                   |
 * |-------------------|-------------------------------------------------|
 * | s/d/c/z pairs     | lag2<From, To>, constrained by lag2_pair        |
 * | INTEGER extents   | std::size_t; LDA / LDSA kept                    |
 * | INFO              | device int @p d_info                            |
 *
 * Usage:
 *   import calaman.lag2;      // also re-exports calaman::Status
 *   // d_a: m x n double (lda); d_sa: m x n float (ldsa); d_info: device int
 *   calaman::lag2<double, float>(stream, m, n, d_a, lda, d_sa, ldsa, d_info);
 */

module;

// CLM_TRY / CLM_REQUIRE arrive by #include in the GMF; lag2_bridge.h declares
// the .cu launcher there too.
#include "error_handling/error_macros.h"

#include "lag2_bridge.h"

export module calaman.lag2;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex

// export import: lag2 RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

/// @brief The four (From, To) pairs LAPACK's ?lag2? covers
export template<typename From, typename To>
concept lag2_pair = (std::same_as<From, double> && std::same_as<To, float>) ||
                    (std::same_as<From, float> && std::same_as<To, double>) ||
                    (std::same_as<From, wwr::wwrDoubleComplex> &&
                     std::same_as<To, wwr::wwrFloatComplex>) ||
                    (std::same_as<From, wwr::wwrFloatComplex> &&
                     std::same_as<To, wwr::wwrDoubleComplex>);

/// @brief SA <- A converted to the other precision; INFO to @p d_info (?lag2?)
///
/// Zeroes @p d_info, enqueues the conversion and returns without synchronizing.
/// @p d_info ends 1 if a narrowing overflowed, else 0.
///
/// @tparam From, To A lag2_pair: source and destination element types
/// @param stream Stream the work is enqueued on; every pointer lives on its device
/// @param d_a Source, column-major, leading dimension @p lda >= m
/// @param d_sa Destination, column-major, leading dimension @p ldsa >= m
/// @param d_info Device int: 0, or 1 on overflow
/// @return Success, an invalid-value Status, or the runtime error reported
export template<typename From, typename To>
  requires lag2_pair<From, To>
Status lag2(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n,
            const From *const d_a, const std::size_t lda, To *const d_sa,
            const std::size_t ldsa, int *const d_info) {
  CLM_REQUIRE(d_info != nullptr, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(lda >= std::max<std::size_t>(1, m) && ldsa >= std::max<std::size_t>(1, m),
              wwr::wwrErrorInvalidValue);
  CLM_TRY(wwr::wwrMemsetAsync(d_info, 0, sizeof(int), stream));
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  device::lag2(stream, m, n, d_a, lda, d_sa, ldsa, d_info);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lag2<double, float>(wwr::wwrStream_t, std::size_t, std::size_t,
                                           const double *, std::size_t, float *, std::size_t,
                                           int *);
extern template Status lag2<float, double>(wwr::wwrStream_t, std::size_t, std::size_t,
                                           const float *, std::size_t, double *, std::size_t,
                                           int *);
extern template Status lag2<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>(
    wwr::wwrStream_t, std::size_t, std::size_t, const wwr::wwrDoubleComplex *, std::size_t,
    wwr::wwrFloatComplex *, std::size_t, int *);
extern template Status lag2<wwr::wwrFloatComplex, wwr::wwrDoubleComplex>(
    wwr::wwrStream_t, std::size_t, std::size_t, const wwr::wwrFloatComplex *, std::size_t,
    wwr::wwrDoubleComplex *, std::size_t, int *);

} // namespace calaman
