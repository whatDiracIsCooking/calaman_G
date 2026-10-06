/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lat2 -- triangular matrix, double to
 *        single, LAPACK's ?lat2?
 *
 * Converts the UPLO triangle of the n-by-n column-major A to single precision
 * in SA: dlat2s and zlat2c (LAPACK has no slat2d / clat2z) as one template over
 * the (From, To) pair. The other triangle of SA is untouched. INFO is 1 if any
 * entry of the triangle (any real or imaginary part) lies outside
 * +-SLAMCH('O') = FLT_MAX, else 0; on INFO = 1 the content of SA is
 * unspecified, as in the reference. The element conversion and overflow check
 * are calaman.lag2's (lag2.cuh). Enqueued on the stream; returns WITHOUT
 * synchronizing; allocates nothing.
 *
 * INFO convention: calaman.lag2's -- a device int the caller owns and reads
 * back, as calaman.sterf and calaman.lahqr report theirs. Status carries only
 * argument and runtime errors.
 *
 * | DLAT2S / ZLAT2C   | calaman::lat2                                   |
 * |-------------------|-------------------------------------------------|
 * | CHARACTER UPLO    | calaman::Uplo enum                              |
 * | d/z variants      | lat2<From, To>, constrained by lat2_pair        |
 * | INTEGER extents   | std::size_t; LDA / LDSA kept                    |
 * | INFO              | device int @p d_info                            |
 *
 * Usage:
 *   import calaman.lat2;      // also re-exports calaman::Uplo and Status
 *   // d_a: n x n double (lda); d_sa: n x n float (ldsa); d_info: device int
 *   calaman::lat2<double, float>(stream, calaman::Uplo::U, n, d_a, lda, d_sa,
 *                                ldsa, d_info);
 */

module;

// CLM_TRY / CLM_REQUIRE arrive by #include in the GMF; lat2_bridge.h declares
// the .cu launcher there too.
#include "error_handling/error_macros.h"

#include "lat2_bridge.h"

export module calaman.lat2;

import std;
import wwr.runtime_api; // wwrStream_t, wwrMemsetAsync, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // Uplo

// export import: lat2 RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lat2;` alone names calaman::Uplo.
export using calaman::Uplo;

/// @brief The two (From, To) pairs LAPACK's ?lat2? covers
export template<typename From, typename To>
concept lat2_pair = (std::same_as<From, double> && std::same_as<To, float>) ||
                    (std::same_as<From, wwr::wwrDoubleComplex> &&
                     std::same_as<To, wwr::wwrFloatComplex>);

/// @brief SA <- A narrowed to single over the @p uplo triangle; INFO to @p d_info
///
/// Zeroes @p d_info, enqueues the conversion and returns without synchronizing.
/// @p d_info ends 1 if an entry of the triangle overflowed, else 0.
///
/// @tparam From, To A lat2_pair: source and destination element types
/// @param stream Stream the work is enqueued on; every pointer lives on its device
/// @param uplo Which triangle of A to convert (Uplo::U or Uplo::L)
/// @param d_a Source, n x n column-major, leading dimension @p lda >= n
/// @param d_sa Destination, n x n column-major, leading dimension @p ldsa >= n
/// @param d_info Device int: 0, or 1 on overflow
/// @return Success, an invalid-value Status, or the runtime error reported
export template<typename From, typename To>
  requires lat2_pair<From, To>
Status lat2(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n,
            const From *const d_a, const std::size_t lda, To *const d_sa,
            const std::size_t ldsa, int *const d_info) {
  CLM_REQUIRE(d_info != nullptr, wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(lda >= std::max<std::size_t>(1, n) && ldsa >= std::max<std::size_t>(1, n),
              wwr::wwrErrorInvalidValue);
  CLM_TRY(wwr::wwrMemsetAsync(d_info, 0, sizeof(int), stream));
  if (n == 0) {
    return wwr::wwrSuccess;
  }
  device::lat2(stream, uplo, n, d_a, lda, d_sa, ldsa, d_info);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lat2<double, float>(wwr::wwrStream_t, Uplo, std::size_t, const double *,
                                           std::size_t, float *, std::size_t, int *);
extern template Status lat2<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>(
    wwr::wwrStream_t, Uplo, std::size_t, const wwr::wwrDoubleComplex *, std::size_t,
    wwr::wwrFloatComplex *, std::size_t, int *);

} // namespace calaman
