/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lasr -- apply a sequence of plane
 *        rotations to a general matrix, LAPACK's ?lasr
 *
 * Applies the k-1 plane rotations (c[j], s[j]) -- exactly what calaman.lartg
 * produces -- to the m-by-n column-major A from the left (A <- P A, k = m) or
 * the right (A <- A P^T, k = n), in the planes Pivot selects and the order
 * Direct selects. Enqueued on the stream; returns WITHOUT synchronizing and
 * allocates nothing (one launch, lasr.cu).
 *
 * Mapping from DLASR (docs/architecture.md §4): SIDE/PIVOT/DIRECT become the
 * calaman::Side/Pivot/Direct enums; s/d become one template over T; INTEGER
 * extents become std::size_t.
 *
 * A kernel that applies the rotations from inside its own launch (?steqr) does
 * not call this: it #includes "lapack/lasr/lasr.h" for the block-cooperative
 * lasr_block.
 *
 * Usage:
 *   import calaman.lasr;  // also re-exports Side, Pivot, Direct and Status
 *   // d_c, d_s: m-1 device rotations; d_A: m-by-n device matrix, leading dim lda
 *   calaman::lasr<double>(stream, calaman::Side::L, calaman::Pivot::V,
 *                         calaman::Direct::F, m, n, d_c, d_s, d_A, lda);
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the GMF.
#include "error_handling/error_macros.h"

#include "lasr_bridge.h"

export module calaman.lasr;

import std;
import wwr.runtime_api;
import calaman.common; // Side, Pivot, Direct (:enums)

// export import: lasr RETURNS calaman::Status, so a consumer must see its
// members, not just its name.
export import calaman.error_handling;

namespace calaman {

// Re-exported so `import calaman.lasr;` alone names the three selectors.
export using calaman::Direct;
export using calaman::Pivot;
export using calaman::Side;

/// @brief Apply ?lasr's k-1 plane rotations to the m-by-n A on @p stream
///
/// Enqueues nothing and succeeds when m or n is 0, or when the rotated
/// dimension k (m for Side::L, n for Side::R) is 1 -- @p c and @p s are then
/// unread. Rotation j is skipped when c[j] == 1 and s[j] == 0, as in DLASR.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the work is enqueued on; all pointers live on its device
/// @param c, s   Device cosines and sines of the k-1 rotations
/// @param d_A    Device matrix, column-major, updated in place
/// @param lda    Leading dimension of A; lda >= max(1, m)
/// @return Success, the launch error, or invalid-value when lda < max(1, m)
export template<typename T>
Status lasr(const wwr::wwrStream_t stream, const Side side, const Pivot pivot,
            const Direct direct, const std::size_t m, const std::size_t n, const T *const c,
            const T *const s, T *const d_A, const std::size_t lda) {
  CLM_REQUIRE(lda >= std::max<std::size_t>(1, m), wwr::wwrErrorInvalidValue);
  const std::size_t k = side == Side::L ? m : n;
  if (m == 0 || n == 0 || k < 2) {
    return wwr::wwrSuccess;
  }
  device::lasr<T>(stream, side, pivot, direct, m, n, c, s, d_A, lda);
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lasr<float>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                   std::size_t, const float *, const float *, float *,
                                   std::size_t);
extern template Status lasr<double>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                    std::size_t, const double *, const double *, double *,
                                    std::size_t);

} // namespace calaman
