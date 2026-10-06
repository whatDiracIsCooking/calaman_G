/**
 * @file align_up.h
 * @brief Integer ceiling-divide and round-up-to-a-multiple (idivup, align_up),
 *        as a header shareable by device .cu code and module GMFs alike
 *
 * A plain header, not a module unit, for the same reason constants.h is: these
 * are `constexpr` helpers that both a device translation unit (a .cu, a plain TU
 * that cannot `import`) and a module's global module fragment want to reach --
 * grid and block sizing straddles the host/device line. The sibling align_up.cppm
 * includes this in its GMF and re-exports the names, so importers of
 * calaman.common see the same `calaman::` spellings as #includers.
 *
 * Backend-neutral by construction (CLAUDE.md): no cu-/hip-prefixed names appear.
 * Both helpers are CLM_HOST_DEVICE (host_device.h), so kernels call them too.
 *
 * Consumers #include this by its root-relative path, "common/align_up.h".
 */

#pragma once

#include <concepts>

#include "host_device.h"

namespace calaman {

// ========================================================================
// Integer rounding
// ========================================================================

/// @brief Ceiling of @p a / @p b: the number of @p b -sized chunks needed to
///        cover @p a. Assumes non-negative operands; undefined for @p b == 0,
///        and @p a + @p b may overflow near the top of T's range.
template <std::integral T>
CLM_HOST_DEVICE constexpr T idivup(const T a, const T b) {
    return (a + b - 1) / b;
}

/// @brief Round @p a up to the next multiple of @p b. Same domain as idivup:
///        non-negative operands, undefined for @p b == 0.
template <std::integral T>
CLM_HOST_DEVICE constexpr T align_up(const T a, const T b) {
    return idivup(a, b) * b;
}

} // namespace calaman
