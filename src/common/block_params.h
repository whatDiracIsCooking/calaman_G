/**
 * @file block_params.h
 * @brief Launch-shape constants and constraints (kWarpSize, num_warps) shared
 *        by device .cu code
 *
 * A plain header, not a module unit, as align_up.h is: block sizing is wanted
 * by device translation units, which cannot `import`. WWR_WARP_SIZE comes from
 * WarpWraps' runtime.h (32 by default, 64 on a CDNA build), reached through the
 * wwr.device include root calaman.common carries.
 *
 * Consumers #include this by its root-relative path, "common/block_params.h".
 */

#pragma once

#include <concepts>

#include "runtime.h" // WWR_WARP_SIZE

namespace calaman {

/// @brief The warp/wavefront size of the build's target, as an unsigned int.
///        `inline` gives it external linkage, which :block_params needs to
///        export it (a plain namespace-scope constexpr is internal).
inline constexpr unsigned int kWarpSize = static_cast<unsigned int>(WWR_WARP_SIZE);

/// @brief A warps-per-block count @p N: integral, >= 1, a power of two, and at
///        most 1024 threads once multiplied by kWarpSize. The bound is checked
///        as N <= 1024 / kWarpSize so a huge N cannot overflow the product.
///        Constrains a value, not a type: `template<auto W> requires num_warps<W>`.
template<auto N>
concept num_warps = std::integral<decltype(N)> && (N >= 1) && ((N & (N - 1)) == 0) &&
                    // The widest unsigned type, so a 64-bit N (e.g. 1ull << 32) is not
                    // truncated to 0 and let through; N >= 1 already ruled out negatives.
                    (static_cast<unsigned long long>(N) <= 1024 / kWarpSize);

} // namespace calaman
