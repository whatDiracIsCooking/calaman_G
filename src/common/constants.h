/**
 * @file constants.h
 * @brief Typed mathematical constants (kZero, kOne, ... kPi) over the element
 *        types, as a header shareable by device .cu code and module GMFs alike
 *
 * A plain header, not a module unit, for the same reason calaman.lacpy's
 * lacpy_bridge.h is: it must be #includable from BOTH a device translation unit
 * (a .cu, which is a plain TU) and a module's global module fragment (which
 * cannot `import`). Everything here is `inline constexpr`, so it is usable in
 * host and device code and needs no separate definition.
 *
 * Backend-neutral by construction (CLAUDE.md): no cu- or hip-prefixed vendor
 * names appear. That is also why this is float/double only. Complex would need
 * wwrFloatComplex / wwrDoubleComplex, and both walls that block them are real:
 *
 *   1. The types are reachable only from `import wwr.complex` (a module, which a
 *      GMF cannot import) or from complex.cuh -- and complex.cuh is device-only:
 *      it includes device_guard.h, which #errors outside a __CUDACC__/__HIP__
 *      pass, so a host compile of constants.cppm cannot pull it in.
 *   2. Even past that, there is no portable `constexpr` spelling of a complex
 *      literal: cuFloatComplex is a brace-initialisable float2 but hipFloatComplex
 *      is a class, so `{1.0f, 0.0f}` compiles on CUDA and breaks on HIP, and the
 *      portable make_gpu*Complex is __device__ __forceinline__ -- not usable in a
 *      host constexpr. WarpWraps documents this in complex.cuh's header.
 *
 * So complex is the same deliberate later extension it is for lacpy and vec_diff:
 * it belongs in a device-compiled `.cuh` reached through the backend switch, not
 * as a constexpr specialization here.
 *
 * Consumers include this by its root-relative path, `"common/constants.h"`; the
 * sibling constants.cppm includes it bare and re-exports these names so importers
 * of `calaman.common` see the same `calaman::` spellings.
 */

#pragma once

namespace calaman {

// ========================================================================
// Mathematical Constants
// ========================================================================

/// @brief The value 0 in element type @p T
template<typename T>
inline constexpr T kZero = T{0.0};

/// @brief The value 1 in element type @p T
template<typename T>
inline constexpr T kOne = T{1.0};

/// @brief The value 2 in element type @p T
template<typename T>
inline constexpr T kTwo = T{2.0};

/// @brief The value -1 in element type @p T
template<typename T>
inline constexpr T kNegativeOne = T{-1.0};

/// @brief Pi in element type @p T
///
/// Braced from a double literal: float-to-float narrowing is well-formed for a
/// constant in range, even when it is not exactly representable, so kPi<float>
/// is the correctly-rounded float rather than a compile error.
template<typename T>
inline constexpr T kPi = T{3.141592653589793238462643383};

} // namespace calaman
