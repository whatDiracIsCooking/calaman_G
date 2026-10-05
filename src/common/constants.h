/**
 * @file constants.h
 * @brief Typed mathematical constants (kZero, kOne, kTwo, kNegativeOne, kPi)
 *        over the four element types, shareable by device .cu and module GMFs
 *
 * A plain header, not a module unit, so it is #includable from BOTH a device .cu
 * and a module's global module fragment (which cannot `import`); the sibling
 * constants.cppm includes it and re-exports these names so importers of
 * calaman.common see the same calaman:: spellings. Everything is `inline
 * constexpr`, usable in host and device code with no separate definition.
 *
 * The real primary templates cover float and double; the complex types
 * (wwrFloatComplex, wwrDoubleComplex) are explicit specialisations built by
 * brace-initialisation `{re, im}` -- the only constexpr-capable spelling, since
 * WarpWraps' make_wwr*Complex builders are not constexpr. Runtime complex
 * construction still goes through those builders (common/elem_ops.cuh); the
 * constants are the one off-surface exception. See docs/architecture.md §6.
 *
 * Those specialisations are `inline constexpr`, so they must exist in every TU
 * that uses kZero<complex>: this header therefore includes complex.h for the
 * types, which is why :constants links wwr.device + wwr_backend as :fp_types
 * does. Consumers include it root-relative as "common/constants.h".
 */

#pragma once

#include <complex.h> // wwrFloatComplex / wwrDoubleComplex (types; host-safe)

namespace calaman {

// ========================================================================
// Mathematical Constants -- real primary templates (float, double)
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

// ========================================================================
// Complex specialisations (wwrFloatComplex, wwrDoubleComplex)
//
// Brace-initialised because make_wwr*Complex is not constexpr (file header; §6).
// One token-pasting macro stamps both precisions so they cannot drift, as
// elem_ops.cuh's complex ops do; #undef'd so it does not leak. CT is the complex
// type, R its real component type; each constant is the real value with a zero
// imaginary part.
// ========================================================================

#define CLM_DEFINE_COMPLEX_CONSTANTS(CT, R)                                                        \
  template<>                                                                                       \
  inline constexpr wwr::CT kZero<wwr::CT> = {R(0), R(0)};                                           \
  template<>                                                                                       \
  inline constexpr wwr::CT kOne<wwr::CT> = {R(1), R(0)};                                            \
  template<>                                                                                       \
  inline constexpr wwr::CT kTwo<wwr::CT> = {R(2), R(0)};                                            \
  template<>                                                                                       \
  inline constexpr wwr::CT kNegativeOne<wwr::CT> = {R(-1), R(0)};                                   \
  template<>                                                                                       \
  inline constexpr wwr::CT kPi<wwr::CT> = {R(3.141592653589793238462643383), R(0)};

CLM_DEFINE_COMPLEX_CONSTANTS(wwrFloatComplex, float)
CLM_DEFINE_COMPLEX_CONSTANTS(wwrDoubleComplex, double)

#undef CLM_DEFINE_COMPLEX_CONSTANTS

} // namespace calaman
