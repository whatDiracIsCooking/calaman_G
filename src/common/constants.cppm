/**
 * @file constants.cppm
 * @brief Primary interface for calaman.common.constants -- the module face of
 *        constants.h
 *
 * Carries the typed constants (kZero, kOne, kTwo, kNegativeOne, kPi) to
 * consumers that `import` rather than `#include`. The values themselves live in
 * constants.h so that device .cu code can share them; a global module fragment
 * cannot `import`, so the header arrives by #include here and the names are then
 * re-exported into the module purview.
 *
 * The re-export is the whole job of this unit: the constants are declared in the
 * GMF (global linkage, not the module purview), and a GMF declaration is never
 * implicitly exported, so `export using` republishes each under its original
 * `calaman::` spelling. Importers and #includers therefore see identical names.
 *
 * Add a constant in ONE place -- constants.h -- then add its `using` below, or
 * `import` consumers will not see it.
 *
 * Usage:
 *   import calaman.common.constants;
 *   const double two_pi = calaman::kTwo<double> * calaman::kPi<double>;
 */

module;

#include "constants.h"

export module calaman.common.constants;

// Republish the header's names, which sit in the global module fragment, into
// the purview so importers see them.
export namespace calaman {

using calaman::kZero;
using calaman::kOne;
using calaman::kTwo;
using calaman::kNegativeOne;
using calaman::kPi;

} // namespace calaman
