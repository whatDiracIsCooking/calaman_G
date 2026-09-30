/**
 * @file constants.cppm
 * @brief The :constants partition of calaman.common -- the module face of
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
 * This is a partition, not a standalone module: outside code reaches these names
 * only through `import calaman.common;` (the primary interface re-exports this
 * partition). Add a constant in ONE place -- constants.h -- then add its `using`
 * below, or importers will not see it.
 */

module;

#include "constants.h"

export module calaman.common:constants;

// Republish the header's names, which sit in the global module fragment, into
// the purview so importers see them.
export namespace calaman {

using calaman::kZero;
using calaman::kOne;
using calaman::kTwo;
using calaman::kNegativeOne;
using calaman::kPi;

} // namespace calaman
