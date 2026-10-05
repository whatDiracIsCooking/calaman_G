/**
 * @file fp_types.cppm
 * @brief The :fp_types partition of calaman.common -- the module face of
 *        fp_types.h
 *
 * Carries the floating-point concepts (real_fp, complex_fp, usual_fp) and the
 * real/complex type maps (ComplexToRealType, RealToComplexType) to consumers
 * that `import` rather than `#include`. The names are WarpWraps' own, brought
 * under `calaman::` in fp_types.h; a global module fragment cannot `import`, so
 * that header arrives by #include here and the `calaman::` names are then
 * re-exported into the module purview.
 *
 * The re-export is the whole job of this unit, exactly as :constants and :enums:
 * fp_types.h places the names in the GMF (global linkage, not the module
 * purview), and a GMF declaration is never implicitly exported, so `export
 * using` republishes each. Importers and #includers therefore see identical
 * `calaman::` spellings backed by the same `wwr::` entities.
 *
 * This is a partition, not a standalone module: outside code reaches these names
 * only through `import calaman.common;` (the primary interface re-exports this
 * partition). Add a trait in ONE place -- fp_types.h -- then add its `using`
 * below, or importers will not see it.
 */

module;

#include "fp_types.h"

export module calaman.common:fp_types;

// Republish the header's names, which sit in the global module fragment, into
// the purview so importers see them.
export namespace calaman {

using calaman::complex_fp;
using calaman::real_fp;
using calaman::usual_fp;

using calaman::ComplexToRealType;
using calaman::RealToComplexType;

} // namespace calaman
