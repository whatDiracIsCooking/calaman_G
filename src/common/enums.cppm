/**
 * @file enums.cppm
 * @brief The :enums partition of calaman.common -- the module face of enums.h
 *
 * Carries the LAPACK selector enums (Jobz, Uplo) to consumers that `import`
 * rather than `#include`. The definitions live in enums.h so device .cu code can
 * share them; a global module fragment cannot `import`, so the header arrives by
 * #include here and the names are then re-exported into the module purview.
 *
 * The re-export is the whole job of this unit: the enums are declared in the GMF
 * (global linkage, not the module purview), and a GMF declaration is never
 * implicitly exported, so `export using` republishes each under its original
 * `calaman::` spelling. Importers and #includers therefore see identical names.
 *
 * This is a partition, not a standalone module: outside code reaches these names
 * only through `import calaman.common;` (the primary interface re-exports this
 * partition). Add an enum in ONE place -- enums.h -- then add its `using` below,
 * or importers will not see it.
 */

module;

#include "enums.h"

export module calaman.common:enums;

// Republish the header's names, which sit in the global module fragment, into
// the purview so importers see them.
export namespace calaman {

using calaman::Jobz;
using calaman::Uplo;
using calaman::Region;
using calaman::Trans;
using calaman::Side;
using calaman::Diag;
using calaman::Range;
using calaman::JobSvd;
using calaman::SortDir;
using calaman::Norm;

} // namespace calaman
