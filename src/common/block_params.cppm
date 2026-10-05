/**
 * @file block_params.cppm
 * @brief The :block_params partition of calaman.common -- the module face of
 *        block_params.h
 *
 * Carries kWarpSize and the num_warps concept to consumers that `import` rather
 * than `#include`. The definitions live in block_params.h so device .cu code can
 * share them; a global module fragment cannot `import`, so the header arrives by
 * #include here and `export using` republishes each name under its original
 * `calaman::` spelling, as align_up.cppm does.
 *
 * Add a name in ONE place -- block_params.h -- then add its `using` below, or
 * importers will not see it.
 */

module;

#include "block_params.h"

export module calaman.common:block_params;

// Republish the header's names, which sit in the global module fragment, into
// the purview so importers see them.
export namespace calaman {

using calaman::kWarpSize;
using calaman::num_warps;

} // namespace calaman
