/**
 * @file fp_types.h
 * @brief The floating-point concepts and real/complex type maps, re-published
 *        under `calaman::` from WarpWraps, as a GMF-shared header
 *
 * A thin re-export layer: the concepts (real_fp, complex_fp, usual_fp) and the
 * type maps (ComplexToRealType, RealToComplexType) that src/ constrains its
 * element templates with are WarpWraps' own, defined in its
 * <wrappers/common/fp_types.h>. This header pulls that in and republishes each
 * under its `calaman::` spelling so this project writes `calaman::usual_fp` and
 * `calaman::ComplexToRealType<T>` rather than reaching into `wwr::` for a type
 * trait -- the vendor names stay for the calls that genuinely are WarpWraps'
 * (BLAS/solver functions, handles, enum constants), not for the compile-time
 * type vocabulary.
 *
 * Re-export, not redefinition: a using-declaration names the entity, so
 * `calaman::usual_fp` and `wwr::usual_fp` are the SAME concept, and a template
 * constrained with one is satisfied by the other. This stays backend-neutral by
 * construction -- it names no cu-/hip- type, only the `wwr*` ones the concepts
 * already abstract (CLAUDE.md).
 *
 * Like constants.h / enums.h this is a plain header, #includable from BOTH a
 * device .cu (a plain TU) and a module's global module fragment (which cannot
 * `import`); the WarpWraps header it wraps carries no __device__ bodies, so the
 * one definition serves both. The sibling fp_types.cppm includes it and
 * re-exports these names into the `calaman.common:fp_types` purview.
 *
 * Consumers include it root-relative as "common/fp_types.h"; the WarpWraps
 * header is reached through the include root wwr.device exports (the same root
 * elem_ops.cuh uses for <wrappers/math/math.cuh>).
 */

#pragma once

#include <wrappers/common/fp_types.h>

namespace calaman {

// ========================================================================
// Floating-Point Type Concepts -- re-exported from wwr
// ========================================================================

using wwr::complex_fp;
using wwr::real_fp;
using wwr::usual_fp;

// ========================================================================
// Real <-> Complex Type Mappings -- re-exported from wwr
// ========================================================================

using wwr::ComplexToRealType;
using wwr::RealToComplexType;

} // namespace calaman
