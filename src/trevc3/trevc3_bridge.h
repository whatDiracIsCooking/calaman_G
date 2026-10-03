/**
 * @file trevc3_bridge.h
 * @brief Device-launcher declaration shared between calaman.trevc3's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by trevc3.cu
 * directly -- the split laqr5_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports.
 *
 * The parameter list is LAPACK's ?trevc3 restricted to HOWMNY = 'A' with the
 * eigenvectors written straight into @p vl / @p vr (no OVER back-transform): the
 * caller-owned quasi-triangular @p t (device, read only), the eigenvector
 * outputs @p vl (SIDE in {'L','B'}) and @p vr (SIDE in {'R','B'}), and the
 * length-3N device scratch @p work the back-substitution uses. One thread runs
 * the whole computation; SELECT is unused and M is always N.
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue the eigenvectors of a real quasi-triangular (Schur) matrix,
///        LAPACK's ?trevc3 for HOWMNY = 'A', on @p stream
///
/// One single-thread kernel back-substitutes the eigenvectors of the upper
/// quasi-triangular @p t into @p vr (when @p want_right) and/or @p vl (when @p
/// want_left), each normalized to unit infinity norm. @p t is read-only device
/// memory; @p vl / @p vr are N-by-N device outputs; @p work is length-3N device
/// scratch. Matrices are column-major with the given leading dimensions.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void trevc3(wwr::wwrStream_t stream, bool want_left, bool want_right, int n, const T *t, int ldt,
            T *vl, int ldvl, T *vr, int ldvr, T *work);

} // namespace calaman::device
