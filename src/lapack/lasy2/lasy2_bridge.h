/**
 * @file lasy2_bridge.h
 * @brief Device-launcher declaration shared between calaman.lasy2's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by lasy2.cu
 * directly -- the same split lacgv_bridge.h uses: the declaration lives in the
 * GMF, not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports, so
 * the module passes its stream straight through. Reading the backend define that
 * header needs is why the module links wwr_backend PRIVATE (CMakeLists.txt).
 *
 * The parameter list is LAPACK's ?lasy2, 1:1 (docs/architecture.md §4): the
 * Fortran LOGICALs become bool, every scalar stays int, and the five outputs
 * (@p scale, @p x, @p xnorm, @p info) are device pointers the caller owns.
 */

#pragma once

#include "runtime.h"

namespace calaman::device {

/// @brief Enqueue the order-(n1,n2) Sylvester solve op(TL)*X + ISGN*X*op(TR) =
///        SCALE*B, LAPACK's ?lasy2, on @p stream
///
/// One single-thread kernel solves for the N1-by-N2 matrix X (1 <= N1,N2 <= 2),
/// writing @p scale, @p x, @p xnorm and @p info on the device and returning
/// without synchronizing. @p tl, @p tr, @p b are device inputs; the outputs are
/// device pointers. A zero @p n1 or @p n2 writes only *info = 0.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void lasy2(wwr::wwrStream_t stream, bool ltranl, bool ltranr, int isgn, int n1, int n2, const T *tl,
           int ldtl, const T *tr, int ldtr, const T *b, int ldb, T *scale, T *x, int ldx, T *xnorm,
           int *info);

} // namespace calaman::device
