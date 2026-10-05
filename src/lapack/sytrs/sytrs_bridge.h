/**
 * @file sytrs_bridge.h
 * @brief Device-launcher declarations shared between calaman.sytrs and its
 *        device-compiled translation unit
 *
 * Included by sytrs.cppm in its GLOBAL MODULE FRAGMENT and by sytrs.cu directly,
 * the split pstrf/lacgv use: the declarations live in the GMF, not the module
 * purview, so a purview name's module linkage cannot stop them binding to the
 * definitions compiled in the plain .cu translation unit.
 *
 * sytrs()'s numerical bulk is wrapped BLAS (ger/geru/gemv/swap). What stays here
 * is the one piece no BLAS call expresses -- applying the block-diagonal D^-1 of
 * the Bunch-Kaufman factorization to the right-hand sides: the 1x1 reciprocal
 * scale of one B row, and the symmetric 2x2 solve across two B rows.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the module passes its
 * handle's stream straight through. Reading the backend define that header needs
 * is why the module links wwr_backend PRIVATE -- see this directory's
 * CMakeLists.txt, the calaman.pstrf arrangement.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief 1x1 pivot: scale one right-hand-side row by 1/D(k,k)
///
/// Divides @p b_row[j*ldb] by the device scalar *@p akk for every j in
/// [0, @p nrhs) -- the DSCAL(1/A(K,K)) of the reference ?sytrs 1x1 branch.
/// @p b_row points at B(k, 0); @p akk at the stored diagonal A(k,k).
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex, wwrDoubleComplex
template<typename T>
void sytrs_scale_row(wwr::wwrStream_t stream, T *b_row, int ldb, int nrhs, const T *akk);

/// @brief 2x2 pivot: apply the inverse of one symmetric 2x2 D block to two B rows
///
/// Solves D * [x_top; x_bot] = [b_top; b_bot] column by column, where D is the
/// symmetric 2x2 [[*@p top_diag, *@p off], [*@p off, *@p bot_diag]] -- the
/// AKM1K/AKM1/AK/DENOM dance of the reference ?sytrs 2x2 branch, read straight
/// off the stored factor. @p b_top / @p b_bot point at the two B rows; the three
/// scalar pointers at the stored block entries.
///
/// @tparam T Element type; instantiated for float, double, wwrFloatComplex, wwrDoubleComplex
template<typename T>
void sytrs_solve_2x2(wwr::wwrStream_t stream, T *b_top, T *b_bot, int ldb, int nrhs,
                     const T *top_diag, const T *off, const T *bot_diag);

} // namespace calaman::device
