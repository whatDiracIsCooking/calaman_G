/**
 * @file laexc_bridge.h
 * @brief Device-launcher declaration shared between calaman.laexc's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by laexc.cu
 * directly -- the split lasy2_bridge.h uses: the declaration lives in the GMF,
 * not the module purview, so a purview name's module linkage cannot stop it
 * binding to the definition compiled in the plain .cu translation unit.
 *
 * NO ELEMENT TYPE APPEARS HERE beyond the template parameter: the launcher is
 * generic in @c T, and the .cu names the concrete float / double only in its
 * explicit instantiations. wwrStream_t arrives from runtime.h, an include-only
 * header (a GMF cannot import); it is the SAME type wwr.runtime_api exports.
 * Reading the backend define that header needs is why the module links
 * wwr_backend PRIVATE (CMakeLists.txt).
 *
 * The parameter list is LAPACK's ?laexc, 1:1: WANTQ is a bool, every dimension
 * (@p n, @p ldt, @p ldq, @p j1 -- 1-based, @p n1, @p n2) is int, and @p t / @p q
 * are caller-owned device matrices. @p info is a device int the kernel writes (0
 * on a completed swap, 1 when the swap was rejected). No WORK array: the kernel
 * runs the whole algorithm in a single thread with its own on-stack scratch.
 */

#pragma once

#include <runtime.h>

namespace calaman::device {

/// @brief Enqueue the adjacent-block swap of a real Schur form, LAPACK's ?laexc,
///        on @p stream
///
/// One single-thread kernel swaps the @p n1-by-@p n1 diagonal block at @p j1
/// (1-based) with the adjacent @p n2-by-@p n2 block, accumulating the transform
/// into @p q when @p wantq, and writes @p info on the device (0 on success, 1 if
/// the swap was rejected). @p t and @p q are caller-owned device matrices.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void laexc(wwr::wwrStream_t stream, bool wantq, int n, T *t, int ldt, T *q, int ldq, int j1, int n1,
           int n2, int *info);

} // namespace calaman::device
