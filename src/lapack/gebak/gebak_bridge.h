/**
 * @file gebak_bridge.h
 * @brief Device-launcher declarations shared between calaman.gebak's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by gebak.cu
 * directly -- the same split gebal_bridge.h / lacpy_bridge.h use: the
 * declarations live in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop them binding to the definitions compiled in the
 * plain .cu translation unit.
 *
 * gebak is the back-transformation companion of calaman.gebal: it undoes, on a
 * matrix of eigenvectors, the similarity gebal applied. Both of its stages are
 * hand-written kernels. Unlike gebal, its host driver does NOT branch on device
 * data -- the permutation sequence is data-dependent but is replayed inside a
 * single cooperating block on the device, so the driver issues two launches and
 * reads nothing back. That is why gebak stays fully asynchronous where gebal must
 * synchronize.
 *
 * COMPLEX TYPES DO NOT APPEAR HERE, for the reason gebal_bridge.h spells out: this
 * header is parsed in a host GMF (no wwr.complex import) and in a device .cu (no
 * host complex builder), with no common complex type. So every launcher is generic
 * in the element type @c T and, where it touches the real scale factors, in a
 * second parameter @c R, which the caller spells calaman::ComplexToRealType<T>. The
 * .cu names the concrete wwrFloatComplex / wwrDoubleComplex only in its explicit
 * instantiations, in device context.
 *
 * wwrStream_t arrives from runtime.h, an include-only header, since a GMF cannot
 * import; it is the SAME type wwr.runtime_api exports, so the host driver passes
 * its stream straight through. Reading the backend define that header needs is
 * why the module links wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Block size for the single-block permutation kernel.
inline constexpr int kGebakPermuteBlock = 256;

/// @brief Scale rows @p ilo0 .. @p ihi0 (0-based, inclusive) of the n-by-m
///        column-major matrix @p V by the real factor @p scale[i] -- or its
///        reciprocal when @p invert is true, which is the left-eigenvector case.
///
/// LAPACK's backward scaling: right eigenvectors multiply row i by scale(i), left
/// eigenvectors by 1/scale(i). Fully parallel across the window and columns -- the
/// scaled rows sit inside [ilo, ihi] and the permutation touches only rows outside
/// it, so there is no overlap to order. @p scale is indexed by the absolute row i,
/// so entries outside the window are simply never read.
///
/// @tparam T Element type (float, double, wwrFloatComplex, wwrDoubleComplex).
/// @tparam R Real scale type; call with calaman::ComplexToRealType<T>.
template<typename T, typename R>
void gebak_scale_rows(wwr::wwrStream_t stream, T *V, int ldv, int m, int ilo0, int ihi0,
                      const R *scale, bool invert);

/// @brief Apply ?gebak's backward row permutation to the n-by-m column-major
///        matrix @p V, in a single cooperating block.
///
/// Replays LAPACK's exact interchange order: for ii = 1..n it derives the row i
/// (indices below @p ilo are walked in reverse, ii-driven; indices above @p ihi
/// forward), reads the 1-based target k = int(scale(i)), and swaps rows i and k
/// when they differ -- rows inside [ilo, ihi] are skipped. The order matters
/// because consecutive swaps can touch overlapping rows, so the whole sequence
/// runs in ONE block with a __syncthreads() between swaps; the block cooperates on
/// the O(m) column work of each swap. Keeping it on the device is what lets the
/// host driver stay asynchronous -- it never reads the scale indices back.
///
/// @tparam T Element type.
/// @tparam R Real scale type; call with calaman::ComplexToRealType<T>. @p scale is read
///         only outside [ilo, ihi], where ?gebal stored the interchange indices.
template<typename T, typename R>
void gebak_permute(wwr::wwrStream_t stream, T *V, int ldv, int m, int n, int ilo, int ihi,
                   const R *scale);

} // namespace calaman::device
