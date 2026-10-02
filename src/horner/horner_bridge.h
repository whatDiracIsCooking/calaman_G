/**
 * @file horner_bridge.h
 * @brief Device-launcher declarations shared between calaman.horner's interface
 *        unit and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by horner.cu
 * directly -- the same split lacpy_bridge.h / laqps_bridge.h use: the
 * declarations live in the GMF, not the module purview, so a purview name's
 * module linkage cannot stop them binding to the definitions compiled in the
 * plain .cu translation unit.
 *
 * Horner's recurrence for a matrix polynomial is a gemm (cuBLAS/hipBLAS, via
 * wwr::gemm) plus "add a scalar to the diagonal". The gemm is BLAS; the two
 * diagonal updates are the only genuinely device-elementwise work, so they are
 * these launchers, run through wwr.extension.parallel_for from horner.cu.
 *
 * Both take the coefficient as a *device* pointer, which is what lets the whole
 * evaluation stay on the handle's stream without a host round-trip: a
 * coefficient produced by an earlier kernel is fed straight in. This is
 * independent of the BLAS handle's pointer mode -- the kernels read device
 * memory directly, while the gemm scalars are the host constants kOne/kZero.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather
 * than an `import`, since a GMF cannot import; it is the SAME type
 * wwr.runtime_api exports, so the module passes its handle's stream straight
 * through. Reading the backend define that header needs is why the module links
 * wwr_backend PRIVATE -- see this directory's CMakeLists.txt.
 *
 * Templated over float and double only. Complex ?horner is a deliberate later
 * extension, for the reasons calaman.common's constants.h documents: there is
 * no portable constexpr spelling of the kOne/kZero gemm scalars in complex.
 */

#pragma once

#include "runtime.h"

#include <cstddef>

namespace calaman::device {

/// @brief Write alpha * I over the n-by-n column-major block at @p d_P on @p stream
///
/// Seeds the Horner recurrence with its leading coefficient. Writes exactly the
/// n-by-n block: element (i, j) becomes *d_alpha when i == j and zero otherwise.
/// The padding rows between n and @p ldp are left untouched, so P may be a
/// submatrix view of a larger allocation. Enqueued on @p stream; returns without
/// synchronizing. Launches nothing when @p n is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream  Stream the launch is enqueued on
/// @param d_P     Device output, n-by-n column-major with leading dimension @p ldp
/// @param n       Matrix order
/// @param ldp     Leading dimension of @p d_P; ldp >= n
/// @param d_alpha Device pointer to the scalar
template<typename T>
void horner_set_scaled_identity(wwr::wwrStream_t stream, T *d_P, int n, int ldp,
                                const T *d_alpha);

/// @brief Add alpha to every diagonal entry of the n-by-n block at @p d_P on @p stream
///
/// The "+ c_k I" half of one Horner step. Off-diagonal entries are left alone,
/// so this composes with the gemm that produced them. Enqueued on @p stream;
/// returns without synchronizing. Launches nothing when @p n is 0.
///
/// @tparam T Element type; instantiated for float, double
/// @param stream  Stream the launch is enqueued on
/// @param d_P     Device in/out, n-by-n column-major with leading dimension @p ldp
/// @param n       Matrix order
/// @param ldp     Leading dimension of @p d_P; ldp >= n
/// @param d_alpha Device pointer to the scalar
template<typename T>
void horner_add_scaled_identity(wwr::wwrStream_t stream, T *d_P, int n, int ldp,
                                const T *d_alpha);

} // namespace calaman::device
