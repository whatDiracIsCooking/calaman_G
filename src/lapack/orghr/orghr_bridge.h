/**
 * @file orghr_bridge.h
 * @brief Device-launcher declaration shared between the module's interface unit
 *        and its device-compiled translation unit
 *
 * Included by interface.cppm in its GLOBAL MODULE FRAGMENT, and by orghr.cu
 * directly. The declaration must live in the GMF, not the module purview: a
 * purview name gets module linkage and can never bind to a definition compiled
 * in a plain TU, which is what the .cu is.
 *
 * wwrStream_t arrives from runtime.h, an include-only header rather than an
 * `import`, since a GMF cannot import. It is the SAME type wwr.runtime_api
 * exports, so the wrapper passes its stream straight through. Reading the
 * backend define selected_backend.h needs is why the module links wwr_backend
 * PRIVATE -- see this directory's CMakeLists.txt.
 *
 * orghr_prep is the ?orghr-specific reflector shuffle (DORGHR's loops before the
 * ?orgqr call): it reads an untouched snapshot @p a_in and writes the shifted,
 * identity-padded matrix @p a_out, so the right-to-left column move is a pure
 * gather with no in-place aliasing. ?orgqr then finishes Q on the window block.
 */

#pragma once

#include <runtime.h>

#include <cstddef>

namespace calaman::device {

/// @brief Enqueue the ?orghr reflector shuffle: shift the gehrd vectors one
///        column right over the window and set the rest to the identity
///
/// Writes the n-by-n column-major @p a_out (leading dimension @p lda) by gathering
/// from the snapshot @p a_in (same layout): column j in (ilo, ihi] takes column
/// j-1's subdiagonal tail, its diagonal becomes 1 and everything else 0; columns
/// outside the window become identity columns. @p ilo / @p ihi are 1-based, as in
/// LAPACK. Enqueued on @p stream, returns without synchronizing; a_in and a_out
/// must not alias. Launches nothing when n is 0.
///
/// @tparam T Element type; instantiated for float, double
template<typename T>
void orghr_prep(wwr::wwrStream_t stream, std::size_t n, int ilo, int ihi, const T *a_in,
                T *a_out, std::size_t lda);

} // namespace calaman::device
