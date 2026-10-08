/**
 * @file interface.cppm
 * @brief Primary interface for calaman.larscl2 -- reciprocal diagonal
 *        row-scaling of a matrix, x <-- D^-1 x
 *
 * Divides the m-by-n column-major X in place by a diagonal D held as a length-m
 * real vector: X(i,j) <-- X(i,j) / d(i). The inverse twin of calaman.lascl2.
 * Enqueued on a stream; returns WITHOUT synchronizing. Takes a stream, not a
 * handle, and allocates nothing (test/shared/README.md).
 *
 * Mapping from ?LARSCL2 (docs/architecture.md §4): s/d/c/z become one template
 * over T, with D of type ComplexToRealType<T> -- real even for complex X, as in
 * the reference; the INTEGER extents become std::size_t; LDX is kept. No INFO,
 * as the reference has none. Each element is a true divide, never a reciprocal
 * multiply (larscl2.cu).
 *
 * `extern template` pairs with instantiations.cpp, so an importer never
 * re-instantiates a body that names the GMF-declared .cu launcher.
 *
 * Usage:
 *   import calaman.larscl2;   // also re-exports calaman::Status
 *   // d_d: length-m real device vector; d_x: column-major device matrix, ld ldx
 *   calaman::larscl2<double>(stream, m, n, d_d, d_x, ldx);
 */

module;

// CLM_TRY / CLM_REQUIRE -- macros, so they arrive by #include in the global
// module fragment; they need calaman::Status visible (the export import below).
#include "error_handling/error_macros.h"

#include "larscl2_bridge.h"

export module calaman.larscl2;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // ComplexToRealType

export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Divide the m-by-n column-major X by a real diagonal D on @p stream
///
/// Enqueues X(i,j) <-- X(i,j) / d(i) and returns without synchronizing.
/// Enqueues nothing when @p m or @p n is 0; rows [m, ldx) are never touched.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the scaling is enqueued on; D and X live on its device
/// @param m Number of rows of X and length of D
/// @param n Number of columns of X
/// @param d_d Device diagonal vector, length m, real for every T
/// @param d_x Device matrix to scale in place, column-major, leading dim @p ldx
/// @param ldx Leading dimension of X; ldx >= m
/// @return Success, wwrErrorInvalidValue for ldx < m or a null pointer, or the
///         runtime error the kernel launch reported
export template<typename T>
Status larscl2(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n,
               const ComplexToRealType<T> *const d_d, T *const d_x, const std::size_t ldx) {
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(ldx >= m && d_d != nullptr && d_x != nullptr, wwr::wwrErrorInvalidValue);
  device::larscl2<T, ComplexToRealType<T>>(stream, m, n, d_d, d_x, ldx);
  // The launcher returns void; the sticky launch error is the only report.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status larscl2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                      float *, std::size_t);
extern template Status larscl2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                       double *, std::size_t);
extern template Status larscl2<wwr::wwrFloatComplex>(wwr::wwrStream_t, std::size_t, std::size_t,
                                                     const float *, wwr::wwrFloatComplex *,
                                                     std::size_t);
extern template Status larscl2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, std::size_t, std::size_t,
                                                      const double *, wwr::wwrDoubleComplex *,
                                                      std::size_t);

} // namespace calaman
