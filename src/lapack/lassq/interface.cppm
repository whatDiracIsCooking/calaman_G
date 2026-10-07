/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lassq -- the scaled sum of squares of a
 *        vector, LAPACK's ?lassq
 *
 * Folds the sum of squares of a strided vector into a caller-held (scale,
 * sumsq) pair of device scalars, so that on return
 * scale^2 * sumsq == sum |x(i)|^2 + scale_in^2 * sumsq_in. Neither the entries
 * nor the running total is ever squared outside the exponent range: entries are
 * split into three bands and each band is scaled before squaring. That is the
 * whole reason ?lassq exists, and the reason an accumulating pair -- not a
 * single norm -- is the return: it is how a Frobenius norm is built column by
 * column. Enqueued on the stream and returns WITHOUT synchronizing.
 *
 * Mapping from DLASSQ (docs/architecture.md §4): the s/d/c/z variants become one
 * template over T, with both scalars in T's real component type; N and INCX stay
 * int (signed, as ?lassq's own INCX is), and SCALE / SUMSQ become device
 * pointers, in and out. A NaN in either on entry leaves both untouched, and
 * n <= 0 applies only ?lassq's canonicalization of a pair representing zero.
 *
 * `extern template` below pairs with instantiations.cpp, so an importer never
 * re-instantiates a body naming the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lassq;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t
 *   // d_scale, d_sumsq: device scalars, seeded (1, 0) for a fresh accumulation
 *   calaman::lassq<double>(stream, n, d_x, 1, d_scale, d_sumsq);
 */

module;

#include "lassq_bridge.h"

export module calaman.lassq;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // ComplexToRealType

// export import, not a plain import: lassq RETURNS calaman::Status, so a
// consumer must see its member functions, not just its name -- as for lanst.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Fold the sum of squares of the strided @p d_x into the pair (?lassq)
///
/// Updates (*d_scale, *d_sumsq) so their represented value gains
/// sum |x(i)|^2, and returns without synchronizing. @p incx may be negative
/// (the same elements, walked from the far end) and its magnitude is the stride;
/// @p n <= 0 only canonicalizes the pair, as ?lassq does.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream  Stream the work is enqueued on; all pointers live on its device
/// @param n       Number of elements of x
/// @param d_x     Device vector, stride @p incx; read, never written
/// @param incx    Stride between elements of x (non-zero)
/// @param d_scale Device scalar, in and out: ?lassq's SCALE
/// @param d_sumsq Device scalar, in and out: ?lassq's SUMSQ
/// @return Success, or the error the kernel launch reported
export template<typename T>
Status lassq(const wwr::wwrStream_t stream, const int n, const T *const d_x, const int incx,
             ComplexToRealType<T> *const d_scale, ComplexToRealType<T> *const d_sumsq) {
  using R = ComplexToRealType<T>;
  // n < 0 is n == 0: the reference returns early for both, but only AFTER the
  // normalization the kernel still has to apply, so the launch is not skipped.
  const std::size_t count = n <= 0 ? 0 : static_cast<std::size_t>(n);
  device::lassq<T, R>(stream, count, d_x, incx, d_scale, d_sumsq);
  return wwr::wwrGetLastError();
}

extern template Status lassq<float>(wwr::wwrStream_t, int, const float *, int, float *, float *);
extern template Status lassq<double>(wwr::wwrStream_t, int, const double *, int, double *, double *);
extern template Status lassq<wwr::wwrFloatComplex>(wwr::wwrStream_t, int,
                                                   const wwr::wwrFloatComplex *, int, float *,
                                                   float *);
extern template Status lassq<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int,
                                                    const wwr::wwrDoubleComplex *, int, double *,
                                                    double *);

} // namespace calaman
