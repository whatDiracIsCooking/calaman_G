/**
 * @file interface.cppm
 * @brief Primary interface for calaman.aberth -- every root of a batch of
 *        polynomials, by the Aberth-Ehrlich iteration
 *
 * One device routine over @p batch polynomials of one degree @p n, each solved
 * by one warp (aberth.cuh's aberth_warp). Enqueued on the given stream and
 * returns WITHOUT synchronizing; every array is a device pointer the caller
 * owns, so a stream is the whole requirement. Coefficients are real or complex;
 * roots are always complex in the same precision, and CT is deduced from @p roots.
 *
 * Not a LAPACK routine, so it is its own module. `extern template` below pairs
 * with instantiations.cpp, as in calaman.lartg.
 *
 * Usage:
 *   import calaman.aberth;    // also re-exports calaman::Status
 *   // d_coeffs: batch*(n+1) ascending coefficients; d_roots: batch*n; d_iters: batch
 *   calaman::aberth(stream, n, batch, d_coeffs, d_roots, d_iters, 1e-14, 100);
 */

module;

// CLM_TRY / CLM_REQUIRE arrive by #include: macros do not cross an import.
#include "error_handling/error_macros.h"

#include "aberth_bridge.h"

export module calaman.aberth;

import std;
import wwr.runtime_api;
import wwr.complex;
import calaman.common; // usual_fp, complex_fp, ComplexToRealType

// export import: aberth RETURNS calaman::Status, whose members a consumer needs.
export import calaman.error_handling;

namespace calaman {

/// @brief Every root of @p batch degree-@p n polynomials on @p stream
///
/// Polynomial b is coeffs[b*(n+1) + k] = a_k, ascending, a_n != 0; its roots land
/// in roots[b*n, b*n + n) and in iters[b] the iterations it took, or -1 if
/// @p max_iter passed first. A root settles when its relative correction is
/// <= @p tol or |p(z)| is at Horner's rounding floor.
///
/// @param n        Degree; 2n * sizeof(CT) must fit device::kAberthMaxSharedBytes
/// @param batch    Polynomials; n == 0 or batch == 0 enqueues nothing
/// @return Success; invalid-value for a negative or oversized argument or a null
///         pointer with work to do; else the runtime error the launch reported
export template<usual_fp T, complex_fp CT>
  requires std::same_as<ComplexToRealType<CT>, ComplexToRealType<T>>
Status aberth(const wwr::wwrStream_t stream, const int n, const int batch, const T *coeffs,
              CT *roots, int *iters, const ComplexToRealType<T> tol, const int max_iter) {
  CLM_REQUIRE(n >= 0 && batch >= 0 && max_iter >= 0 && tol >= 0, wwr::wwrErrorInvalidValue);
  if (n == 0 || batch == 0) {
    return wwr::wwrSuccess;
  }
  CLM_REQUIRE(2 * static_cast<std::size_t>(n) * sizeof(CT) <= device::kAberthMaxSharedBytes,
              wwr::wwrErrorInvalidValue);
  CLM_REQUIRE(coeffs != nullptr && roots != nullptr && iters != nullptr, wwr::wwrErrorInvalidValue);
  device::aberth(stream, n, batch, coeffs, roots, iters, tol, max_iter);
  // The launcher returns void: the runtime's sticky error is the only report.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status aberth<float, wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                                           const float *, wwr::wwrFloatComplex *,
                                                           int *, float, int);
extern template Status aberth<double, wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                                             const double *,
                                                             wwr::wwrDoubleComplex *, int *, double,
                                                             int);
extern template Status
aberth<wwr::wwrFloatComplex, wwr::wwrFloatComplex>(wwr::wwrStream_t, int, int,
                                                   const wwr::wwrFloatComplex *,
                                                   wwr::wwrFloatComplex *, int *, float, int);
extern template Status
aberth<wwr::wwrDoubleComplex, wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, int,
                                                     const wwr::wwrDoubleComplex *,
                                                     wwr::wwrDoubleComplex *, int *, double, int);

} // namespace calaman
