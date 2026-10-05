/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lacgv -- conjugate a complex vector
 *
 * One device routine: overwrite the length-n strided complex vector x with its
 * element-wise conjugate, in place. Enqueued on the given stream and returns
 * WITHOUT synchronizing, like a BLAS call; the caller synchronizes when it needs
 * x. x is a device pointer the caller owns; nothing is allocated here.
 *
 * A stream, not a device handle, is the whole requirement: this routine enqueues
 * one kernel and allocates nothing, so it needs no device index and no memory
 * pool -- matching calaman.lacpy. See test/shared/README.md.
 *
 * COMPLEX ONLY. ?lacgv exists only as clacgv / zlacgv: the conjugate of a real
 * vector is the vector, so there is no s/d variant and the surface is the two
 * complex types, constrained by calaman::complex_fp -- the same scope complex_cast
 * has. The result is sign-of-incx-independent (conjugation is per-element), so a
 * negative incx conjugates the same elements, matching LAPACK's ioff stepping.
 *
 * Mapping from LAPACK's CLACGV (docs/architecture.md §4 -- keep the name, drop
 * the Fortran calling convention): the c/z variants become one template over
 * ComplexT, and N / INCX stay int (as LAPACKE_?lacgv takes them, INCX signed).
 *
 * `extern template` below pairs with instantiations.cpp: the wrapper is
 * instantiated once inside this library, so an importer never re-instantiates a
 * body that names the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lacgv;      // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   import wwr.complex;       // wwrDoubleComplex
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_x: device vector of n complex elements, stride incx
 *   calaman::lacgv<wwr::wwrDoubleComplex>(stream, n, d_x, 1);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

#include "lacgv_bridge.h"

export module calaman.lacgv;

import std;
import wwr.runtime_api;      // wwrStream_t, wwrGetLastError
import wwr.complex;          // wwrFloatComplex, wwrDoubleComplex (extern template list)
import calaman.common;  // complex_fp

// export import, not a plain import: lacgv RETURNS calaman::Status, so a consumer
// of `import calaman.lacgv;` must see Status's member functions, not just its
// name -- the same re-export diff_norm and lacpy do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Conjugate the length-@p n strided complex vector @p x on @p stream
///
/// Overwrites x[k*incx] with conj(x[k*incx]) for k in [0, n) and returns without
/// synchronizing. Enqueues nothing when @p n is 0 or negative. @p incx must be
/// non-zero; its sign does not change which elements are conjugated. x must live
/// on @p stream's device.
///
/// @tparam ComplexT Complex element type (wwrFloatComplex or wwrDoubleComplex)
/// @param stream Stream the launch is enqueued on; x lives on its device
/// @param n Number of elements of x
/// @param x Device vector, conjugated in place, stride @p incx
/// @param incx Stride between elements of x (non-zero)
/// @return Success, or the runtime error the kernel launch reported
export template<calaman::complex_fp ComplexT>
Status lacgv(const wwr::wwrStream_t stream, const int n, ComplexT *const x, const int incx) {
  if (n <= 0) {
    return wwr::wwrSuccess;
  }
  device::lacgv(stream, n, x, incx);
  // The launcher returns void, so the only way to catch a bad launch is the
  // runtime's sticky error -- checked the moment it is enqueued, as lacpy does.
  CLM_TRY(wwr::wwrGetLastError());
  return wwr::wwrSuccess;
}

extern template Status lacgv<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, wwr::wwrFloatComplex *,
                                                   int);
extern template Status lacgv<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, wwr::wwrDoubleComplex *,
                                                    int);

} // namespace calaman
