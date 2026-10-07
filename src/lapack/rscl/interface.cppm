/**
 * @file interface.cppm
 * @brief Primary interface for calaman.rscl -- scale a vector by the reciprocal
 *        of a real scalar, LAPACK's ?rscl
 *
 * Multiply the length-n strided vector x in place by 1/a. The reciprocal is NOT
 * formed directly -- 1/a can overflow for a tiny a, and a*x can underflow where
 * x/a would not. Instead the guarded multiplier loop (verbatim from netlib
 * ?rscl) decomposes 1/a on the HOST into a chain of factors each in range, and
 * this routine enqueues the per-element multiply once per factor, as the
 * reference calls ?scal once per factor. Correct whenever x/a itself is
 * representable, which is ?rscl's own contract.
 *
 * Enqueued on the given stream and returns WITHOUT synchronizing, like a BLAS
 * call; the caller synchronizes when it needs x. x is a device pointer the
 * caller owns; nothing is allocated here -- so a stream, not a device handle,
 * matching calaman.lascl, whose multiplier loop this one mirrors.
 *
 * Scope: the REAL-scalar family -- srscl, drscl, csrscl and zdrscl, one
 * template over T with the scalar in T's real component type. The complex-
 * scalar crscl / zrscl, whose safe complex reciprocal is a different algorithm,
 * are not implemented. incx <= 0 leaves x untouched (the reference's ?scal
 * returns early for it); a == 0 or a non-finite a is rejected as InvalidValue --
 * a deliberate divergence, since the reference loops forever on an infinite a.
 *
 * Usage:
 *   import calaman.rscl;      // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t
 *   calaman::rscl<double>(stream, n, a, d_x, 1);   // d_x <-- d_x / a
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "rscl_bridge.h"

export module calaman.rscl;

import std;
import wwr.runtime_api; // wwrStream_t, wwrGetLastError
import wwr.complex;     // wwrFloatComplex, wwrDoubleComplex
import calaman.common;  // ComplexToRealType

// export import, not a plain import: rscl RETURNS calaman::Status, so a
// consumer must see its member functions, not just its name -- as for lascl.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Scale the length-@p n strided vector @p d_x by 1/@p a on @p stream
///
/// Enqueues x <-- x / a as a chain of overflow-safe multiplies and returns
/// without synchronizing. Enqueues nothing when @p n < 1 or @p incx <= 0, as
/// the reference's ?scal does. x must live on @p stream's device.
///
/// @tparam T Element type; float, double, wwrFloatComplex or wwrDoubleComplex
/// @param stream Stream the scaling is enqueued on; x lives on its device
/// @param n Number of elements of x
/// @param a Real divisor; must be non-zero and finite
/// @param d_x Device vector, scaled in place, stride @p incx
/// @param incx Stride between elements of x; <= 0 is a no-op
/// @return Success, InvalidValue for a bad @p a, or a launch error
export template<typename T>
Status rscl(const wwr::wwrStream_t stream, const int n, const ComplexToRealType<T> a,
            T *const d_x, const int incx) {
  using R = ComplexToRealType<T>;

  // Not ?rscl's: the reference reports no INFO and, on an infinite a, spins
  // forever multiplying by smlnum. a == 0 is outside its stated contract too
  // (x/0 does not exist), so both become the one error this return type can
  // speak -- lascl's InvalidValue for a bad cfrom.
  if (a == R{0} || !std::isfinite(a)) {
    return wwr::wwrErrorInvalidValue;
  }
  // The reference's ?scal returns early for incx <= 0, leaving x untouched,
  // which makes the whole call a no-op however the multiplier chain runs.
  if (n < 1 || incx <= 0) {
    return wwr::wwrSuccess;
  }

  // The guarded multiplier loop, verbatim from netlib ?rscl: carry the fraction
  // as a separate numerator and denominator and shift them by smlnum / bignum
  // until their quotient is in range, so no factor applied to x over- or
  // underflows. smlnum is DLAMCH 'S' (the smallest normal, == numeric_limits
  // min on the host, where this scalar arithmetic runs); bignum its reciprocal.
  const R smlnum = std::numeric_limits<R>::min();
  const R bignum = R{1} / smlnum;

  R cden = a;
  R cnum = R{1};
  bool done = false;
  do {
    const R cden1 = cden * smlnum;
    const R cnum1 = cnum / bignum;
    R mul;
    if (std::abs(cden1) > std::abs(cnum) && cnum != R{0}) {
      // The denominator is large compared to the numerator: shrink x first.
      mul = smlnum;
      cden = cden1;
    } else if (std::abs(cnum1) > std::abs(cden)) {
      // The denominator is small compared to the numerator: grow x first.
      mul = bignum;
      cnum = cnum1;
    } else {
      mul = cnum / cden;
      done = true;
    }

    device::rscl<T, R>(stream, static_cast<std::size_t>(n), mul, d_x, incx);
    // The launcher returns void, so the only way to catch a bad launch is the
    // runtime's sticky error -- checked the moment it is enqueued, as lascl does.
    CLM_TRY(wwr::wwrGetLastError());
  } while (!done);

  return wwr::wwrSuccess;
}

extern template Status rscl<float>(wwr::wwrStream_t, int, float, float *, int);
extern template Status rscl<double>(wwr::wwrStream_t, int, double, double *, int);
extern template Status rscl<wwr::wwrFloatComplex>(wwr::wwrStream_t, int, float,
                                                  wwr::wwrFloatComplex *, int);
extern template Status rscl<wwr::wwrDoubleComplex>(wwr::wwrStream_t, int, double,
                                                   wwr::wwrDoubleComplex *, int);

} // namespace calaman
