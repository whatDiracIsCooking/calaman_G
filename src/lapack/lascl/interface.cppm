/**
 * @file interface.cppm
 * @brief Primary interface for calaman.lascl -- multiply a matrix by cto/cfrom
 *        without over/underflow, LAPACK's ?lascl for a full matrix (TYPE='G')
 *
 * A single device routine: scale the m-by-n column-major matrix A in place by
 * the ratio cto/cfrom, A(i,j) <-- (cto/cfrom) * A(i,j). The ratio is NOT formed
 * directly -- that could over/underflow. Instead the guarded multiplier loop
 * (verbatim from netlib ?lascl) decomposes it on the HOST into a chain of safe
 * factors, and this routine enqueues the per-element multiply once per factor.
 *
 * Enqueued on the given stream and returns WITHOUT synchronizing, like a BLAS
 * call; the caller synchronizes when it needs A. A is a device pointer the
 * caller owns; nothing is allocated here -- so a stream, not a device handle, is
 * the whole requirement, matching calaman.lascl2 and calaman.laset.
 *
 * Scope: TYPE='G' (full matrix); the banded and triangular TYPE codes ?lascl
 * also accepts are not implemented. The s/d variants become one template over T;
 * the INTEGER extents become std::size_t; LDA is kept (column-major storage).
 * cfrom == 0 or a NaN cfrom/cto is rejected -- ?lascl's INFO=-4/-5 -- as an
 * InvalidValue runtime Status, the one error this can report.
 *
 * `extern template` below pairs with instantiations.cpp, so an importer never
 * re-instantiates a body naming the .cu-side launcher declared only in the GMF.
 *
 * Usage:
 *   import calaman.lascl;     // also re-exports calaman::Status
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_a: column-major device matrix, leading dim lda
 *   calaman::lascl(stream, cfrom, cto, m, n, d_a, lda);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the export import
// below supplies.
#include "error_handling/error_macros.h"

#include "lascl_bridge.h"

export module calaman.lascl;

import std;
import wwr.runtime_api;

// export import, not a plain import: lascl RETURNS calaman::Status, so a
// consumer of `import calaman.lascl;` must see Status's member functions, not
// just its name -- the same re-export laset and lascl2 do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Not an `export namespace` block: an explicit instantiation declaration
// (`extern template`) cannot be exported, so the template carries its own
// `export` and the declarations below sit in the plain namespace.

/// @brief Scale the m-by-n column-major matrix A by cto/cfrom on @p stream
///
/// Enqueues A(i,j) <-- (cto/cfrom) * A(i,j) as a sequence of overflow-safe
/// multiplies and returns without synchronizing. Enqueues nothing when @p m or
/// @p n is 0. A must live on @p stream's device.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the scaling is enqueued on; A lives on its device
/// @param cfrom Denominator of the scale ratio; must be non-zero and finite
/// @param cto Numerator of the scale ratio; must be finite
/// @param m Number of rows of A
/// @param n Number of columns of A
/// @param d_a Device matrix to scale in place, column-major, leading dim @p lda
/// @param lda Leading dimension of A; lda >= m
/// @return Success, InvalidValue for a bad cfrom/cto, or a launch error
export template<typename T>
Status lascl(const wwr::wwrStream_t stream, const T cfrom, const T cto, const std::size_t m,
             const std::size_t n, T *const d_a, const std::size_t lda) {
  // ?lascl's INFO=-4/-5: cfrom == 0 cannot form a ratio, and a NaN cfrom/cto
  // poisons every factor. Reported as a runtime InvalidValue -- the one domain
  // this routine's return type can speak and the closest match to a bad arg.
  if (cfrom == T{0} || std::isnan(cfrom) || std::isnan(cto)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (m == 0 || n == 0) {
    return wwr::wwrSuccess;
  }

  // The guarded multiplier loop, verbatim from netlib ?lascl: reduce cto/cfrom
  // to a chain of factors each guaranteed in range, so no product over- or
  // underflows. smlnum is DLAMCH 'S' (the smallest normal, == numeric_limits
  // min on the host, where this scalar arithmetic runs); bignum its reciprocal.
  const T smlnum = std::numeric_limits<T>::min();
  const T bignum = T{1} / smlnum;

  T cfromc = cfrom;
  T ctoc = cto;
  bool done = false;
  do {
    const T cfrom1 = cfromc * smlnum;
    T mul;
    if (cfrom1 == cfromc) {
      // cfromc is an inf or a NaN -- the one-step ratio is the answer (and
      // cannot be reached for finite cfrom, which the guard above required).
      mul = ctoc / cfromc;
      done = true;
    } else {
      const T cto1 = ctoc / bignum;
      if (cto1 == ctoc) {
        // ctoc is an inf or a NaN: multiply by it and reset cfromc to 1.
        mul = ctoc;
        done = true;
        cfromc = T{1};
      } else if (std::abs(cfrom1) > std::abs(ctoc) && ctoc != T{0}) {
        mul = smlnum;
        done = false;
        cfromc = cfrom1;
      } else if (std::abs(cto1) > std::abs(cfromc)) {
        mul = bignum;
        done = false;
        ctoc = cto1;
      } else {
        mul = ctoc / cfromc;
        done = true;
      }
    }

    device::lascl(stream, m, n, mul, d_a, lda);
    // The launcher returns void, so the only way to catch a bad launch is the
    // runtime's sticky error -- checked the moment it is enqueued, as laset does.
    CLM_TRY(wwr::wwrGetLastError());
  } while (!done);

  return wwr::wwrSuccess;
}

extern template Status lascl<float>(wwr::wwrStream_t, float, float, std::size_t, std::size_t,
                                    float *, std::size_t);
extern template Status lascl<double>(wwr::wwrStream_t, double, double, std::size_t, std::size_t,
                                     double *, std::size_t);

} // namespace calaman
