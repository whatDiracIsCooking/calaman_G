/**
 * @file laqr1.cppm
 * @brief The calaman.laqr1 module -- the first column of the shift polynomial
 *        (H - s1*I)(H - s2*I), LAPACK's ?laqr1
 *
 * Given an N-by-N upper-Hessenberg H (N is 2 or 3) and two shifts s1 = SR1+i*SI1
 * and s2 = SR2+i*SI2, sets the length-N vector V to the first column of
 * M = (H - s1*I)(H - s2*I), scaled by the sum of the magnitudes of the entries
 * of M's first column so V(1) stays modest and the product never overflows. The
 * multishift QR sweep (?laqr5) calls this once per bulge to seed the reflector
 * that chases it. An N other than 2 or 3 leaves V untouched, as the reference
 * returns; a first column that scales to zero gives V == 0 (early return).
 *
 * LAPACK's arithmetic is reproduced verbatim -- the N=2 and N=3 branches, the
 * scale S = |H11-SR2| + |SI2| + |H21| (+ |H31| for N=3), and the grouping of the
 * shift products -- so the oracle agrees to a rounding tolerance.
 *
 * No BLAS and no device kernel: ?laqr1 is O(1) scalar arithmetic on the top-left
 * 3-by-3 of H, so -- like calaman.lanv2 -- it takes the NARROWEST handle that
 * still orders its device reads/writes, a `wwrStream_t` (CLAUDE.md). H is a
 * device matrix the caller owns, read back column by column; V is a device vector
 * this writes, so both are ordered on @p stream and follow the caller's upload of
 * H and precede its later use of V. The shifts are host scalars (the caller holds
 * them). Templated over float and double -- LAPACK ships no complex ?laqr1.
 *
 * Usage:
 *   import calaman.laqr1;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_H: device N-by-N upper-Hessenberg, leading dimension ldh; d_V: length N
 *   calaman::laqr1<double>(stream, 3, d_H, ldh, sr1, si1, sr2, si2, d_V);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.laqr1;

import wwr.runtime_api; // wwrStream_t, wwrMemcpy(Async) + wwrMemcpy{Device,Host}To*, wwrSuccess
import std;             // std::abs

// export import, not a plain import: laqr1 RETURNS calaman::Status, so a consumer
// of `import calaman.laqr1;` must see Status's member functions, not just its
// name -- the same re-export lanv2 / lacpy do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief First column of the shift polynomial (H-s1*I)(H-s2*I) (LAPACK ?laqr1)
///
/// Reads the top-left block of the upper-Hessenberg @p h from the device, forms
/// the scaled first column on the host under LAPACK's exact algorithm, and writes
/// the length-@p n vector @p v back to the device. @p n must be 2 or 3; any other
/// value leaves @p v untouched and returns success, as the reference does. The
/// device reads/writes are ordered on @p stream, so they follow the caller's
/// upload of @p h and precede any use of @p v.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the device reads/writes are ordered on
/// @param n Order of the block; 2 or 3 (any other value is a no-op)
/// @param h Device upper-Hessenberg matrix, column-major, leading dimension @p ldh
/// @param ldh Leading dimension of @p h
/// @param sr1 Real part of the first shift s1
/// @param si1 Imaginary part of the first shift s1
/// @param sr2 Real part of the second shift s2
/// @param si2 Imaginary part of the second shift s2
/// @param v Device length-@p n vector; overwritten with the scaled first column
/// @return The Status of the failing copy, otherwise wwr::wwrSuccess
export template<typename T>
Status laqr1(const wwr::wwrStream_t stream, const int n, const T *h, const int ldh, const T sr1,
             const T si1, const T sr2, const T si2, T *v) {
  if (n != 2 && n != 3) {
    return wwr::wwrSuccess;
  }

  // Read the top-left n-by-n of H column by column: column j starts at h + j*ldh.
  // N=2 needs H(1,1..2), H(2,1..2); N=3 adds the third row/column. Only the
  // entries the branches touch are copied.
  T h11{}, h12{}, h21{}, h22{};
  T h13{}, h23{}, h31{}, h32{}, h33{};
  CLM_TRY(wwr::wwrMemcpyAsync(&h11, h, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&h21, h + 1, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&h12, h + ldh, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&h22, h + ldh + 1, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  if (n == 3) {
    CLM_TRY(wwr::wwrMemcpyAsync(&h31, h + 2, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(&h32, h + ldh + 2, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(wwr::wwrMemcpyAsync(&h13, h + 2 * ldh, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(
        wwr::wwrMemcpyAsync(&h23, h + 2 * ldh + 1, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
    CLM_TRY(
        wwr::wwrMemcpyAsync(&h33, h + 2 * ldh + 2, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  }
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  constexpr T zero{0};
  T v1{}, v2{}, v3{};

  if (n == 2) {
    const T s = std::abs(h11 - sr2) + std::abs(si2) + std::abs(h21);
    if (s == zero) {
      v1 = zero;
      v2 = zero;
    } else {
      const T h21s = h21 / s;
      v1 = h21s * h12 + (h11 - sr1) * ((h11 - sr2) / s) - si1 * (si2 / s);
      v2 = h21s * (h11 + h22 - sr1 - sr2);
    }
  } else {
    const T s = std::abs(h11 - sr2) + std::abs(si2) + std::abs(h21) + std::abs(h31);
    if (s == zero) {
      v1 = zero;
      v2 = zero;
      v3 = zero;
    } else {
      const T h21s = h21 / s;
      const T h31s = h31 / s;
      v1 = (h11 - sr1) * ((h11 - sr2) / s) - si1 * (si2 / s) + h12 * h21s + h13 * h31s;
      v2 = h21s * (h11 + h22 - sr1 - sr2) + h23 * h31s;
      v3 = h31s * (h11 + h33 - sr1 - sr2) + h21s * h32;
    }
  }

  // Write the first column back on the same stream, then block so the host locals
  // outlive the copies.
  CLM_TRY(wwr::wwrMemcpyAsync(v, &v1, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(v + 1, &v2, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  if (n == 3) {
    CLM_TRY(wwr::wwrMemcpyAsync(v + 2, &v3, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  }
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::wwrSuccess;
}

} // namespace calaman
