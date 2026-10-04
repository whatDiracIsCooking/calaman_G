/**
 * @file lanv2.cppm
 * @brief The calaman.lanv2 module -- Schur factorisation of a real 2-by-2
 *        nonsymmetric matrix in standard form, LAPACK's ?lanv2
 *
 * Standardises [a b; c d] = [cs -sn; sn cs] [aa bb; cc dd] [cs sn; -sn cs] so
 * that either cc == 0 (aa, dd are real eigenvalues) or aa == dd with bb*cc < 0
 * (aa +/- sqrt(bb*cc) is a complex-conjugate pair). On return a, b, c, d hold
 * the standardised block, (rt1r, rt1i) and (rt2r, rt2i) the two eigenvalues
 * (rt1i > 0 for a complex pair), and (cs, sn) the rotation. This is the 2-by-2
 * deflation step ?lahqr / ?hseqr lean on in the real Schur reduction.
 *
 * LAPACK's algorithm is reproduced verbatim -- V. Sima's cancellation-reducing
 * revision and all -- so the oracle agrees to a rounding tolerance: the sign
 * convention via std::copysign (Fortran SIGN(A,B)), dlapy2 via std::hypot, and
 * the DLAMCH scaling constants (eps, safmin, safmn2/safmx2) rebuilt from
 * std::numeric_limits so the near-equal-eigenvalue normalisation loop matches.
 * The per-type constants mean float standardises against SLAMCH-scale bounds and
 * double against DLAMCH-scale, exactly as ?lanv2 branches on precision.
 *
 * There is NO BLAS and NO device code here: ?lanv2 is pure scalar arithmetic on
 * the four entries of one 2-by-2, not a linear-algebra primitive. So, unlike its
 * QR-algorithm siblings (calaman.larfg takes a BLAS handle for nrm2/scal), lanv2
 * takes only a `wwrStream_t` -- the NARROWEST handle that still orders its device
 * reads/writes (CLAUDE.md, lacpy's note on taking a stream). a, b, c, d are
 * device pointers the caller owns (the block lives in the matrix being reduced);
 * they are read back, standardised on the host, and written back on that stream,
 * so they follow the caller's uploads and precede its later rotations. The six
 * eigenvalue/rotation scalars are host outputs this writes (as larfg writes tau
 * and beta), ready to feed a host-pointer-mode rot applied to the rest of H.
 *
 * Templated over `float` and `double`. Complex ?lanv2 does not exist in LAPACK
 * (the complex Schur form needs no 2-by-2 standardisation), so there is nothing
 * to extend here -- the real pair is the whole routine.
 *
 * Usage:
 *   import calaman.lanv2;
 *   import wwr.runtime_api;   // wwrStream_t, wwrStreamCreate
 *   wwr::wwrStream_t stream{};
 *   wwr::wwrStreamCreate(&stream);
 *   // d_a, d_b, d_c, d_d: device scalars of the 2-by-2 block
 *   double rt1r, rt1i, rt2r, rt2i, cs, sn;
 *   calaman::lanv2(stream, d_a, d_b, d_c, d_d,
 *                  &rt1r, &rt1i, &rt2r, &rt2i, &cs, &sn);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.lanv2;

import wwr.runtime_api; // wwrStream_t, wwrMemcpy(Async) + wwrMemcpy{Device,Host}To*, wwrSuccess
import std;             // std::abs, std::sqrt, std::hypot, std::copysign, std::pow, numeric_limits

// export import, not a plain import: lanv2 RETURNS calaman::Status, so a consumer
// of `import calaman.lanv2;` must see Status's member functions, not just its
// name -- the same re-export lacpy / larfg do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

/// @brief Schur-standardise a real 2-by-2 nonsymmetric block (LAPACK ?lanv2)
///
/// Reads the block [a b; c d] from the device, forms its standardised Schur form
/// on the host under LAPACK's exact algorithm, and writes the standardised a, b,
/// c, d back to the device. The two eigenvalues are written to (@p rt1r,@p rt1i)
/// and (@p rt2r,@p rt2i) -- rt1i > 0 for a complex-conjugate pair, both imaginary
/// parts zero for real eigenvalues -- and the rotation to (@p cs,@p sn). The
/// device reads and writes are ordered on @p stream, so they follow the caller's
/// upload of the block and precede any rotation it later applies.
///
/// Short-circuits: if any device copy does not succeed its Status is returned and
/// the host outputs are left unwritten.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param stream Stream the block's device reads/writes are ordered on
/// @param a Device scalar (1,1); read, overwritten with the standardised entry
/// @param b Device scalar (1,2); read, overwritten with the standardised entry
/// @param c Device scalar (2,1); read, overwritten with the standardised entry
/// @param d Device scalar (2,2); read, overwritten with the standardised entry
/// @param rt1r Host scalar; real part of the first eigenvalue
/// @param rt1i Host scalar; imaginary part of the first eigenvalue (>= 0)
/// @param rt2r Host scalar; real part of the second eigenvalue
/// @param rt2i Host scalar; imaginary part of the second eigenvalue
/// @param cs Host scalar; cosine of the standardising rotation
/// @param sn Host scalar; sine of the standardising rotation
/// @return The Status of the failing copy, otherwise wwr::wwrSuccess
export template<typename T>
Status lanv2(const wwr::wwrStream_t stream, T *a, T *b, T *c, T *d, T *rt1r, T *rt1i, T *rt2r,
             T *rt2i, T *cs, T *sn) {
  // Read the 2-by-2 block back and block until it lands; everything below is
  // host arithmetic on these four values.
  T va{}, vb{}, vc{}, vd{};
  CLM_TRY(wwr::wwrMemcpyAsync(&va, a, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&vb, b, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&vc, c, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&vd, d, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  constexpr T zero{0}, half{0.5}, one{1}, two{2}, multpl{4};

  // DLAMCH constants, per type: eps = DLAMCH('P') (= numeric_limits epsilon),
  // safmin = DLAMCH('S'), base = DLAMCH('B'); safmn2/safmx2 bound the
  // near-equal-eigenvalue normalisation loop, built exactly as ?lanv2 does.
  const T eps = std::numeric_limits<T>::epsilon();
  const T safmin = std::numeric_limits<T>::min();
  const T base = static_cast<T>(std::numeric_limits<T>::radix);
  const T safmn2 = std::pow(
      base, static_cast<T>(static_cast<int>(std::log(safmin / eps) / std::log(base) / two)));
  const T safmx2 = one / safmn2;

  T vcs = one, vsn = zero;

  if (vc == zero) {
    // Already block upper triangular: identity rotation.
    vcs = one;
    vsn = zero;
  } else if (vb == zero) {
    // Swap rows and columns.
    vcs = zero;
    vsn = one;
    const T temp = vd;
    vd = va;
    va = temp;
    vb = -vc;
    vc = zero;
  } else if ((va - vd) == zero && std::copysign(one, vb) != std::copysign(one, vc)) {
    vcs = one;
    vsn = zero;
  } else {
    T temp = va - vd;
    T p = half * temp;
    const T bcmax = std::max(std::abs(vb), std::abs(vc));
    const T bcmis =
        std::min(std::abs(vb), std::abs(vc)) * std::copysign(one, vb) * std::copysign(one, vc);
    T scale = std::max(std::abs(p), bcmax);
    T z = (p / scale) * p + (bcmax / scale) * bcmis;

    if (z >= multpl * eps) {
      // Real eigenvalues. Compute a and d.
      z = p + std::copysign(std::sqrt(scale) * std::sqrt(z), p);
      va = vd + z;
      vd = vd - (bcmax / z) * bcmis;
      // Compute b and the rotation matrix.
      const T tau = std::hypot(vc, z);
      vcs = z / tau;
      vsn = vc / tau;
      vb = vb - vc;
      vc = zero;
    } else {
      // Complex eigenvalues, or real (almost) equal eigenvalues. Make the
      // diagonal elements equal, rescaling temp/sigma to a safe range first.
      T sigma = vb + vc;
      for (int count = 1;; ++count) {
        scale = std::max(std::abs(temp), std::abs(sigma));
        if (scale >= safmx2) {
          sigma = sigma * safmn2;
          temp = temp * safmn2;
          if (count <= 20) {
            continue;
          }
        }
        if (scale <= safmn2) {
          sigma = sigma * safmx2;
          temp = temp * safmx2;
          if (count <= 20) {
            continue;
          }
        }
        break;
      }
      p = half * temp;
      T tau = std::hypot(sigma, temp);
      vcs = std::sqrt(half * (one + std::abs(sigma) / tau));
      vsn = -(p / (tau * vcs)) * std::copysign(one, sigma);

      // [aa bb; cc dd] = [a b; c d] [cs -sn; sn cs]
      const T aa = va * vcs + vb * vsn;
      const T bb = -va * vsn + vb * vcs;
      const T cc = vc * vcs + vd * vsn;
      const T dd = -vc * vsn + vd * vcs;

      // [a b; c d] = [cs sn; -sn cs] [aa bb; cc dd]. Parentheses mirror the
      // reference's FMA-suppressing grouping (Reference-LAPACK issue 1031).
      va = aa * vcs + cc * vsn;
      vb = (bb * vcs) + (dd * vsn);
      vc = -(aa * vsn) + (cc * vcs);
      vd = -bb * vsn + dd * vcs;

      temp = half * (va + vd);
      va = temp;
      vd = temp;

      if (vc != zero) {
        if (vb != zero) {
          if (std::copysign(one, vb) == std::copysign(one, vc)) {
            // Real eigenvalues: reduce to upper triangular form.
            const T sab = std::sqrt(std::abs(vb));
            const T sac = std::sqrt(std::abs(vc));
            p = std::copysign(sab * sac, vc);
            tau = one / std::sqrt(std::abs(vb + vc));
            va = temp + p;
            vd = temp - p;
            vb = vb - vc;
            vc = zero;
            const T cs1 = sab * tau;
            const T sn1 = sac * tau;
            const T rot = vcs * cs1 - vsn * sn1;
            vsn = vcs * sn1 + vsn * cs1;
            vcs = rot;
          }
        } else {
          vb = -vc;
          vc = zero;
          const T rot = vcs;
          vcs = -vsn;
          vsn = rot;
        }
      }
    }
  }

  // Write the standardised block back on the same stream, then block so the
  // host locals outlive the copies.
  CLM_TRY(wwr::wwrMemcpyAsync(a, &va, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(b, &vb, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(c, &vc, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(d, &vd, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  // Store eigenvalues in (rt1r, rt1i) and (rt2r, rt2i).
  *rt1r = va;
  *rt2r = vd;
  if (vc == zero) {
    *rt1i = zero;
    *rt2i = zero;
  } else {
    const T ri = std::sqrt(std::abs(vb)) * std::sqrt(std::abs(vc));
    *rt1i = ri;
    *rt2i = -ri;
  }
  *cs = vcs;
  *sn = vsn;
  return wwr::wwrSuccess;
}

} // namespace calaman
