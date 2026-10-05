/**
 * @file compute_quadrature.cppm
 * @brief The FEAST contour quadrature for an interval, and the filter it defines
 *
 * The :compute_quadrature partition of calaman.feast.
 *
 * The spectral projector onto the eigenvectors of A whose eigenvalues lie inside
 * a closed contour C is
 *
 *   P = (1 / 2 pi i) \oint_C (z I - A)^{-1} dz.
 *
 * FEAST takes C to be the circle through Emin and Emax -- centre
 * c = (Emin + Emax) / 2, radius r = (Emax - Emin) / 2 -- so z = c + r e^{i theta}.
 * For real symmetric A, (conj(z) I - A)^{-1} = conj((z I - A)^{-1}): the lower
 * half of the circle contributes the conjugate of the upper half, and
 *
 *   P = (1 / pi) Re \int_0^pi r e^{i theta} (z(theta) I - A)^{-1} d theta.
 *
 * Gauss-Legendre in theta = (pi / 2)(1 - x), x in [-1, 1], turns that into
 *
 *   P ~ sum_e Re[ w_e (Z_e I - A)^{-1} ],
 *   Z_e = c + r e^{i theta_e},   w_e = (omega_e / 2) r e^{i theta_e}.
 *
 * On an eigenvector with eigenvalue lambda the sum is multiplication by the
 * rational filter
 *
 *   rho(lambda) = sum_e Re[ w_e / (Z_e - lambda) ],
 *
 * which is 1 at the centre exactly (every term is omega_e / 2, and the weights
 * sum to 2), 1/2 at Emin and Emax, and falls away outside -- faster with more
 * nodes. FEAST is subspace iteration with rho(A).
 *
 * The overall sign of the weights is immaterial to FEAST, since Rayleigh-Ritz
 * sees only the span of rho(A) Y. It is chosen so rho approximates the projector
 * itself (+1 inside) rather than its negative, which is what lets the tests
 * compare the device filter against rho directly.
 *
 * The nodes and weights live in device::FeastContour as plain real components
 * (feast_bridge.h explains why no complex type crosses that boundary); the host
 * analysis helper feast_rational_filter rebuilds the complex values locally with
 * std::complex.
 */

module;

#include "feast_bridge.h"

export module calaman.feast:compute_quadrature;

import std;
import calaman.common;      // kPi, real_fp
import :feast_quadrature;

namespace calaman {

/**
 * @brief FEAST contour nodes and weights for the interval [Emin, Emax], packed
 *        as the kernels take them.
 *
 * @tparam T Real floating-point type (float or double).
 * @tparam N Number of quadrature points (4 or 8).
 * @param Emin Left end of the interval (Emin < Emax).
 * @param Emax Right end.
 */
template<calaman::real_fp T, std::size_t N>
  requires(N == 4 || N == 8)
device::FeastContour<T> make_feast_contour(const T Emin, const T Emax) {
  const auto &gl = feast_gauss_legendre<T, N>();
  const T r = (Emax - Emin) / T(2);
  const T center = (Emin + Emax) / T(2);

  device::FeastContour<T> contour{};
  for (std::size_t e = 0; e < N; ++e) {
    const T theta = (kPi<T> / T(2)) * (T(1) - gl.x[e]);
    const T re = r * std::cos(theta);
    const T im = r * std::sin(theta);
    const T half_omega = gl.omega[e] / T(2);

    contour.zr[e] = center + re;
    contour.zi[e] = im;
    contour.wr[e] = half_omega * re;
    contour.wi[e] = half_omega * im;
  }
  contour.count = static_cast<int>(N);
  return contour;
}

/**
 * @brief rho(lambda), the factor the FEAST filter for [Emin, Emax] multiplies an
 *        eigenvector with eigenvalue lambda by. Host-side, for analysis and tests.
 */
export template<calaman::real_fp T, std::size_t N>
  requires(N == 4 || N == 8)
T feast_rational_filter(const T Emin, const T Emax, const T lambda) {
  const device::FeastContour<T> c = make_feast_contour<T, N>(Emin, Emax);

  T rho{0};
  for (std::size_t e = 0; e < N; ++e) {
    const std::complex<T> ze{c.zr[e], c.zi[e]};
    const std::complex<T> we{c.wr[e], c.wi[e]};
    rho += (we / (ze - lambda)).real();
  }
  return rho;
}

} // namespace calaman
