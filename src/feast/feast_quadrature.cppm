/**
 * @file feast_quadrature.cppm
 * @brief Compile-time FEAST contour-integration quadrature constants
 *
 * The :feast_quadrature partition of calaman.feast. Stores the Gauss-Legendre
 * nodes and weights the FEAST eigenvalue algorithm uses on the upper half of its
 * contour (Polizzi, arXiv:0901.2665).
 *
 * For each quadrature point e the contour integration uses
 *   Z_e = center + r * exp(i theta_e)   [runtime: depends on the interval]
 * with center = (Emax + Emin) / 2 and r = (Emax - Emin) / 2; see
 * compute_quadrature.cppm, which turns these constants into Z_e and w_e.
 * Module-internal: nothing here is exported.
 */

export module calaman.feast:feast_quadrature;

import std;
import calaman.common;  // real_fp

namespace calaman {

/**
 * @brief Compile-time FEAST Gauss-Legendre quadrature table.
 *
 * @tparam T Real floating-point type (float or double).
 * @tparam N Number of quadrature points (4 or 8).
 */
template<calaman::real_fp T, std::size_t N>
  requires(N == 4 || N == 8)
struct FeastQuadrature {
  std::array<T, N> x;     ///< GL nodes on [-1, 1]
  std::array<T, N> omega; ///< GL weights
};

/**
 * @brief N=4 FEAST quadrature constants.
 *
 * Nodes and weights from np.polynomial.legendre.leggauss(4).
 */
template<calaman::real_fp T>
inline constexpr FeastQuadrature<T, 4> feast_quadrature_4 = {
    .x =
        {
            T(-0.86113631159405257),
            T(-0.33998104358485626),
            T(0.33998104358485626),
            T(0.86113631159405257),
        },
    .omega =
        {
            T(0.34785484513745357),
            T(0.65214515486254643),
            T(0.65214515486254643),
            T(0.34785484513745357),
        },
};

/**
 * @brief N=8 FEAST quadrature constants.
 *
 * Nodes and weights from np.polynomial.legendre.leggauss(8), matching the
 * reference values in Fig. 2 of Polizzi (arXiv:0901.2665).
 */
template<calaman::real_fp T>
inline constexpr FeastQuadrature<T, 8> feast_quadrature_8 = {
    .x =
        {
            T(-0.96028985649753618),
            T(-0.79666647741362673),
            T(-0.52553240991632899),
            T(-0.18343464249564978),
            T(0.18343464249564978),
            T(0.52553240991632899),
            T(0.79666647741362673),
            T(0.96028985649753618),
        },
    .omega =
        {
            T(0.10122853629037706),
            T(0.22238103445337443),
            T(0.31370664587788688),
            T(0.36268378337836166),
            T(0.36268378337836166),
            T(0.31370664587788688),
            T(0.22238103445337443),
            T(0.10122853629037706),
        },
};

/// @brief The quadrature table for @p N nodes.
template<calaman::real_fp T, std::size_t N>
  requires(N == 4 || N == 8)
constexpr const FeastQuadrature<T, N> &feast_gauss_legendre() {
  if constexpr (N == 4) {
    return feast_quadrature_4<T>;
  } else {
    return feast_quadrature_8<T>;
  }
}

} // namespace calaman
