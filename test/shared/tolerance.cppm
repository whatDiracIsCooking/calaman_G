/**
 * @file tolerance.cppm
 * @brief Shared numerical tolerance for the linalg oracle suites
 *
 * One place for the tolerance convention every factorization test
 * (larfg/larf/laqp2/laqps/geqp3 and the linalg scaffold) checks against, so the
 * constant lives once rather than
 * drifting per suite: O(eps * ||A|| * min(m,n)) -- the first-order bound on the
 * error a backward-stable factorization of an m-by-n matrix of norm ||A||
 * accumulates, with a modest factor (kTolFactor) absorbing the big-O constant.
 *
 * Deliberately under test/, not src/: nothing calaman ships has a tolerance (a
 * factorization is exact arithmetic on the device); only a test COMPARING the
 * device result to the reference LAPACK needs one. Pure host math -- no device
 * code, no GoogleTest -- so like the other test/shared/ modules it carries
 * neither a .cu nor an implementation unit.
 *
 * Usage:
 *   import calaman.test.shared.tolerance;
 *   using calaman::test::factorization_tol;
 *   const double tol = factorization_tol<double>(frobenius_norm(a), m, n);
 *   EXPECT_LE(residual, tol);
 */

export module calaman.test.shared.tolerance;

import std;

export namespace calaman::test {

/// @brief Machine epsilon for T -- the unit roundoff the bound is built on
template<typename T>
constexpr T eps() {
  return std::numeric_limits<T>::epsilon();
}

/// @brief The big-O constant made concrete: generous enough that a correct
///        backend-vs-reference factorization passes across summation orders,
///        tight enough that a wrong result still fails. One value serves both
///        precisions because eps<T>() already carries the per-type scale.
template<typename T>
inline constexpr T kTolFactor = T{64};

/// @brief Frobenius norm of a host matrix (or vector), ||A||_F = sqrt(sum a^2)
///
/// Column-major vs row-major is irrelevant -- the Frobenius norm sums every
/// stored element -- so callers pass the dense storage as-is.
template<typename T>
T frobenius_norm(const std::vector<T> &a) {
  T sum{};
  for (const T v : a) {
    sum += v * v;
  }
  return std::sqrt(sum);
}

/// @brief The shared factorization tolerance kTolFactor * eps * ||A|| * min(m,n)
///
/// @tparam T Element type (float, double)
/// @param norm_a A norm of the input, typically frobenius_norm(A)
/// @param m Row count
/// @param n Column count
template<typename T>
T factorization_tol(T norm_a, std::size_t m, std::size_t n) {
  const auto k = static_cast<T>(std::min(m, n));
  return kTolFactor<T> * eps<T>() * norm_a * k;
}

} // namespace calaman::test
