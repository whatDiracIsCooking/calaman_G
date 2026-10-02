// Scaffold suite for the linalg tier: it proves every linalg module imports and
// links, the shared tolerance helper computes the agreed bound, and the
// reference LAPACK (LAPACKE) is reachable as the oracle -- the pieces every
// reflector/pivoted-QR PR slots into.
//
// Host-only ON PURPOSE: it allocates no device memory and launches no kernel,
// so it is NOT labeled REQUIRES_GPU and runs on a card-less CI runner too. The
// numerical suites that follow (larfg, larf, laqp2, ...) run real kernels and
// will carry REQUIRES_GPU. Built only when calaman::lapack_reference exists
// (CMakeLists.txt returns early otherwise, so its absence is a named missing
// tier, not a silent pass -- docs/architecture.md §3).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

// Imported only to prove every linalg module compiles and links on a card-less
// runner (none of their names are used here); the numerical suites exercise them.
import calaman.larfg;
import calaman.larf;
import calaman.laqp2;
import calaman.laqps;
import calaman.geqp3;
import calaman.test.shared.tolerance;

namespace calaman {
namespace {

using test::eps;
using test::factorization_tol;
using test::frobenius_norm;

TEST(LinalgScaffoldTests, FrobeniusNormIsEuclidean) {
  // 3-4-5: the Frobenius norm of {3, 4} is exactly 5.
  EXPECT_DOUBLE_EQ(frobenius_norm<double>({3.0, 4.0}), 5.0);
  EXPECT_FLOAT_EQ(frobenius_norm<float>({3.0F, 4.0F}), 5.0F);
}

TEST(LinalgScaffoldTests, FactorizationTolMatchesFormula) {
  // kTolFactor * eps * ||A|| * min(m,n), min(m,n) = 3 here.
  const double norm_a = 2.0;
  const double expect = test::kTolFactor<double> * eps<double>() * norm_a * 3.0;
  EXPECT_DOUBLE_EQ(factorization_tol<double>(norm_a, 5, 3), expect);
  EXPECT_GT(factorization_tol<double>(norm_a, 5, 3), 0.0);
}

TEST(LinalgScaffoldTests, OracleIsReachable) {
  // The reference LAPACK is linked and callable: LAPACKE_dlange's Frobenius
  // norm of a small column-major matrix must agree with our host helper. This
  // is the oracle the numerical suites compare against (via calaman::lapack_
  // reference), exercised here only to prove it is wired in.
  const std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}; // 3x2 column-major
  const double ref = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', 3, 2, a.data(), 3);
  EXPECT_NEAR(frobenius_norm<double>(a), ref, factorization_tol<double>(ref, 3, 2));
}

} // namespace
} // namespace calaman
