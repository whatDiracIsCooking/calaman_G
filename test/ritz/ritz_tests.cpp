// Suite for calaman.ritz:
//
//   * ritz_select -- positions per RitzWhich and the nested-selection property
//     (host-only);
//   * classify_ritz -- the convergence predicate, the scale floor, the boundary,
//     the absolute (scale-free) form, a length mismatch and the vacuous empty
//     selection (host-only);
//   * ritz_rotate -- argument checks before any handle use (host-only; the
//     device product is ritz_rotate_tests.cpp).
//
// No card and no oracle needed, so none of this is REQUIRES_GPU.

#include <gtest/gtest.h>

import std;

import wwr.blas;
import calaman.ritz;
import calaman.error_handling;

namespace calaman {
namespace {

constexpr auto kInvalidValue = wwr::WWRBLAS_STATUS_INVALID_VALUE;

// ── ritz_select (host-only) ──────────────────────────────────────────────────

TEST(RitzSelectTests, PositionsPerWhich) {
  using V = std::vector<int>;
  EXPECT_EQ(ritz_select(RitzWhich::smallest, 10, 3), (V{0, 1, 2}));
  EXPECT_EQ(ritz_select(RitzWhich::largest, 10, 3), (V{7, 8, 9}));
  EXPECT_EQ(ritz_select(RitzWhich::both_ends, 10, 3), (V{0, 8, 9})); // 2 top, 1 bottom
  EXPECT_EQ(ritz_select(RitzWhich::both_ends, 10, 4), (V{0, 1, 8, 9}));
  EXPECT_EQ(ritz_select(RitzWhich::both_ends, 10, 1), (V{9}));
  EXPECT_EQ(ritz_select(RitzWhich::largest, 5, 5), (V{0, 1, 2, 3, 4}));
  EXPECT_TRUE(ritz_select(RitzWhich::smallest, 10, 0).empty());
  EXPECT_TRUE(ritz_select(RitzWhich::largest, 10, 11).empty());
  EXPECT_TRUE(ritz_select(RitzWhich::both_ends, 0, 0).empty());
}

TEST(RitzSelectTests, LargerSelectionContainsSmaller) {
  for (const RitzWhich which : {RitzWhich::smallest, RitzWhich::largest, RitzWhich::both_ends}) {
    for (int nev = 1; nev <= 12; ++nev) {
      const std::vector<int> wanted = ritz_select(which, 12, nev);
      for (int k = nev; k <= 12; ++k) {
        const std::vector<int> kept = ritz_select(which, 12, k);
        EXPECT_TRUE(std::ranges::includes(kept, wanted)) << nev << " in " << k;
      }
    }
  }
}

// ── classify_ritz (host-only) ────────────────────────────────────────────────

TEST(RitzClassifyTests, PredicateAndScale) {
  const std::vector<double> values = {-3.0, -1.0, 2.0, 5.0, 0.0};
  const std::vector<double> residuals = {2e-9, 1.0, 2e-12, 0.6, 4e-6};
  // threshold 1e-6 * max(|value|, 4) = 4e-6, 4e-6, 4e-6, 5e-6, 4e-6.
  const auto sel = classify_ritz<double>(values, residuals, 1e-6, 4.0);
  EXPECT_TRUE(sel.index.empty());
  EXPECT_EQ(sel.values, values);
  EXPECT_EQ(sel.residuals, residuals);
  EXPECT_EQ(sel.converged, (std::vector<bool>{true, false, true, false, true})); // last: equality
  EXPECT_EQ(sel.converged_count, 3);
  EXPECT_FALSE(sel.all_converged());
}

TEST(RitzClassifyTests, ValueDominatesASmallScale) {
  const std::vector<float> values = {100.0f, -100.0f};
  const std::vector<float> residuals = {5e-5f, 2e-4f};
  // scale 0: the threshold is tolerance * |value| = 1e-4.
  const auto sel = classify_ritz<float>(values, residuals, 1e-6f, 0.0f);
  EXPECT_EQ(sel.converged, (std::vector<bool>{true, false}));
  EXPECT_EQ(sel.converged_count, 1);
}

TEST(RitzClassifyTests, AbsoluteIgnoresValues) {
  const std::vector<double> values = {1e6, -1e6, 0.0, 0.5};
  const std::vector<double> residuals = {2e-6, 1e-6, 1e-6, 0.0};
  // threshold 1e-6 whatever the value; the scaled form would pass the first two.
  const auto sel = classify_ritz<double>(values, residuals, 1e-6);
  EXPECT_TRUE(sel.index.empty());
  EXPECT_EQ(sel.values, values);
  EXPECT_EQ(sel.residuals, residuals);
  EXPECT_EQ(sel.converged, (std::vector<bool>{false, true, true, true})); // 2nd, 3rd: equality
  EXPECT_EQ(sel.converged_count, 3);
  EXPECT_TRUE(classify_ritz<double>(values, std::vector<double>{0.0}, 1.0).values.empty());
}

TEST(RitzClassifyTests, AllConvergedAndEmpty) {
  const std::vector<double> values = {1.0, 2.0};
  const std::vector<double> residuals = {0.0, 0.0};
  EXPECT_TRUE(classify_ritz<double>(values, residuals, 1e-12, 1.0).all_converged());

  const auto none = classify_ritz<double>({}, {}, 1e-6, 1.0);
  EXPECT_TRUE(none.values.empty());
  EXPECT_TRUE(none.all_converged()); // vacuously
}

TEST(RitzClassifyTests, LengthMismatchIsEmpty) {
  const std::vector<double> values = {1.0, 2.0};
  const std::vector<double> residuals = {0.0};
  const auto sel = classify_ritz<double>(values, residuals, 1e-6, 1.0);
  EXPECT_TRUE(sel.values.empty());
  EXPECT_TRUE(sel.residuals.empty());
  EXPECT_TRUE(sel.converged.empty());
  EXPECT_EQ(sel.converged_count, 0);
}

// ── ritz_rotate argument checks (host-only) ──────────────────────────────────

TEST(RitzRotateArgTests, RejectsBadArguments) {
  const wwr::wwrblasHandle_t no_blas{}; // never used on the rejection path
  double b = 0.0;
  double s = 0.0;
  double c = 0.0;
  // n = 8, k = 5, count = 2, ldb = 8, lds = 5, ldc = 8 is the valid shape.
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 2, nullptr, 8, &s, 5, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 2, &b, 8, nullptr, 5, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 2, &b, 8, &s, 5, nullptr, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 0, 5, 2, &b, 8, &s, 5, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 0, &b, 8, &s, 5, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 6, &b, 8, &s, 5, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 2, &b, 7, &s, 5, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 2, &b, 8, &s, 4, &c, 8).code, kInvalidValue);
  EXPECT_EQ(ritz_rotate<double>(no_blas, 8, 5, 2, &b, 8, &s, 5, &c, 7).code, kInvalidValue);
}

} // namespace
} // namespace calaman
