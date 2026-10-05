/**
 * @file warp_reduce_tests.cpp
 * @brief GPU suite for common/warp_reduce.cuh: sums, NaN propagation, partial
 *        tiles, and lane order under a non-commutative op
 *
 * Every launch runs one tile through warp_reduce_bridge.h and checks EVERY
 * lane, since the contract is the fold in all of them. Inactive lanes are
 * seeded with poison so a lane >= nactive leaking into the fold shows. Needs a
 * card (REQUIRES_GPU when wired into CMake).
 */
#include "warp_reduce_bridge.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using calaman::test::kMat2Prime;
using calaman::test::warp_size;

constexpr float kPoison = 1.0e30F;

/// Host 2x2 product mod kMat2Prime, the oracle for the device Mat2Op.
unsigned long long mat2_mul(const unsigned long long x, const unsigned long long y) {
  const auto at = [](const unsigned long long v, const int i) { return (v >> (48 - 16 * i)) & 0xFFFF; };
  const unsigned long long a = (at(x, 0) * at(y, 0) + at(x, 1) * at(y, 2)) % kMat2Prime;
  const unsigned long long b = (at(x, 0) * at(y, 1) + at(x, 1) * at(y, 3)) % kMat2Prime;
  const unsigned long long c = (at(x, 2) * at(y, 0) + at(x, 3) * at(y, 2)) % kMat2Prime;
  const unsigned long long d = (at(x, 2) * at(y, 1) + at(x, 3) * at(y, 3)) % kMat2Prime;
  return (a << 48) | (b << 32) | (c << 16) | d;
}

/// Lane i's matrix: distinct, small entries, so products stay well mixed.
unsigned long long lane_matrix(const unsigned int i) {
  const unsigned long long a = 1 + i;
  const unsigned long long b = 3 * i + 2;
  const unsigned long long c = 7 * i + 5;
  const unsigned long long d = 11 * i + 1;
  return (a << 48) | (b << 32) | (c << 16) | d;
}

/// in[i] = i + 1 for i < nactive, kPoison past it.
std::vector<float> iota_with_poison(const unsigned int nactive) {
  std::vector<float> in(warp_size(), kPoison);
  for (unsigned int i = 0; i < nactive; ++i) {
    in[i] = static_cast<float>(i + 1);
  }
  return in;
}

void expect_sum(const unsigned int nactive) {
  const std::vector<float> in = iota_with_poison(nactive);
  std::vector<float> out(warp_size());
  ASSERT_EQ(calaman::test::warp_reduce_sum(in.data(), nactive, out.data()), 0);
  // Integers this small are exact in float, so the sum is exact in any order.
  const float expected = static_cast<float>(nactive) * static_cast<float>(nactive + 1) / 2.0F;
  for (unsigned int lane = 0; lane < warp_size(); ++lane) {
    EXPECT_EQ(out[lane], expected) << "lane " << lane << ", nactive " << nactive;
  }
}

void expect_mat2(const unsigned int nactive) {
  std::vector<unsigned long long> in(warp_size(), 0);
  for (unsigned int i = 0; i < nactive; ++i) {
    in[i] = lane_matrix(i);
  }
  unsigned long long expected = in[0];
  for (unsigned int i = 1; i < nactive; ++i) {
    expected = mat2_mul(expected, in[i]);
  }
  std::vector<unsigned long long> out(warp_size());
  ASSERT_EQ(calaman::test::warp_reduce_mat2(in.data(), nactive, out.data()), 0);
  for (unsigned int lane = 0; lane < warp_size(); ++lane) {
    EXPECT_EQ(out[lane], expected) << "lane " << lane << ", nactive " << nactive;
  }
}

TEST(CommonWarpReduceTests, SumFullTile) { expect_sum(warp_size()); }

TEST(CommonWarpReduceTests, SumEveryPartialTile) {
  for (unsigned int nactive = 1; nactive < warp_size(); ++nactive) {
    expect_sum(nactive);
  }
}

TEST(CommonWarpReduceTests, MaxNanFindsMaxAnywhere) {
  for (unsigned int argmax = 0; argmax < warp_size(); ++argmax) {
    std::vector<float> in(warp_size(), -1.0F);
    in[argmax] = 42.0F;
    std::vector<float> out(warp_size());
    ASSERT_EQ(calaman::test::warp_reduce_max_nan(in.data(), warp_size(), out.data()), 0);
    for (unsigned int lane = 0; lane < warp_size(); ++lane) {
      EXPECT_EQ(out[lane], 42.0F) << "lane " << lane << ", argmax " << argmax;
    }
  }
}

TEST(CommonWarpReduceTests, MaxNanPropagatesNan) {
  std::vector<float> in(warp_size(), 1.0F);
  in[warp_size() / 2 + 1] = std::nanf("");
  std::vector<float> out(warp_size());
  ASSERT_EQ(calaman::test::warp_reduce_max_nan(in.data(), warp_size(), out.data()), 0);
  for (unsigned int lane = 0; lane < warp_size(); ++lane) {
    EXPECT_TRUE(std::isnan(out[lane])) << "lane " << lane;
  }
}

TEST(CommonWarpReduceTests, MaxNanIgnoresInactiveNan) {
  std::vector<float> in(warp_size(), std::nanf(""));
  in[0] = 3.0F;
  in[1] = 5.0F;
  std::vector<float> out(warp_size());
  ASSERT_EQ(calaman::test::warp_reduce_max_nan(in.data(), 2, out.data()), 0);
  for (unsigned int lane = 0; lane < warp_size(); ++lane) {
    EXPECT_EQ(out[lane], 5.0F) << "lane " << lane;
  }
}

// The order probe: a halving ladder would pair lane 0 with lane W/2 first and
// fail here, while still passing every commutative case above.
TEST(CommonWarpReduceTests, NonCommutativeKeepsLaneOrder) {
  for (unsigned int nactive = 1; nactive <= warp_size(); ++nactive) {
    expect_mat2(nactive);
  }
}

} // namespace
