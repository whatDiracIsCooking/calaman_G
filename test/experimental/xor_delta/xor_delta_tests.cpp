// Suite for calaman.experimental.xor_delta. Host-only ON PURPOSE: the module
// allocates no device memory and launches no kernel, so this is NOT labeled
// REQUIRES_GPU and runs on a card-less CI runner. No oracle either -- XOR delta
// coding is exact integer bookkeeping, so the checks are against literals and
// the round-trip identity, not the reference LAPACK.

#include <gtest/gtest.h>

import std;

import calaman.experimental.xor_delta;

namespace calaman::experimental {
namespace {

TEST(XorDeltaTests, EncodeProducesXorOfAdjacentOriginals) {
  // data[0] is the untouched baseline; data[i] becomes orig[i] ^ orig[i-1].
  std::vector<std::uint64_t> v{10, 11, 11, 20};
  xor_delta_encode(v.data(), v.size());
  EXPECT_EQ(v[0], 10u);           // baseline unchanged
  EXPECT_EQ(v[1], 10u ^ 11u);
  EXPECT_EQ(v[2], 11u ^ 11u);     // equal neighbours -> 0
  EXPECT_EQ(v[3], 11u ^ 20u);
}

TEST(XorDeltaTests, DecodeInvertsEncode) {
  const std::vector<std::uint64_t> original{
      0, 1, 1, 42, 42, 0xFFFFFFFFFFFFFFFFull, 7, 7};
  std::vector<std::uint64_t> v = original;
  xor_delta_encode(v.data(), v.size());
  xor_delta_decode(v.data(), v.size());
  EXPECT_EQ(v, original);
}

TEST(XorDeltaTests, RoundTripOnLongerStream) {
  std::vector<std::uint64_t> v(1000);
  std::uint64_t x = 0x1234567u;
  for (auto& e : v) {
    // A cheap deterministic LCG -- Math.random() is not needed and the module
    // is exact, so any reproducible fill exercises the round trip.
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    e = x;
  }
  const std::vector<std::uint64_t> original = v;
  xor_delta_encode(v.data(), v.size());
  xor_delta_decode(v.data(), v.size());
  EXPECT_EQ(v, original);
}

TEST(XorDeltaTests, SmallCountsAreNoOps) {
  // count <= 1 returns immediately: nothing to difference against.
  std::vector<std::uint64_t> empty;
  xor_delta_encode(empty.data(), empty.size()); // nullptr data, count 0
  xor_delta_decode(empty.data(), empty.size());
  EXPECT_TRUE(empty.empty());

  std::vector<std::uint64_t> one{99};
  xor_delta_encode(one.data(), one.size());
  EXPECT_EQ(one[0], 99u);
  xor_delta_decode(one.data(), one.size());
  EXPECT_EQ(one[0], 99u);
}

} // namespace
} // namespace calaman::experimental
