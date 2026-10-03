// Tests for calaman.experimental.byte_transpose: the significance-grouped byte
// layout and its inverse. Pure host (no device memory, no kernel, no oracle),
// so the suite is NOT marked REQUIRES_GPU and runs on a card-less CI runner.
//
// The roundtrip cases are the real assurance: untranspose(transpose(v)) == v on
// nontrivial data (random, patterned, and boundary values) for a range of
// lengths. One layout case pins the byte ordering itself -- that block 0 holds
// the MSBs and block 7 the LSBs -- so a roundtrip-preserving but mis-ordered
// implementation (e.g. LSB-first) is still caught.

#include <gtest/gtest.h>

import std;

import calaman.experimental.byte_transpose;

namespace calaman::experimental {
namespace {

// A reproducible mixed bag: a linear-congruential stream xored with the index
// so neighbouring values differ in every byte, not just the low ones.
std::vector<std::uint64_t> make_values(std::size_t n) {
  std::vector<std::uint64_t> v(n);
  std::uint64_t state = 0x9e3779b97f4a7c15ull;
  for (std::size_t i = 0; i < n; ++i) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    v[i] = state ^ (static_cast<std::uint64_t>(i) * 0x0101010101010101ull);
  }
  return v;
}

// Roundtrip: untranspose(transpose(v)) must reproduce v exactly, for a spread
// of lengths including 0 and 1.
TEST(ByteTransposeTest, RoundtripRandomLengths) {
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                        std::size_t{7}, std::size_t{64}, std::size_t{1000},
                        std::size_t{4097}}) {
    const std::vector<std::uint64_t> v = make_values(n);
    std::vector<std::uint64_t> planes(n);
    std::vector<std::uint64_t> back(n);

    byte_transpose(v.data(), planes.data(), n);
    byte_untranspose(planes.data(), back.data(), n);

    EXPECT_EQ(back, v) << "roundtrip failed at n = " << n;
  }
}

// Boundary values (all-zero, all-ones, and the two single-bit extremes) survive
// the roundtrip -- these are where an off-by-one in the shift would show.
TEST(ByteTransposeTest, RoundtripBoundaryValues) {
  const std::vector<std::uint64_t> v = {
      0ull, ~0ull, 1ull, 0x8000000000000000ull, 0x00000000000000FFull,
      0xFF00000000000000ull, 0x0123456789ABCDEFull, 0xFEDCBA9876543210ull};
  std::vector<std::uint64_t> planes(v.size());
  std::vector<std::uint64_t> back(v.size());

  byte_transpose(v.data(), planes.data(), v.size());
  byte_untranspose(planes.data(), back.data(), v.size());

  EXPECT_EQ(back, v);
}

// Pin the layout contract: block b (bytes [b*n, (b+1)*n)) must hold byte
// (7 - b) of every value -- block 0 the MSBs, block 7 the LSBs.
TEST(ByteTransposeTest, LayoutIsSignificanceGrouped) {
  const std::vector<std::uint64_t> v = make_values(37);
  const std::size_t n = v.size();
  std::vector<std::uint64_t> planes(n);
  byte_transpose(v.data(), planes.data(), n);

  const auto* bytes = reinterpret_cast<const std::byte*>(planes.data());
  for (std::size_t block = 0; block < 8; ++block) {
    const unsigned shift = 8u * (7u - static_cast<unsigned>(block));
    for (std::size_t i = 0; i < n; ++i) {
      const auto expected =
          static_cast<std::uint8_t>((v[i] >> shift) & 0xFFu);
      EXPECT_EQ(std::to_integer<std::uint8_t>(bytes[block * n + i]), expected)
          << "block " << block << ", index " << i;
    }
  }
}

}  // namespace
}  // namespace calaman::experimental
