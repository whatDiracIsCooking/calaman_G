/**
 * @file block_reduce_tests.cpp
 * @brief GPU suite for common/block_reduce.cuh: every overload, at one warp and
 *        at four, over sums, NaN propagation, partial blocks, thread order and
 *        back-to-back calls
 *
 * Every launch runs one block through block_reduce_bridge.h and checks EVERY
 * thread, since the contract is the fold in all of them. Inactive threads are
 * seeded with poison so a thread >= nactive leaking into the fold shows. Needs
 * a card (REQUIRES_GPU).
 */
#include "block_reduce_bridge.h"
#include "mat2.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace {

using calaman::test::block_threads;
using calaman::test::BlockReduceShape;
using calaman::test::mat2_mul;
using calaman::test::mat2_of;

constexpr float kPoison = 1.0e30F;

constexpr std::array kShapes = {BlockReduceShape::kTree, BlockReduceShape::kBlock};
constexpr std::array kWarps = {1U, 4U};

const char *name(const BlockReduceShape shape) {
  switch (shape) {
  case BlockReduceShape::kTree:
    return "tree";
  case BlockReduceShape::kBlock:
    return "block";
  }
  return "?";
}

/// in[i] = i + 1 for i < nactive, kPoison past it.
std::vector<float> iota_with_poison(const unsigned int n, const unsigned int nactive) {
  std::vector<float> in(n, kPoison);
  for (unsigned int i = 0; i < nactive; ++i) {
    in[i] = static_cast<float>(i + 1);
  }
  return in;
}

/// Calls @p check(shape, warps) for every overload and block size.
template<typename Check>
void for_each_config(const Check &check) {
  for (const BlockReduceShape shape : kShapes) {
    for (const unsigned int warps : kWarps) {
      SCOPED_TRACE(::testing::Message() << name(shape) << ", " << warps << " warp(s)");
      check(shape, warps);
    }
  }
}

TEST(CommonBlockReduceTests, SumEveryPartialBlock) {
  for_each_config([](const BlockReduceShape shape, const unsigned int warps) {
    const unsigned int n = block_threads(warps);
    for (unsigned int nactive = 1; nactive <= n; ++nactive) {
      const std::vector<float> in = iota_with_poison(n, nactive);
      std::vector<float> out(n);
      ASSERT_EQ(calaman::test::block_reduce_sum(shape, warps, in.data(), nactive, out.data()), 0);
      // Integers this small are exact in float, so the sum is exact in any order.
      const float expected = static_cast<float>(nactive) * static_cast<float>(nactive + 1) / 2.0F;
      for (unsigned int t = 0; t < n; ++t) {
        ASSERT_EQ(out[t], expected) << "thread " << t << ", nactive " << nactive;
      }
    }
  });
}

// Back-to-back calls: the tree reuses its own buffer behind its leading
// barrier, the warp overload alternates two caller buffers with no barrier.
// Either slipping lets a fast warp overwrite a partial a slow one has not read.
TEST(CommonBlockReduceTests, ChainedCallsAgree) {
  for_each_config([](const BlockReduceShape shape, const unsigned int warps) {
    const unsigned int n = block_threads(warps);
    const std::vector<float> in = iota_with_poison(n, n);
    std::vector<float> out(n);
    ASSERT_EQ(calaman::test::block_reduce_chained_sum(shape, warps, in.data(), n, out.data()), 0);
    const float first = static_cast<float>(n) * static_cast<float>(n + 1) / 2.0F;
    const float expected = first * static_cast<float>(n + 1); // sum(in[t] + first)
    for (unsigned int t = 0; t < n; ++t) {
      ASSERT_EQ(out[t], expected) << "thread " << t;
    }
  });
}

TEST(CommonBlockReduceTests, MaxNanFindsMaxAnywhere) {
  for_each_config([](const BlockReduceShape shape, const unsigned int warps) {
    const unsigned int n = block_threads(warps);
    for (unsigned int argmax = 0; argmax < n; ++argmax) {
      std::vector<float> in(n, -1.0F);
      in[argmax] = 42.0F;
      std::vector<float> out(n);
      ASSERT_EQ(calaman::test::block_reduce_max_nan(shape, warps, in.data(), n, out.data()), 0);
      for (unsigned int t = 0; t < n; ++t) {
        ASSERT_EQ(out[t], 42.0F) << "thread " << t << ", argmax " << argmax;
      }
    }
  });
}

TEST(CommonBlockReduceTests, MaxNanPropagatesNan) {
  for_each_config([](const BlockReduceShape shape, const unsigned int warps) {
    const unsigned int n = block_threads(warps);
    std::vector<float> in(n, 1.0F);
    in[n - 1] = std::nanf("");
    std::vector<float> out(n);
    ASSERT_EQ(calaman::test::block_reduce_max_nan(shape, warps, in.data(), n, out.data()), 0);
    for (unsigned int t = 0; t < n; ++t) {
      EXPECT_TRUE(std::isnan(out[t])) << "thread " << t;
    }
  });
}

TEST(CommonBlockReduceTests, MaxNanIgnoresInactiveNan) {
  for_each_config([](const BlockReduceShape shape, const unsigned int warps) {
    const unsigned int n = block_threads(warps);
    std::vector<float> in(n, std::nanf(""));
    in[0] = 3.0F;
    in[1] = 5.0F;
    std::vector<float> out(n);
    ASSERT_EQ(calaman::test::block_reduce_max_nan(shape, warps, in.data(), 2, out.data()), 0);
    for (unsigned int t = 0; t < n; ++t) {
      EXPECT_EQ(out[t], 5.0F) << "thread " << t;
    }
  });
}

// The order probe: a fold that pairs threads out of order (across warps as
// well as within one) passes every commutative case above and fails here.
TEST(CommonBlockReduceTests, NonCommutativeKeepsThreadOrder) {
  for_each_config([](const BlockReduceShape shape, const unsigned int warps) {
    const unsigned int n = block_threads(warps);
    for (unsigned int nactive = 1; nactive <= n; ++nactive) {
      std::vector<unsigned long long> in(n, 0);
      for (unsigned int i = 0; i < nactive; ++i) {
        in[i] = mat2_of(i);
      }
      unsigned long long expected = in[0];
      for (unsigned int i = 1; i < nactive; ++i) {
        expected = mat2_mul(expected, in[i]);
      }
      std::vector<unsigned long long> out(n);
      ASSERT_EQ(calaman::test::block_reduce_mat2(shape, warps, in.data(), nactive, out.data()),
                0);
      for (unsigned int t = 0; t < n; ++t) {
        ASSERT_EQ(out[t], expected) << "thread " << t << ", nactive " << nactive;
      }
    }
  });
}

TEST(CommonBlockReduceTests, RejectsUnsupportedWarpCount) {
  std::vector<float> buf(block_threads(2));
  EXPECT_EQ(calaman::test::block_reduce_sum(BlockReduceShape::kBlock, 2, buf.data(), 1, buf.data()),
            -1);
}

} // namespace
