// Host-only suite for calaman.common's :workspace partition -- the
// WorkspaceLayout byte arithmetic and the carve-once idiom (slices_for +
// carve_workspace). Pure host logic, no device memory and no kernel, so it is
// NOT labeled REQUIRES_GPU and runs on a card-less CI runner.

#include <gtest/gtest.h>

#include <cstddef> // offsetof

import std;

import calaman.common;

namespace calaman {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t aligned(const std::size_t bytes) { return ((bytes + kAlign - 1) / kAlign) * kAlign; }

// A slices struct in the shape the carve-once convention asks for: one carve()
// member over WorkspaceLayout, fixed regions first, the aliasing scratch last.
struct TestSlices {
  double *a = nullptr;
  int *piv = nullptr;
  double *scratch_a = nullptr;
  float *scratch_b = nullptr;

  void carve(WorkspaceLayout &layout, const int n, const int lwork) {
    const std::size_t nz = static_cast<std::size_t>(n);
    a = layout.fixed<double>(nz * nz);
    piv = layout.fixed<int>(nz);
    scratch_a = layout.scratch<double>(static_cast<std::size_t>(lwork));
    scratch_b = layout.scratch<float>(nz);
  }
};

struct NotSlices {};

static_assert(slices_for<TestSlices, int, int>);
static_assert(!slices_for<TestSlices, int, int, int>); // carve() takes exactly (n, lwork)
static_assert(!slices_for<NotSlices, int, int>);       // no carve() at all

TEST(CommonWorkspaceTests, LayoutFixedAccumulatesAndScratchMaxes) {
  WorkspaceLayout layout(nullptr);
  (void)layout.fixed<double>(10);     // 80 bytes -> one 256 block
  (void)layout.fixed<int>(100);       // 400 bytes -> one 512-byte block
  (void)layout.scratch<double>(1000); // 8000 bytes, the larger scratch
  (void)layout.scratch<float>(10);    // aliases, smaller, must not add
  EXPECT_EQ(layout.total(), aligned(80) + aligned(400) + aligned(8000));
}

TEST(CommonWorkspaceTests, CarveWorkspaceSizesAndCarvesIdentically) {
  const int n = 7;
  const int lwork = 33;
  const std::size_t bytes = carve_workspace<TestSlices>(nullptr, nullptr, n, lwork);

  // The sizing call spans exactly the hand-derived layout.
  const std::size_t nz = static_cast<std::size_t>(n);
  const std::size_t fixed_bytes = aligned(nz * nz * sizeof(double)) + aligned(nz * sizeof(int));
  const std::size_t scratch_bytes =
      std::max(aligned(static_cast<std::size_t>(lwork) * sizeof(double)),
               aligned(nz * sizeof(float)));
  EXPECT_EQ(bytes, fixed_bytes + scratch_bytes);

  // The carving call lays the same regions into the provided buffer.
  alignas(kAlign) static std::byte buffer[16 * 1024];
  ASSERT_GE(sizeof(buffer), bytes);
  TestSlices s;
  EXPECT_EQ(carve_workspace(static_cast<void *>(buffer), &s, n, lwork), bytes);

  EXPECT_EQ(reinterpret_cast<std::byte *>(s.a), buffer);
  EXPECT_EQ(reinterpret_cast<std::byte *>(s.piv), buffer + aligned(nz * nz * sizeof(double)));
  // Both scratch regions alias at the end of the fixed run.
  EXPECT_EQ(reinterpret_cast<std::byte *>(s.scratch_a), buffer + fixed_bytes);
  EXPECT_EQ(reinterpret_cast<std::byte *>(s.scratch_b), buffer + fixed_bytes);
}

TEST(CommonWorkspaceTests, CarveWorkspaceNullBaseYieldsNullSlices) {
  TestSlices s;
  const std::size_t bytes = carve_workspace(nullptr, &s, 5, 11);
  EXPECT_GT(bytes, 0U);
  EXPECT_EQ(s.a, nullptr);
  EXPECT_EQ(s.piv, nullptr);
  EXPECT_EQ(s.scratch_a, nullptr);
  EXPECT_EQ(s.scratch_b, nullptr);
}

TEST(CommonWorkspaceTests, LayoutRegionsStartAligned) {
  alignas(kAlign) static std::byte buffer[8 * 1024];
  WorkspaceLayout layout(buffer);
  auto *first = layout.fixed<char>(1);
  auto *second = layout.fixed<double>(3);
  auto *scratch = layout.scratch<double>(5);
  for (const void *p : {static_cast<const void *>(first), static_cast<const void *>(second),
                        static_cast<const void *>(scratch)}) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % kAlign, 0U);
  }
}

// A status block in the shape fixed_struct is for: ints plus an array member.
struct TestStatus {
  int info[3];
  int count;
  double residual;
};

TEST(CommonWorkspaceTests, FixedStructAndMemberPtrAreNullWhenSizing) {
  WorkspaceLayout layout(nullptr);
  TestStatus *const status = layout.fixed_struct<TestStatus>();
  EXPECT_EQ(status, nullptr);
  EXPECT_EQ(layout.total(), aligned(sizeof(TestStatus)));
  EXPECT_EQ(member_ptr(status, &TestStatus::count), nullptr);
  EXPECT_EQ(member_ptr(status, &TestStatus::info), nullptr);
}

TEST(CommonWorkspaceTests, MemberPtrMatchesOffsetof) {
  alignas(kAlign) static std::byte buffer[2 * kAlign];
  WorkspaceLayout layout(buffer);
  (void)layout.fixed<char>(1);
  TestStatus *const status = layout.fixed_struct<TestStatus>();
  ASSERT_EQ(reinterpret_cast<std::byte *>(status), buffer + kAlign);

  const auto at = [&](const void *p) {
    return static_cast<std::size_t>(static_cast<const std::byte *>(p) -
                                    reinterpret_cast<std::byte *>(status));
  };
  int *const info = member_ptr(status, &TestStatus::info); // the array overload decays
  EXPECT_EQ(at(info), offsetof(TestStatus, info));
  EXPECT_EQ(at(member_ptr(status, &TestStatus::count)), offsetof(TestStatus, count));
  EXPECT_EQ(at(member_ptr(status, &TestStatus::residual)), offsetof(TestStatus, residual));
}

} // namespace
} // namespace calaman
