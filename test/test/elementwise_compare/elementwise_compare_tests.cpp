// Self-test for the calaman.test.elementwise_compare utility: exact mismatch
// counting, magnitude (max |a-b|), and tolerance-based counting, each for both
// instantiated element types. Values are chosen to be exactly representable, so
// the reported counts and maxima are exact.
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and
// launches the kernels, so the suite is excluded by `ctest -LE gpu`.

#include <gtest/gtest.h>

import std;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.test.elementwise_compare;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;

namespace calaman::test {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

// AbortPolicy and DeviceHandle need no using-declaration here: this suite is
// itself in calaman::test, where test/shared/ puts them. WarpWraps ships
// neither -- see test/shared/README.md.
using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceHandle, DeviceAbort>;

/// @brief Upload `host` to a fresh device buffer on `handle`'s stream
template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const HostBuffer<T> &host,
                          std::size_t n) {
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, host, handle->stream().get());
  return device;
}

/// @brief Two equal ramps of length n differ in zero elements
template<typename T>
void identical_arrays_count_zero() {
  auto handle = std::make_shared<DeviceHandle>(0);
  const std::size_t n = 1024;

  HostBuffer<T> host_a(n);
  HostBuffer<T> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a.data()[i] = static_cast<T>(i);
    host_b.data()[i] = static_cast<T>(i);
  }
  auto d_a = to_device(handle, host_a, n);
  auto d_b = to_device(handle, host_b, n);
  wwr::wwrStreamSynchronize(handle->stream().get());

  EXPECT_EQ(count_mismatches(handle, d_a.data(), d_b.data(), n), 0u);
  EXPECT_EQ(max_abs_diff(handle, d_a.data(), d_b.data(), n), T(0));
}

/// @brief Planting k differing elements is counted as exactly k
template<typename T>
void planted_differences_are_counted() {
  auto handle = std::make_shared<DeviceHandle>(0);
  const std::size_t n = 1000;
  const unsigned int planted = 37;

  HostBuffer<T> host_a(n);
  HostBuffer<T> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a.data()[i] = static_cast<T>(1);
    host_b.data()[i] = static_cast<T>(1);
  }
  // Every 27th element up to `planted` of them, so they are scattered, not
  // packed into one warp/block region.
  unsigned int made = 0;
  for (std::size_t i = 0; i < n && made < planted; i += 27, ++made) {
    host_b.data()[i] = static_cast<T>(2);
  }
  ASSERT_EQ(made, planted);

  auto d_a = to_device(handle, host_a, n);
  auto d_b = to_device(handle, host_b, n);
  wwr::wwrStreamSynchronize(handle->stream().get());

  EXPECT_EQ(count_mismatches(handle, d_a.data(), d_b.data(), n), planted);
}

/// @brief max_abs_diff reports the largest gap, regardless of its sign or place
template<typename T>
void max_abs_diff_finds_worst_gap() {
  auto handle = std::make_shared<DeviceHandle>(0);
  const std::size_t n = 1000;

  HostBuffer<T> host_a(n);
  HostBuffer<T> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a.data()[i] = static_cast<T>(1);
    host_b.data()[i] = static_cast<T>(1);
  }
  host_b.data()[100] = static_cast<T>(1) + static_cast<T>(2); // gap 2
  host_b.data()[900] = static_cast<T>(1) - static_cast<T>(5); // gap 5 (worst), negative side

  auto d_a = to_device(handle, host_a, n);
  auto d_b = to_device(handle, host_b, n);
  wwr::wwrStreamSynchronize(handle->stream().get());

  EXPECT_EQ(max_abs_diff(handle, d_a.data(), d_b.data(), n), T(5));
}

/// @brief count_beyond_tolerance flags only gaps larger than the tolerance
template<typename T>
void tolerance_counts_only_beyond() {
  auto handle = std::make_shared<DeviceHandle>(0);
  const std::size_t n = 1000;

  HostBuffer<T> host_a(n);
  HostBuffer<T> host_b(n);
  for (std::size_t i = 0; i < n; ++i) {
    host_a.data()[i] = static_cast<T>(1);
    host_b.data()[i] = static_cast<T>(1);
  }
  // 10 large gaps (1.0) at i = 0, 40, 80, ...; 5 small gaps (0.25) at i = 3, 43, ...
  unsigned int large = 0;
  for (std::size_t i = 0; i < n && large < 10; i += 40, ++large) {
    host_b.data()[i] = static_cast<T>(1) + static_cast<T>(1);
  }
  unsigned int small = 0;
  for (std::size_t i = 3; i < n && small < 5; i += 40, ++small) {
    host_b.data()[i] = static_cast<T>(1) + static_cast<T>(0.25);
  }
  ASSERT_EQ(large, 10u);
  ASSERT_EQ(small, 5u);

  auto d_a = to_device(handle, host_a, n);
  auto d_b = to_device(handle, host_b, n);
  wwr::wwrStreamSynchronize(handle->stream().get());

  // atol 0.5: only the ten gaps of 1.0 exceed it.
  EXPECT_EQ(count_beyond_tolerance(handle, d_a.data(), d_b.data(), n, T(0.5)), 10u);
  // atol 0 (the default): every one of the fifteen nonzero gaps counts.
  EXPECT_EQ(count_beyond_tolerance(handle, d_a.data(), d_b.data(), n, T(0)), 15u);
}

TEST(ElementwiseCompareTests, IdenticalArraysAreClean) {
  identical_arrays_count_zero<float>();
  identical_arrays_count_zero<double>();
}

TEST(ElementwiseCompareTests, PlantedDifferencesAreCounted) {
  planted_differences_are_counted<float>();
  planted_differences_are_counted<double>();
}

TEST(ElementwiseCompareTests, MaxAbsDiffFindsWorstGap) {
  max_abs_diff_finds_worst_gap<float>();
  max_abs_diff_finds_worst_gap<double>();
}

TEST(ElementwiseCompareTests, CountBeyondToleranceRespectsTolerance) {
  tolerance_counts_only_beyond<float>();
  tolerance_counts_only_beyond<double>();
}

TEST(ElementwiseCompareTests, EmptyRangeIsZero) {
  auto handle = std::make_shared<DeviceHandle>(0);
  EXPECT_EQ(count_mismatches<float>(handle, nullptr, nullptr, 0), 0u);
  EXPECT_EQ(count_beyond_tolerance<double>(handle, nullptr, nullptr, 0, 1.0), 0u);
  EXPECT_EQ(max_abs_diff<double>(handle, nullptr, nullptr, 0), 0.0);
}

} // namespace
} // namespace calaman::test
