// Oracle test for calaman.lacpy: the GPU copy must agree with the reference
// LAPACK's LAPACKE_?lacpy bit for bit -- lacpy is an exact copy, so the check is
// exact (count_mismatches == 0), not a tolerance. Every case compares the WHOLE
// B buffer (ldb*n), which also proves the region outside the copy is untouched.
//
// A and B are seeded with distinct, exactly-representable integer ramps, so no
// rounding enters and float and double behave identically. Padded leading
// dimensions (lda, ldb > m) are exercised to catch column-stride bugs.
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and
// launches the kernel, so the suite is excluded by `ctest -LE gpu`. The suite
// is built only when calaman::lapack_reference exists (docs/architecture.md §3);
// its CMakeLists.txt returns early otherwise, so its absence is a missing tier,
// not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lacpy;
import calaman.test.elementwise_compare;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using calaman::test::count_mismatches;

// AbortPolicy and DeviceHandle are this repo's own, under test/shared/:
// WarpWraps ships neither, so a consumer names the policy it wants and supplies
// a concrete device_handle. See test/shared/README.md.
using test::AbortPolicy;
using test::DeviceHandle;

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

/// @brief The reference oracle: LAPACKE_?lacpy, column-major, dispatched by type
void ref_lacpy(char uplo, lapack_int m, lapack_int n, const float *a, lapack_int lda, float *b,
               lapack_int ldb) {
  LAPACKE_slacpy(LAPACK_COL_MAJOR, uplo, m, n, a, lda, b, ldb);
}
void ref_lacpy(char uplo, lapack_int m, lapack_int n, const double *a, lapack_int lda, double *b,
               lapack_int ldb) {
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, uplo, m, n, a, lda, b, ldb);
}

/// @brief The LAPACKE uplo char that matches a calaman copy_region
char uplo_char(copy_region region) {
  switch (region) {
  case copy_region::upper:
    return 'U';
  case copy_region::lower:
    return 'L';
  default:
    return 'A'; // anything not 'U'/'L' means "all" to LAPACKE
  }
}

/// @brief calaman::lacpy must match the reference over the whole B buffer
///
/// A is filled column-major (lda*n), B with a distinct sentinel ramp (ldb*n) so
/// an untouched element differs from any copied one. The oracle runs on a host
/// copy of that B; the GPU runs on the same initial B; the two results are
/// compared on the device, exactly.
template<typename T>
void expect_matches_reference(copy_region region, std::size_t m, std::size_t n, std::size_t lda,
                              std::size_t ldb) {
  ASSERT_GE(lda, m);
  ASSERT_GE(ldb, m);
  auto handle = std::make_shared<DeviceHandle>(0);

  const std::size_t a_size = lda * n;
  const std::size_t b_size = ldb * n;

  HostBuffer<T> host_a(a_size);
  HostBuffer<T> host_b_init(b_size);
  HostBuffer<T> host_b_ref(b_size);
  for (std::size_t i = 0; i < a_size; ++i) {
    host_a.data()[i] = static_cast<T>(1 + (i % 97));
  }
  for (std::size_t i = 0; i < b_size; ++i) {
    const T sentinel = -static_cast<T>(1 + (i % 89));
    host_b_init.data()[i] = sentinel;
    host_b_ref.data()[i] = sentinel;
  }

  // The oracle: copy into the reference B in place.
  ref_lacpy(uplo_char(region), static_cast<lapack_int>(m), static_cast<lapack_int>(n),
            host_a.data(), static_cast<lapack_int>(lda), host_b_ref.data(),
            static_cast<lapack_int>(ldb));

  auto d_a = to_device(handle, host_a, a_size);
  auto d_b = to_device(handle, host_b_init, b_size);
  auto d_ref = to_device(handle, host_b_ref, b_size);
  wwr::wwrStreamSynchronize(handle->stream().get());

  // lacpy takes the stream, not the handle: it allocates nothing, so a stream is
  // its whole requirement (src/lacpy/interface.cppm).
  lacpy(handle->stream().get(), region, m, n, d_a.data(), lda, d_b.data(), ldb);

  EXPECT_EQ(count_mismatches(handle, d_b.data(), d_ref.data(), b_size), 0u)
      << "region=" << static_cast<int>(region) << " m=" << m << " n=" << n << " lda=" << lda
      << " ldb=" << ldb;
}

template<typename T>
void run_all_regions() {
  // Contiguous (lda == ldb == m) and padded (lda, ldb > m), square and both
  // rectangular orientations.
  expect_matches_reference<T>(copy_region::full, 5, 4, 5, 5);
  expect_matches_reference<T>(copy_region::full, 5, 4, 8, 7);
  expect_matches_reference<T>(copy_region::upper, 6, 6, 9, 8);
  expect_matches_reference<T>(copy_region::lower, 6, 6, 9, 8);
  expect_matches_reference<T>(copy_region::upper, 4, 7, 6, 5); // wide trapezoid
  expect_matches_reference<T>(copy_region::lower, 7, 4, 9, 8); // tall trapezoid
}

TEST(LacpyOracleTests, MatchesReferenceFloat) {
  run_all_regions<float>();
}

TEST(LacpyOracleTests, MatchesReferenceDouble) {
  run_all_regions<double>();
}

TEST(LacpyOracleTests, EmptyExtentIsNoop) {
  // m == 0 or n == 0 copies nothing; both the oracle and the GPU leave B as the
  // sentinel, so they still agree.
  expect_matches_reference<double>(copy_region::full, 0, 4, 5, 5);
  expect_matches_reference<double>(copy_region::upper, 6, 0, 9, 8);
}

} // namespace
} // namespace calaman
