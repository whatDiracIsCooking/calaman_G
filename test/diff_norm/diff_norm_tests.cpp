// Oracle test for calaman.diff_norm: ||y - x|| in each Norm must agree with the
// reference CBLAS -- cblas_?asum for l1, cblas_?nrm2 for l2, cblas_i?amax (plus a
// read of that element) for inf -- computed in the SAME precision on the host.
//
// Inputs are small exactly-representable integer ramps of mixed sign, so y - x is
// exact and float and double behave identically; the only difference between GPU
// and reference is summation order, absorbed by a relative tolerance. inf is
// exact in principle (both pick the same max magnitude), but shares the tolerance.
//
// diff_norm OVERWRITES y with y - x, so every case uploads a fresh y. The handle
// is left in its default (host) pointer mode, which is what diff_norm requires:
// the -1 scalar and the result are host pointers, and the reduction blocks.
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and runs
// the kernels, so the suite is excluded by `ctest -LE gpu`. It is built only when
// calaman::lapack_reference exists (docs/architecture.md §3); its CMakeLists.txt
// returns early otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <cblas.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.diff_norm;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

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
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

/// @brief Upload `host` to a fresh device buffer on `handle`'s stream
template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const HostBuffer<T> &host,
                          std::size_t n) {
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, host, handle->stream().get());
  return device;
}

// The reference oracle: reference CBLAS, dispatched by type, same precision as the
// device path. cblas_i?amax returns a 0-based index into the difference vector.
float ref_asum(int n, const float *d) {
  return cblas_sasum(n, d, 1);
}
double ref_asum(int n, const double *d) {
  return cblas_dasum(n, d, 1);
}
float ref_nrm2(int n, const float *d) {
  return cblas_snrm2(n, d, 1);
}
double ref_nrm2(int n, const double *d) {
  return cblas_dnrm2(n, d, 1);
}
std::size_t ref_iamax(int n, const float *d) {
  return cblas_isamax(n, d, 1);
}
std::size_t ref_iamax(int n, const double *d) {
  return cblas_idamax(n, d, 1);
}

/// @brief ||diff|| in norm @p which, from the reference CBLAS
template<typename T>
T ref_norm(Norm which, const std::vector<T> &diff) {
  const int n = static_cast<int>(diff.size());
  if (which == Norm::l1) {
    return ref_asum(n, diff.data());
  }
  if (which == Norm::inf) {
    if (n == 0) {
      return T{0};
    }
    const T v = diff[ref_iamax(n, diff.data())];
    return v < T{0} ? -v : v;
  }
  return ref_nrm2(n, diff.data()); // l2
}

/// @brief A relative tolerance generous enough for cuBLAS-vs-CBLAS summation
///        order, tight enough to catch a wrong norm or a missing subtraction
template<typename T>
T norm_tol(T ref) {
  const T rel = std::is_same_v<T, float> ? T(1e-4) : T(1e-11);
  const T atol = std::is_same_v<T, float> ? T(1e-4) : T(1e-11);
  const T a = ref < T{0} ? -ref : ref;
  return rel * a + atol;
}

// The input ramps: mixed sign so l1 and inf are non-trivial, small so y - x is
// exact and the l1/l2 accumulations stay well inside the mantissa.
template<typename T>
T x_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 13) - 6);
}
template<typename T>
T y_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 17) - 8);
}

/// @brief diff_norm on the device must match the reference for one (norm, n, inc)
///
/// The device vectors are the strided elements of length-(n*inc) buffers, so the
/// reference difference is gathered over the same stride. n >= 1 here; the empty
/// case is its own test because diff_norm leaves `result` untouched.
template<Norm which, typename T>
void expect_matches_reference(int n, int inc) {
  ASSERT_GE(n, 1);
  ASSERT_GE(inc, 1);
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(inc);

  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  // One stream for the uploads and the BLAS work, so the axpy sees the copies.
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_x(total);
  HostBuffer<T> host_y(total);
  for (std::size_t i = 0; i < total; ++i) {
    host_x.data()[i] = x_at<T>(i);
    host_y.data()[i] = y_at<T>(i);
  }

  std::vector<T> diff(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) {
    const std::size_t i = static_cast<std::size_t>(k) * static_cast<std::size_t>(inc);
    diff[static_cast<std::size_t>(k)] = host_y.data()[i] - host_x.data()[i];
  }
  const T ref = ref_norm<T>(which, diff);

  auto d_x = to_device(handle, host_x, total);
  auto d_y = to_device(handle, host_y, total);

  T got = static_cast<T>(-12345);
  const auto status = diff_norm<which>(blas, n, d_x.data(), inc, d_y.data(), inc, &got);
  wwr::wwrblasDestroy(blas);

  EXPECT_TRUE(status.ok()) << "norm=" << static_cast<int>(which) << " n=" << n << " inc=" << inc
                           << " status=" << status.name();
  EXPECT_NEAR(got, ref, norm_tol(ref))
      << "norm=" << static_cast<int>(which) << " n=" << n << " inc=" << inc;
}

template<typename T>
void run_all_norms(int n, int inc) {
  expect_matches_reference<Norm::l1, T>(n, inc);
  expect_matches_reference<Norm::l2, T>(n, inc);
  expect_matches_reference<Norm::inf, T>(n, inc);
}

/// @brief Same oracle check, but with the handle in DEVICE pointer mode
///
/// The handle is switched to WWRBLAS_POINTER_MODE_DEVICE and `result` is a device
/// pointer, so diff_norm must force host mode for its axpy, restore device mode,
/// and land the norm in device memory -- for ell_inf that is the calaman.set_element
/// path (iamax's device index -> set_element_abs). The reference is identical; only
/// where the result lives changes, so the device scalar is read back to compare.
template<Norm which, typename T>
void expect_matches_reference_device(int n, int inc) {
  ASSERT_GE(n, 1);
  ASSERT_GE(inc, 1);
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(inc);

  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetPointerMode(blas, wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_x(total);
  HostBuffer<T> host_y(total);
  for (std::size_t i = 0; i < total; ++i) {
    host_x.data()[i] = x_at<T>(i);
    host_y.data()[i] = y_at<T>(i);
  }
  std::vector<T> diff(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) {
    const std::size_t i = static_cast<std::size_t>(k) * static_cast<std::size_t>(inc);
    diff[static_cast<std::size_t>(k)] = host_y.data()[i] - host_x.data()[i];
  }
  const T ref = ref_norm<T>(which, diff);

  auto d_x = to_device(handle, host_x, total);
  auto d_y = to_device(handle, host_y, total);
  DeviceBuffer<T> d_result(1, handle);

  const auto status =
      diff_norm<which>(blas, n, d_x.data(), inc, d_y.data(), inc, d_result.data());
  EXPECT_TRUE(status.ok()) << "device norm=" << static_cast<int>(which) << " n=" << n
                           << " inc=" << inc << " status=" << status.name();

  // The norm was written to device memory; read it back to compare.
  HostBuffer<T> host_result(1);
  wwr::extension::copy(host_result, d_result, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  wwr::wwrblasDestroy(blas);

  EXPECT_NEAR(host_result.data()[0], ref, norm_tol(ref))
      << "device norm=" << static_cast<int>(which) << " n=" << n << " inc=" << inc;
}

template<typename T>
void run_all_norms_device(int n, int inc) {
  expect_matches_reference_device<Norm::l1, T>(n, inc);
  expect_matches_reference_device<Norm::l2, T>(n, inc);
  expect_matches_reference_device<Norm::inf, T>(n, inc);
}

TEST(DiffNormOracleTests, MatchesReferenceFloat) {
  run_all_norms<float>(1, 1);
  run_all_norms<float>(7, 1);
  run_all_norms<float>(1000, 1);
  run_all_norms<float>(500, 2); // strided
}

TEST(DiffNormOracleTests, MatchesReferenceDouble) {
  run_all_norms<double>(1, 1);
  run_all_norms<double>(64, 1);
  run_all_norms<double>(1000, 1);
  run_all_norms<double>(500, 3); // strided
}

TEST(DiffNormOracleTests, DevicePointerModeFloat) {
  run_all_norms_device<float>(1, 1);
  run_all_norms_device<float>(7, 1);
  run_all_norms_device<float>(1000, 1);
  run_all_norms_device<float>(500, 2); // strided
}

TEST(DiffNormOracleTests, DevicePointerModeDouble) {
  run_all_norms_device<double>(1, 1);
  run_all_norms_device<double>(64, 1);
  run_all_norms_device<double>(1000, 1);
  run_all_norms_device<double>(500, 3); // strided
}

TEST(DiffNormOracleTests, EmptyIsNoopSuccess) {
  // n == 0: diff_norm enqueues nothing, returns success, and leaves result
  // untouched -- so the sentinel must survive.
  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<double> host_x(1);
  HostBuffer<double> host_y(1);
  host_x.data()[0] = 1.0;
  host_y.data()[0] = 2.0;
  auto d_x = to_device(handle, host_x, 1);
  auto d_y = to_device(handle, host_y, 1);

  double got = -12345.0;
  const auto status = diff_norm<Norm::l2>(blas, 0, d_x.data(), 1, d_y.data(), 1, &got);
  wwr::wwrblasDestroy(blas);

  EXPECT_TRUE(status.ok());
  EXPECT_DOUBLE_EQ(got, -12345.0); // untouched
}

} // namespace
} // namespace calaman
