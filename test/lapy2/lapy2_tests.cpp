// Oracle test for calaman.lapy2 -- the batched device lift of ?lapy2,
// sqrt(x^2 + y^2) without spurious overflow and with NaN propagation. The kernel
// calls the shared lapy2_scalar from "lapack/sym2x2/sym2x2.cuh", so this suite
// is that helper's oracle check. LAPACKE exposes no ?lapy2 binding, so the
// reference is the Fortran slapy2_ / dlapy2_ functions via
// calaman::lapack_reference.
//
// Finite results must agree to kTolFactor * eps relative; an infinite reference
// must be matched exactly; a NaN reference must be matched by a NaN. Cases
// cover zeros, y = 0 / x = 0, magnitudes near the overflow threshold (where the
// naive formula overflows), near the underflow threshold, infinities and NaN
// in either or both arguments.
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lapy2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// The reference ?lapy2, Fortran-mangled REAL / DOUBLE PRECISION functions.
extern "C" {
float slapy2_(const float *x, const float *y);
double dlapy2_(const double *x, const double *y);
}

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;
using test::kTolFactor;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const std::vector<T> &host) {
  const std::size_t n = host.size();
  HostBuffer<T> staging(n == 0 ? 1 : n);
  for (std::size_t i = 0; i < n; ++i) {
    staging.data()[i] = host[i];
  }
  DeviceBuffer<T> device(n == 0 ? 1 : n, handle);
  wwr::extension::copy(device, staging, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return device;
}

template<typename T>
std::vector<T> from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<T> &device,
                           std::size_t n) {
  HostBuffer<T> host(n == 0 ? 1 : n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  std::vector<T> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = host.data()[i];
  }
  return out;
}

float ref_lapy2(const float x, const float y) {
  return slapy2_(&x, &y);
}
double ref_lapy2(const double x, const double y) {
  return dlapy2_(&x, &y);
}

// (x, y) pairs, labelled by the branch or edge each pins.
template<typename T>
std::vector<std::pair<std::array<T, 2>, const char *>> pairs() {
  const T big = std::numeric_limits<T>::max();
  const T tiny = std::numeric_limits<T>::min(); // smallest normal
  const T inf = std::numeric_limits<T>::infinity();
  const T nan = std::numeric_limits<T>::quiet_NaN();
  return {
      {{T{3}, T{4}}, "3-4-5"},
      {{T{-3}, T{4}}, "negative x"},
      {{T{4}, T{-3}}, "negative y, x larger"},
      {{T{0}, T{0}}, "both zero"},
      {{T{0}, T{5}}, "x = 0"},
      {{T{-5}, T{0}}, "y = 0"},
      {{T{1}, T{1}}, "x = y"},
      {{T{1}, T{1e-6}}, "y negligible"},
      {{T{0.75} * big, T{0.5} * big}, "near overflow: naive x*x overflows"},
      {{T{0.5} * big, T{0.5} * big}, "near overflow, x = y"},
      {{big, T{1}}, "max, 1"},
      {{T{3} * tiny, T{4} * tiny}, "near underflow: naive x*x underflows"},
      {{inf, T{1}}, "x = +inf"},
      {{T{1}, -inf}, "y = -inf"},
      {{inf, inf}, "both inf"},
      {{nan, T{1}}, "x NaN"},
      {{T{1}, nan}, "y NaN"},
      {{nan, nan}, "both NaN"},
      {{nan, inf}, "NaN beats inf (x)"},
      {{inf, nan}, "NaN beats inf (y)"},
  };
}

template<typename T>
void matches_reference() {
  auto handle = shared_device();
  const auto cases = pairs<T>();
  const std::size_t n = cases.size();
  std::vector<T> x(n), y(n);
  for (std::size_t k = 0; k < n; ++k) {
    x[k] = cases[k].first[0];
    y[k] = cases[k].first[1];
  }
  auto d_x = to_device(handle, x);
  auto d_y = to_device(handle, y);
  DeviceBuffer<T> d_r(n, handle);

  const auto status =
      calaman::lapy2<T>(handle->stream().get(), n, d_x.data(), d_y.data(), d_r.data());
  ASSERT_TRUE(status.ok()) << "lapy2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_r, n);

  for (std::size_t k = 0; k < n; ++k) {
    const T ref = ref_lapy2(x[k], y[k]);
    const char *ctx = cases[k].second;
    if (std::isnan(ref)) {
      EXPECT_TRUE(std::isnan(got[k])) << ctx << ": got " << got[k];
    } else if (std::isinf(ref)) {
      EXPECT_EQ(got[k], ref) << ctx;
    } else {
      EXPECT_TRUE(std::isfinite(got[k])) << ctx << ": got " << got[k];
      EXPECT_NEAR(got[k], ref, kTolFactor<T> * eps<T>() * ref) << ctx;
    }
  }
}

// n == 0 enqueues nothing: the sentinel-filled output survives untouched.
template<typename T>
void zero_n_is_noop() {
  auto handle = shared_device();
  const std::vector<T> sentinel(4, static_cast<T>(-99));
  auto d_in = to_device(handle, sentinel);
  auto d_r = to_device(handle, sentinel);
  const auto status =
      calaman::lapy2<T>(handle->stream().get(), 0, d_in.data(), d_in.data(), d_r.data());
  ASSERT_TRUE(status.ok()) << "lapy2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  for (const T v : from_device(handle, d_r, sentinel.size())) {
    EXPECT_EQ(v, static_cast<T>(-99));
  }
}

} // namespace

TEST(Lapy2OracleTests, MatchesReferenceFloat) {
  matches_reference<float>();
}

TEST(Lapy2OracleTests, MatchesReferenceDouble) {
  matches_reference<double>();
}

TEST(Lapy2OracleTests, ZeroNIsNoop) {
  zero_n_is_noop<float>();
  zero_n_is_noop<double>();
}

} // namespace calaman
