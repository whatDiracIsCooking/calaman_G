// Oracle test for calaman.lae2 -- the batched device lift of ?lae2, the
// eigenvalues (rt1, rt2) of the symmetric 2x2 [a b; b c]. The kernel calls the
// shared lae2_scalar from "lapack/sym2x2/sym2x2.cuh", so this suite is that
// helper's oracle check. LAPACKE exposes no ?lae2 binding (an auxiliary
// routine), so the reference is the Fortran slae2_ / dlae2_ via
// calaman::lapack_reference.
//
// Cases cover b = 0, a = c, |b| dominating, a + c = 0 (the sm == 0 branch), the
// zero matrix, both signs of sm, and large / tiny scales. The port is verbatim,
// so the device and reference take identical branches; only FMA contraction
// separates them, bounded by kTolFactor * eps scaled by max(|a|, |b|, |c|).
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lae2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// The reference ?lae2, Fortran-mangled; everything by reference.
extern "C" {
void slae2_(const float *a, const float *b, const float *c, float *rt1, float *rt2);
void dlae2_(const double *a, const double *b, const double *c, double *rt1, double *rt2);
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

void ref_lae2(const float *a, const float *b, const float *c, float *rt1, float *rt2) {
  slae2_(a, b, c, rt1, rt2);
}
void ref_lae2(const double *a, const double *b, const double *c, double *rt1, double *rt2) {
  dlae2_(a, b, c, rt1, rt2);
}

// (a, b, c) triples, labelled by the branch or edge each pins.
template<typename T>
std::vector<std::pair<std::array<T, 3>, const char *>> triples() {
  return {
      {{T{1}, T{2}, T{3}}, "generic, sm > 0, adf < ab"},
      {{T{3}, T{0.5}, T{1}}, "generic, adf > ab"},
      {{T{-4}, T{1}, T{2}}, "sm < 0"},
      {{T{2}, T{-5}, T{-7}}, "negative b, sm < 0"},
      {{T{2}, T{0}, T{5}}, "b = 0, a < c"},
      {{T{5}, T{0}, T{2}}, "b = 0, a > c"},
      {{T{-3}, T{0}, T{-3}}, "b = 0 and a = c (adf = ab = 0)"},
      {{T{2}, T{1}, T{2}}, "a = c"},
      {{T{-1}, T{3}, T{-1}}, "a = c, negative"},
      {{T{0}, T{1}, T{0}}, "a = c = 0, pure off-diagonal (sm = 0)"},
      {{T{1}, T{2}, T{-1}}, "a + c = 0 (sm = 0)"},
      {{T{1}, T{1e6}, T{2}}, "|b| dominating"},
      {{T{1e-3}, T{1e3}, T{-1e-3}}, "|b| dominating, sm = 0"},
      {{T{0.5}, T{0.25}, T{0.75}}, "adf = ab"},
      {{T{0}, T{0}, T{0}}, "zero matrix"},
      {{T{1e8}, T{3e8}, T{-2e8}}, "large scale"},
      {{T{1e-20}, T{3e-20}, T{2e-20}}, "tiny scale"},
      {{T{-6.5}, T{2.25}, T{3.5}}, "non-integers"},
  };
}

template<typename T>
void matches_reference() {
  auto handle = shared_device();
  const auto cases = triples<T>();
  const std::size_t n = cases.size();
  std::vector<T> a(n), b(n), c(n);
  for (std::size_t k = 0; k < n; ++k) {
    a[k] = cases[k].first[0];
    b[k] = cases[k].first[1];
    c[k] = cases[k].first[2];
  }
  auto d_a = to_device(handle, a);
  auto d_b = to_device(handle, b);
  auto d_c = to_device(handle, c);
  DeviceBuffer<T> d_rt1(n, handle), d_rt2(n, handle);

  const auto status = calaman::lae2<T>(handle->stream().get(), n, d_a.data(), d_b.data(),
                                       d_c.data(), d_rt1.data(), d_rt2.data());
  ASSERT_TRUE(status.ok()) << "lae2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got_rt1 = from_device(handle, d_rt1, n);
  const auto got_rt2 = from_device(handle, d_rt2, n);

  for (std::size_t k = 0; k < n; ++k) {
    T rt1{}, rt2{};
    ref_lae2(&a[k], &b[k], &c[k], &rt1, &rt2);
    const T scale = std::max({std::abs(a[k]), std::abs(b[k]), std::abs(c[k])});
    const T tol = kTolFactor<T> * eps<T>() * scale;
    EXPECT_NEAR(got_rt1[k], rt1, tol) << "rt1: " << cases[k].second;
    EXPECT_NEAR(got_rt2[k], rt2, tol) << "rt2: " << cases[k].second;
  }
}

// n == 0 enqueues nothing: the sentinel-filled outputs survive untouched.
template<typename T>
void zero_n_is_noop() {
  auto handle = shared_device();
  const std::vector<T> sentinel(4, static_cast<T>(-99));
  auto d_in = to_device(handle, sentinel);
  auto d_rt1 = to_device(handle, sentinel);
  auto d_rt2 = to_device(handle, sentinel);
  const auto status = calaman::lae2<T>(handle->stream().get(), 0, d_in.data(), d_in.data(),
                                       d_in.data(), d_rt1.data(), d_rt2.data());
  ASSERT_TRUE(status.ok()) << "lae2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  for (const T v : from_device(handle, d_rt1, sentinel.size())) {
    EXPECT_EQ(v, static_cast<T>(-99));
  }
}

} // namespace

TEST(Lae2OracleTests, MatchesReferenceFloat) {
  matches_reference<float>();
}

TEST(Lae2OracleTests, MatchesReferenceDouble) {
  matches_reference<double>();
}

TEST(Lae2OracleTests, ZeroNIsNoop) {
  zero_n_is_noop<float>();
  zero_n_is_noop<double>();
}

} // namespace calaman
