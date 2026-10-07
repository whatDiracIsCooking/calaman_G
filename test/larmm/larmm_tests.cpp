// Oracle test for calaman.larmm -- the batched device lift of ?larmm, the scale
// factor that keeps C - A*B from overflowing. The kernel calls the header-only
// larmm_scalar from "lapack/larmm/larmm.cuh", so this suite is that helper's
// oracle check. LAPACKE exposes no ?larmm binding, so the reference is the
// Fortran slarmm_ / dlarmm_ functions via calaman::lapack_reference.
//
// The result is one of 1, 1/2 or 1/(2*bnorm) -- each a single exact operation
// -- so the device must match the reference bit for bit. Cases pin both
// branches (bnorm <= 1, bnorm > 1) on each side of the overflow threshold
// bignum = DLAMCH('P') / (4 * DLAMCH('S')), plus zeros.
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.larmm;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

// The reference ?larmm, Fortran-mangled REAL / DOUBLE PRECISION functions.
extern "C" {
float slarmm_(const float *anorm, const float *bnorm, const float *cnorm);
double dlarmm_(const double *anorm, const double *bnorm, const double *cnorm);
}

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
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

float ref_larmm(const float a, const float b, const float c) {
  return slarmm_(&a, &b, &c);
}
double ref_larmm(const double a, const double b, const double c) {
  return dlarmm_(&a, &b, &c);
}

// (anorm, bnorm, cnorm) triples, labelled by the branch each pins.
template<typename T>
std::vector<std::pair<std::array<T, 3>, const char *>> triples() {
  const T bignum =
      std::numeric_limits<T>::epsilon() / (T{4} * std::numeric_limits<T>::min());
  return {
      {{T{1}, T{1}, T{1}}, "small: no scaling"},
      {{T{0}, T{0}, T{0}}, "all zero"},
      {{T{0}, T{5}, T{0}}, "anorm = 0, bnorm > 1"},
      {{bignum, T{0}, T{0}}, "bnorm = 0"},
      {{T{2} * bignum, T{0.75}, T{0}}, "bnorm <= 1: product overflows"},
      {{T{0.5} * bignum, T{0.75}, T{0}}, "bnorm <= 1: product fits"},
      {{T{0.5} * bignum, T{1}, T{0.75} * bignum}, "bnorm = 1: cnorm tips it over"},
      {{T{0.25} * bignum, T{1}, T{0.5} * bignum}, "bnorm = 1: cnorm fits"},
      {{bignum, T{8}, T{0}}, "bnorm > 1: product overflows"},
      {{T{0.1} * bignum, T{8}, T{0}}, "bnorm > 1: product fits"},
      {{T{0.1} * bignum, T{4}, T{0.75} * bignum}, "bnorm > 1: cnorm tips it over"},
      {{T{1}, T{2} * bignum, T{0}}, "huge bnorm, unit anorm"},
      {{T{1}, T{1}, bignum}, "cnorm = bignum, product 1"},
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
  DeviceBuffer<T> d_s(n, handle);

  const auto status = calaman::larmm<T>(handle->stream().get(), n, d_a.data(), d_b.data(),
                                        d_c.data(), d_s.data());
  ASSERT_TRUE(status.ok()) << "larmm returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_s, n);

  int scaled = 0;
  for (std::size_t k = 0; k < n; ++k) {
    const T ref = ref_larmm(a[k], b[k], c[k]);
    EXPECT_EQ(got[k], ref) << cases[k].second;
    scaled += ref < T{1} ? 1 : 0;
  }
  // The cases must actually reach the scaling branches, not only return 1.
  EXPECT_GE(scaled, 4);
}

// n == 0 enqueues nothing: the sentinel-filled output survives untouched.
template<typename T>
void zero_n_is_noop() {
  auto handle = shared_device();
  const std::vector<T> sentinel(4, static_cast<T>(-99));
  auto d_in = to_device(handle, sentinel);
  auto d_s = to_device(handle, sentinel);
  const auto status = calaman::larmm<T>(handle->stream().get(), 0, d_in.data(), d_in.data(),
                                        d_in.data(), d_s.data());
  ASSERT_TRUE(status.ok()) << "larmm returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  for (const T v : from_device(handle, d_s, sentinel.size())) {
    EXPECT_EQ(v, static_cast<T>(-99));
  }
}

} // namespace

TEST(LarmmOracleTests, MatchesReferenceFloat) {
  matches_reference<float>();
}

TEST(LarmmOracleTests, MatchesReferenceDouble) {
  matches_reference<double>();
}

TEST(LarmmOracleTests, ZeroNIsNoop) {
  zero_n_is_noop<float>();
  zero_n_is_noop<double>();
}

} // namespace calaman
