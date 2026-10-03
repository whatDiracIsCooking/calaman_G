// Oracle test for calaman.ladiv (built under src/ladiv/) -- real-arithmetic
// complex division by Smith's algorithm: p + i*q = (a + i*b) / (c + i*d). The
// module exports a BATCHED device driver so the per-thread scalar helper
// (ladiv_scalar, which laln2 will call) can be exercised on a card; the oracle
// batches n divisions and checks each against the reference run on the host.
//
// LAPACKE exposes NO ?ladiv C binding (it is a LAPACK auxiliary routine), so the
// oracle calls the Fortran symbols sladiv_ / dladiv_ directly, reached through
// calaman::lapack_reference, which links LAPACK::LAPACK. Netlib's modern ?ladiv
// uses the Baudin-Smith refinement; this kernel uses the plain Smith branch the
// issue names, so the two agree only where no extreme rescaling is triggered --
// the cases below keep |a|,|b|,|c|,|d| moderate on purpose, where the branch
// selection and arithmetic are identical to the reference's and the results
// match to a few eps.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the inputs on the device
// and runs the kernel, so `ctest -LE gpu` excludes it. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.ladiv;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

// The reference ?ladiv, Fortran-mangled (lcname + '_', this build's default).
// By reference, as Fortran passes everything: a, b, c, d in; p, q out.
extern "C" {
void sladiv_(const float *a, const float *b, const float *c, const float *d, float *p, float *q);
void dladiv_(const double *a, const double *b, const double *c, const double *d, double *p,
             double *q);
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

// Dispatch to the matching-precision reference symbol.
void ref_ladiv(const float *a, const float *b, const float *c, const float *d, float *p, float *q) {
  sladiv_(a, b, c, d, p, q);
}
void ref_ladiv(const double *a, const double *b, const double *c, const double *d, double *p,
               double *q) {
  dladiv_(a, b, c, d, p, q);
}

// (a, b, c, d) quads -- the numerator a+ib over the divisor c+id. Moderate
// magnitudes so the plain Smith branch matches netlib's refined ?ladiv: both
// Smith branches (|d|<|c| and |c|<=|d|), purely real and purely imaginary
// divisors, every sign quadrant, and |c| vs |d| far apart but still in range.
template<typename T>
std::vector<std::array<T, 4>> quads() {
  return {
      {T{1}, T{2}, T{3}, T{4}},        // |d| < |c|? 4 > 3, so |c|<=|d| branch
      {T{1}, T{2}, T{4}, T{3}},        // |d| < |c| branch
      {T{-1}, T{2}, T{3}, T{-4}},      // mixed signs, |c|<=|d|
      {T{5}, T{-3}, T{-2}, T{1}},      // mixed signs, |d| < |c|
      {T{7}, T{0}, T{2}, T{0}},        // real / real
      {T{0}, T{5}, T{0}, T{-3}},       // imag / imag
      {T{3}, T{4}, T{1}, T{0}},        // divide by a purely real number
      {T{3}, T{4}, T{0}, T{2}},        // divide by a purely imaginary number
      {T{1}, T{1}, T{1000}, T{1}},     // |c| >> |d|
      {T{1}, T{1}, T{1}, T{1000}},     // |d| >> |c|
      {T{-6.5}, T{2.25}, T{3.5}, T{-8.75}}, // arbitrary non-integers
      {T{100}, T{-200}, T{-50}, T{25}},     // larger magnitudes, both branches sane
  };
}

// Per-output tolerance: the port differs from the reference only by the rounding
// of the handful of multiplies and divides Smith's algorithm performs, so a few
// eps -- relative, scaled by the result magnitude -- bounds every element.
template<typename T>
T tol(T ref) {
  return T{64} * std::numeric_limits<T>::epsilon() * std::max<T>(T{1}, std::abs(ref));
}

template<typename T>
void matches_reference() {
  auto handle = shared_device();
  const auto quad = quads<T>();
  const std::size_t n = quad.size();

  std::vector<T> a(n), b(n), c(n), d(n);
  for (std::size_t k = 0; k < n; ++k) {
    a[k] = quad[k][0];
    b[k] = quad[k][1];
    c[k] = quad[k][2];
    d[k] = quad[k][3];
  }

  auto d_a = to_device(handle, a);
  auto d_b = to_device(handle, b);
  auto d_c = to_device(handle, c);
  auto d_d = to_device(handle, d);
  DeviceBuffer<T> d_p(n, handle), d_q(n, handle);

  const auto status = calaman::ladiv<T>(handle->stream().get(), n, d_a.data(), d_b.data(),
                                        d_c.data(), d_d.data(), d_p.data(), d_q.data());
  ASSERT_TRUE(status.ok()) << "ladiv returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());

  const auto got_p = from_device(handle, d_p, n);
  const auto got_q = from_device(handle, d_q, n);

  for (std::size_t k = 0; k < n; ++k) {
    T rp{}, rq{};
    ref_ladiv(&a[k], &b[k], &c[k], &d[k], &rp, &rq);
    const auto ctx = [&] {
      return "k=" + std::to_string(k) + " a=" + std::to_string(a[k]) + " b=" + std::to_string(b[k]) +
             " c=" + std::to_string(c[k]) + " d=" + std::to_string(d[k]);
    };
    EXPECT_NEAR(got_p[k], rp, tol<T>(rp)) << "p: " << ctx();
    EXPECT_NEAR(got_q[k], rq, tol<T>(rq)) << "q: " << ctx();
  }
}

// n == 0: ladiv enqueues nothing and returns success. The output buffers are
// pre-filled with a sentinel that must survive untouched.
template<typename T>
void zero_n_is_noop() {
  auto handle = shared_device();
  const std::size_t cap = 4;
  std::vector<T> sentinel(cap, static_cast<T>(-99));
  auto d_a = to_device(handle, sentinel);
  auto d_b = to_device(handle, sentinel);
  auto d_c = to_device(handle, sentinel);
  auto d_d = to_device(handle, sentinel);
  auto d_p = to_device(handle, sentinel);
  auto d_q = to_device(handle, sentinel);

  const auto status = calaman::ladiv<T>(handle->stream().get(), 0, d_a.data(), d_b.data(),
                                        d_c.data(), d_d.data(), d_p.data(), d_q.data());
  ASSERT_TRUE(status.ok()) << "ladiv returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());

  for (const T v : from_device(handle, d_p, cap)) {
    EXPECT_EQ(v, static_cast<T>(-99));
  }
}

} // namespace

TEST(LadivOracleTests, MatchesReferenceFloat) {
  matches_reference<float>();
}

TEST(LadivOracleTests, MatchesReferenceDouble) {
  matches_reference<double>();
}

TEST(LadivOracleTests, ZeroNIsNoop) {
  zero_n_is_noop<float>();
  zero_n_is_noop<double>();
}

} // namespace calaman
