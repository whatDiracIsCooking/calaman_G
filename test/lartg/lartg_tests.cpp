// Oracle test for calaman.lartg (built under src/lartg/) -- the batched device
// lift of LAPACK's ?lartg: n plane rotations (c, s, r) from n pairs (f, g).
//
// LAPACKE exposes NO ?lartg C binding (only lartgp / lartgs, which carry a
// different sign convention), so the oracle calls the Fortran symbols slartg_ /
// dlartg_ directly -- the exact reference routine the kernel ports -- reached
// through calaman::lapack_reference, which links LAPACK::LAPACK. The device
// batches one call per element; the reference is that same routine run on the
// host per element, over the identical inputs.
//
// The port reproduces the reference formula term for term, so the two agree to
// within the rounding of a single sqrt and divide: the comparison is a tight
// relative tolerance (a few eps), NOT a factorization-style norm bound. Cases
// cover both signs, the g==0 and f==0 special branches (including (0,0)), the
// unscaled fast path, and -- the point of the safe-scaling algorithm -- inputs
// whose magnitudes fall below rtmin or above rtmax and drive the scaled
// fallback, plus the n==0 no-op.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the inputs on the device
// and runs the kernel, so `ctest -LE gpu` excludes it. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lartg;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

// The reference ?lartg, Fortran-mangled (lcname + '_', this build's default).
// By reference, as Fortran passes everything: f, g in; c, s, r out.
extern "C" {
void slartg_(const float *f, const float *g, float *c, float *s, float *r);
void dlartg_(const double *f, const double *g, double *c, double *s, double *r);
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
void ref_lartg(const float *f, const float *g, float *c, float *s, float *r) {
  slartg_(f, g, c, s, r);
}
void ref_lartg(const double *f, const double *g, double *c, double *s, double *r) {
  dlartg_(f, g, c, s, r);
}

// The (f, g) pairs. The moderate ones are shape-identical across precisions; the
// extreme ones differ, because the magnitudes that fall outside [rtmin, rtmax]
// -- and so exercise the scaled fallback -- are precision-specific (rtmax is
// ~6.5e18 for float, ~1e154 for double). ext<T>::extremes() supplies those.
template<typename T>
struct ext;

template<>
struct ext<float> {
  static std::vector<std::pair<float, float>> extremes() {
    return {
        {1e-25f, 2e-25f},  // both below rtmin -> scaled
        {-1e-22f, 4e-24f}, // below rtmin, negative f
        {1e25f, 3e24f},    // above rtmax -> scaled
        {-1e20f, 1e19f},   // above rtmax, negative f
        {6e18f, 7e18f},    // straddle rtmax (one in, one out)
        {1e-30f, 1e24f},   // tiny and huge together
    };
  }
};

template<>
struct ext<double> {
  static std::vector<std::pair<double, double>> extremes() {
    return {
        {1e-170, 2e-175},  // both below rtmin -> scaled
        {-1e-160, 4e-200}, // below rtmin, negative f
        {1e200, 3e199},    // above rtmax -> scaled
        {-1e180, 1e170},   // above rtmax, negative f
        {1e154, 2e154},    // straddle rtmax
        {1e-200, 1e180},   // tiny and huge together
    };
  }
};

template<typename T>
std::vector<std::pair<T, T>> pairs() {
  std::vector<std::pair<T, T>> p = {
      {T{3}, T{4}},      {T{-3}, T{4}},     {T{3}, T{-4}},   {T{-3}, T{-4}}, // all sign quadrants
      {T{5}, T{0}},      {T{-5}, T{0}},                                     // g == 0 branch
      {T{0}, T{7}},      {T{0}, T{-7}},                                     // f == 0 branch
      {T{0}, T{0}},                                                         // both zero
      {T{1}, T{1}},      {T{1}, T{-1}},                                     // equal magnitude
      {T{1}, T{1000}},   {T{1000}, T{1}},                                   // one dominates
      {T{123.5}, T{-78.25}},                                                // arbitrary
  };
  for (const auto &e : ext<T>::extremes()) {
    p.push_back(e);
  }
  return p;
}

// Per-output tolerance: the port differs from the reference only by the rounding
// of one sqrt and one divide, so a few eps -- relative, since r spans the whole
// exponent range -- bounds every element. c and s are bounded by 1, so the
// max(1, |ref|) factor leaves their tolerance at ~eps.
template<typename T>
T tol(T ref) {
  return T{32} * std::numeric_limits<T>::epsilon() * std::max<T>(T{1}, std::abs(ref));
}

template<typename T>
void matches_reference() {
  auto handle = shared_device();
  const auto p = pairs<T>();
  const std::size_t n = p.size();

  std::vector<T> f(n), g(n);
  for (std::size_t k = 0; k < n; ++k) {
    f[k] = p[k].first;
    g[k] = p[k].second;
  }

  auto d_f = to_device(handle, f);
  auto d_g = to_device(handle, g);
  DeviceBuffer<T> d_c(n, handle), d_s(n, handle), d_r(n, handle);

  const auto status =
      calaman::lartg<T>(handle->stream().get(), n, d_f.data(), d_g.data(), d_c.data(), d_s.data(),
                        d_r.data());
  ASSERT_TRUE(status.ok()) << "lartg returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());

  const auto got_c = from_device(handle, d_c, n);
  const auto got_s = from_device(handle, d_s, n);
  const auto got_r = from_device(handle, d_r, n);

  for (std::size_t k = 0; k < n; ++k) {
    T rc{}, rs{}, rr{};
    ref_lartg(&f[k], &g[k], &rc, &rs, &rr);
    const auto ctx = [&] { return "k=" + std::to_string(k) + " f=" + std::to_string(f[k]) +
                                  " g=" + std::to_string(g[k]); };
    EXPECT_NEAR(got_c[k], rc, tol<T>(rc)) << "c: " << ctx();
    EXPECT_NEAR(got_s[k], rs, tol<T>(rs)) << "s: " << ctx();
    EXPECT_NEAR(got_r[k], rr, tol<T>(rr)) << "r: " << ctx();
    // c >= 0 is part of the ?lartg contract -- check the device honours it.
    EXPECT_GE(got_c[k], T{0}) << "c sign: " << ctx();
  }
}

// n == 0: lartg enqueues nothing and returns success. The output buffers are
// pre-filled with a sentinel that must survive untouched.
template<typename T>
void zero_n_is_noop() {
  auto handle = shared_device();
  const std::size_t cap = 4;
  std::vector<T> sentinel(cap, static_cast<T>(-99));
  auto d_f = to_device(handle, sentinel);
  auto d_g = to_device(handle, sentinel);
  auto d_c = to_device(handle, sentinel);
  auto d_s = to_device(handle, sentinel);
  auto d_r = to_device(handle, sentinel);

  const auto status =
      calaman::lartg<T>(handle->stream().get(), 0, d_f.data(), d_g.data(), d_c.data(), d_s.data(),
                        d_r.data());
  ASSERT_TRUE(status.ok()) << "lartg returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());

  for (const T v : from_device(handle, d_c, cap)) {
    EXPECT_EQ(v, static_cast<T>(-99));
  }
}

} // namespace

TEST(LartgOracleTests, MatchesReferenceFloat) {
  matches_reference<float>();
}

TEST(LartgOracleTests, MatchesReferenceDouble) {
  matches_reference<double>();
}

TEST(LartgOracleTests, ZeroNIsNoop) {
  zero_n_is_noop<float>();
  zero_n_is_noop<double>();
}

} // namespace calaman
