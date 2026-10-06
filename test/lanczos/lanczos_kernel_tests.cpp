// Kernel-level suite for calaman.lanczos's device library (lanczos.cu), driven
// through the launchers in lanczos_bridge.h against host-computed expectations:
//
//   * step: T(j,j) / T(j+1,j) / T(j,j+1) carry alpha_j / beta_j, the rest of
//     column/row j past the band is zeroed, and v_{j+1} = w / beta_j;
//   * breakdown: beta_j under eps * ||T_j|| (relative, so a tiny-but-nonzero
//     beta against a large T trips) or NaN sets the flag and the FIRST step,
//     writes a zero coupling and leaves w undivided -- no inf/nan reaches v;
//   * arrowhead: columns/rows 0..k-1 written in full, (k,k) and the trailing
//     block untouched; then steps k..ncv-1 leave no stale entry anywhere in T.
//
// REQUIRES_GPU: every case launches kernels. The T writes are copies, so they
// are compared exactly; the normalise is a division, compared to 2 ulps.

#include "lanczos/lanczos_bridge.h"

#include <gtest/gtest.h>

#include <cstddef>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using device::LanczosStatus;
using test::AbortPolicy;
using test::DeviceHandle;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

/// A sentinel no kernel writes: an entry still holding it was never touched.
template<typename T>
constexpr T kStale = T(-12345.5);

template<typename T>
DeviceBuffer<T> to_device(const std::shared_ptr<DeviceHandle> &handle, const std::vector<T> &host) {
  HostBuffer<T> staging(host.size());
  std::copy(host.begin(), host.end(), staging.data());
  DeviceBuffer<T> device(host.size(), handle);
  (void)wwr::extension::copy(device, staging, handle->stream().get());
  (void)wwr::wwrStreamSynchronize(handle->stream().get());
  return device;
}

template<typename T>
std::vector<T> from_device(const std::shared_ptr<DeviceHandle> &handle,
                           const DeviceBuffer<T> &device, std::size_t n) {
  HostBuffer<T> host(n);
  (void)wwr::extension::copy(host, device, handle->stream().get());
  (void)wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

template<typename T>
std::vector<T> random_vector(std::size_t n, std::uint32_t seed, T lo = T(-1), T hi = T(1)) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(lo, hi);
  std::vector<T> v(n);
  for (T &x : v) {
    x = dist(gen);
  }
  return v;
}

LanczosStatus read_status(const std::shared_ptr<DeviceHandle> &handle,
                          const DeviceBuffer<LanczosStatus> &d_status) {
  return from_device(handle, d_status, 1)[0];
}

DeviceBuffer<LanczosStatus> fresh_status(const std::shared_ptr<DeviceHandle> &handle) {
  DeviceBuffer<LanczosStatus> d_status =
      to_device(handle, std::vector<LanczosStatus>{{7, 7, 7}});
  device::lanczos_status_reset(handle->stream().get(), d_status.data());
  return d_status;
}

/// The step-loop fixture: alpha/beta histories, T (pre-filled stale), status.
template<typename T>
struct StepRig {
  std::shared_ptr<DeviceHandle> handle = shared_device();
  int n;
  int ncv;
  std::vector<T> alpha;
  std::vector<T> beta;
  DeviceBuffer<T> d_alpha;
  DeviceBuffer<T> d_beta;
  DeviceBuffer<T> d_T;
  DeviceBuffer<LanczosStatus> d_status;

  StepRig(int n_, int ncv_, std::vector<T> alpha_, std::vector<T> beta_)
      : n(n_), ncv(ncv_), alpha(std::move(alpha_)), beta(std::move(beta_)),
        d_alpha(to_device(handle, alpha)), d_beta(to_device(handle, beta)),
        d_T(to_device(handle, std::vector<T>(static_cast<std::size_t>(ncv_) * ncv_, kStale<T>))),
        d_status(fresh_status(handle)) {}

  /// Run step j on w; return v_next as the kernel left it.
  std::vector<T> step(int j, const std::vector<T> &w) {
    DeviceBuffer<T> d_v = to_device(handle, w);
    device::lanczos_step<T>(handle->stream().get(), n, ncv, j, d_alpha.data(), d_beta.data(),
                            d_v.data(), d_T.data(), d_status.data());
    return from_device(handle, d_v, w.size());
  }

  std::vector<T> matrix() const {
    return from_device(handle, d_T, static_cast<std::size_t>(ncv) * ncv);
  }
  T at(const std::vector<T> &Tm, int r, int c) const {
    return Tm[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * ncv];
  }
  LanczosStatus status() const { return read_status(handle, d_status); }
};

template<typename T>
void expect_normalised(const std::vector<T> &v, const std::vector<T> &w, T b) {
  for (std::size_t i = 0; i < w.size(); ++i) {
    const T want = w[i] / b;
    EXPECT_NEAR(v[i], want, T(2) * std::numeric_limits<T>::epsilon() * std::abs(want)) << i;
  }
}

// ── status reset ─────────────────────────────────────────────────────────────

TEST(LanczosKernelTests, StatusReset) {
  auto handle = shared_device();
  const auto d_status = fresh_status(handle);
  const LanczosStatus st = read_status(handle, d_status);
  EXPECT_EQ(st.breakdown, 0);
  EXPECT_EQ(st.breakdown_step, -1);
  EXPECT_EQ(st.eig_info, 0);
}

// ── step: T writes + normalise, no breakdown ────────────────────────────────

template<typename T>
void check_full_cycle() {
  constexpr int n = 300; // several blocks: the normalise is a grid-stride launch
  constexpr int ncv = 9;
  StepRig<T> rig(n, ncv, random_vector<T>(ncv, 11), random_vector<T>(ncv, 12, T(0.5), T(2)));

  for (int j = 0; j < ncv; ++j) {
    SCOPED_TRACE(::testing::Message() << "j=" << j);
    const std::vector<T> w = random_vector<T>(n, 100u + static_cast<std::uint32_t>(j));
    expect_normalised(rig.step(j, w), w, rig.beta[static_cast<std::size_t>(j)]);
  }

  // Every entry written: the tridiagonal of alpha and beta[0..ncv-2], zeros else.
  const std::vector<T> Tm = rig.matrix();
  for (int c = 0; c < ncv; ++c) {
    for (int r = 0; r < ncv; ++r) {
      T want = T(0);
      if (r == c) {
        want = rig.alpha[static_cast<std::size_t>(c)];
      } else if (r == c + 1) {
        want = rig.beta[static_cast<std::size_t>(c)];
      } else if (c == r + 1) {
        want = rig.beta[static_cast<std::size_t>(r)];
      }
      EXPECT_EQ(rig.at(Tm, r, c), want) << "T(" << r << "," << c << ")";
    }
  }
  const LanczosStatus st = rig.status();
  EXPECT_EQ(st.breakdown, 0);
  EXPECT_EQ(st.breakdown_step, -1);
}

TEST(LanczosKernelTests, StepWritesTridiagonalAndNormalisesFloat) { check_full_cycle<float>(); }
TEST(LanczosKernelTests, StepWritesTridiagonalAndNormalisesDouble) { check_full_cycle<double>(); }

// ── breakdown guard ──────────────────────────────────────────────────────────

template<typename T>
void expect_finite_unchanged(const std::vector<T> &v, const std::vector<T> &w) {
  for (std::size_t i = 0; i < w.size(); ++i) {
    EXPECT_TRUE(std::isfinite(v[i])) << i;
    EXPECT_EQ(v[i], w[i]) << i;
  }
}

/// beta_2 is far above any absolute floor but under eps * ||T_2|| (alpha ~ 1e6),
/// so only a RELATIVE guard trips; beta_4 trips again and must not move the step.
template<typename T>
void check_breakdown(T tiny) {
  constexpr int n = 50;
  constexpr int ncv = 6;
  std::vector<T> alpha(ncv, T(1e6));
  std::vector<T> beta = {T(1), T(1), tiny, T(1), T(0), T(1)};
  StepRig<T> rig(n, ncv, alpha, beta);
  const std::vector<T> w = random_vector<T>(n, 7);

  expect_normalised(rig.step(0, w), w, beta[0]);
  expect_normalised(rig.step(1, w), w, beta[1]);
  expect_finite_unchanged(rig.step(2, w), w); // trips here
  LanczosStatus st = rig.status();
  EXPECT_EQ(st.breakdown, 1);
  EXPECT_EQ(st.breakdown_step, 2);

  expect_finite_unchanged(rig.step(3, w), w); // after a trip: no divide
  expect_finite_unchanged(rig.step(4, w), w); // beta_4 = 0: a second trip
  st = rig.status();
  EXPECT_EQ(st.breakdown, 1);
  EXPECT_EQ(st.breakdown_step, 2); // the FIRST trip is kept

  const std::vector<T> Tm = rig.matrix();
  EXPECT_EQ(rig.at(Tm, 3, 2), T(0)); // tripping coupling written as zero
  EXPECT_EQ(rig.at(Tm, 2, 3), T(0));
  EXPECT_EQ(rig.at(Tm, 4, 3), beta[3]); // a healthy step after a trip still writes T
  EXPECT_EQ(rig.at(Tm, 5, 4), T(0));
  EXPECT_EQ(rig.at(Tm, 2, 2), alpha[2]);
}

TEST(LanczosKernelTests, BreakdownIsRelativeFloat) { check_breakdown<float>(1e-3f); }
TEST(LanczosKernelTests, BreakdownIsRelativeDouble) { check_breakdown<double>(1e-12); }

/// Just above the guard does NOT trip: the threshold is eps * ||T_j||_F.
TEST(LanczosKernelTests, JustAboveGuardDoesNotTrip) {
  constexpr int n = 8;
  constexpr int ncv = 3;
  // T_0 = [alpha_0]: ||T_0||_F = 1e6, so the guard is eps * 1e6.
  const double guard = std::numeric_limits<double>::epsilon() * 1e6;
  StepRig<double> rig(n, ncv, {1e6, 1.0, 1.0}, {4.0 * guard, 1.0, 1.0});
  const std::vector<double> w = random_vector<double>(n, 3);
  expect_normalised(rig.step(0, w), w, 4.0 * guard);
  EXPECT_EQ(rig.status().breakdown, 0);
}

TEST(LanczosKernelTests, NanBetaTripsWithoutPoisoningV) {
  constexpr int n = 20;
  constexpr int ncv = 4;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  StepRig<float> rig(n, ncv, {1.0f, 1.0f, 1.0f, 1.0f}, {nan, 1.0f, 1.0f, 1.0f});
  const std::vector<float> w = random_vector<float>(n, 5);
  expect_finite_unchanged(rig.step(0, w), w);
  const LanczosStatus st = rig.status();
  EXPECT_EQ(st.breakdown, 1);
  EXPECT_EQ(st.breakdown_step, 0);
  EXPECT_EQ(rig.at(rig.matrix(), 1, 0), 0.0f);
}

TEST(LanczosKernelTests, ZeroOperatorTrips) {
  constexpr int n = 10;
  constexpr int ncv = 3;
  StepRig<double> rig(n, ncv, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});
  const std::vector<double> w(n, 0.0);
  expect_finite_unchanged(rig.step(0, w), w);
  EXPECT_EQ(rig.status().breakdown_step, 0);
}

// ── arrowhead (thick restart) ────────────────────────────────────────────────

template<typename T>
void check_arrowhead_then_steps() {
  constexpr int n = 40;
  constexpr int ncv = 8;
  constexpr int k = 3;
  StepRig<T> rig(n, ncv, random_vector<T>(ncv, 21), random_vector<T>(ncv, 22, T(0.5), T(2)));

  const std::vector<T> theta = random_vector<T>(ncv, 23, T(-3), T(3));
  const std::vector<T> S = random_vector<T>(static_cast<std::size_t>(ncv) * ncv, 24);
  const T beta_m = T(0.75);
  auto d_theta = to_device(rig.handle, theta);
  auto d_S = to_device(rig.handle, S);
  auto d_beta_m = to_device(rig.handle, std::vector<T>{beta_m});
  // The last row of S (ncv x ncv, ld ncv): element (ncv-1, i) at (ncv-1) + i*ncv.
  device::lanczos_arrowhead<T>(rig.handle->stream().get(), ncv, k, d_theta.data(),
                               d_S.data() + (ncv - 1), ncv, d_beta_m.data(), rig.d_T.data());

  const auto coupling = [&](int i) {
    return beta_m * S[static_cast<std::size_t>(ncv - 1) + static_cast<std::size_t>(i) * ncv];
  };
  // The arrowhead's expected value at (r, c) when r < k or c < k.
  const auto arrow = [&](int r, int c) {
    if (r == c) {
      return theta[static_cast<std::size_t>(r)];
    }
    if (r == k && c < k) {
      return coupling(c);
    }
    if (c == k && r < k) {
      return coupling(r);
    }
    return T(0);
  };

  std::vector<T> Tm = rig.matrix();
  for (int c = 0; c < ncv; ++c) {
    for (int r = 0; r < ncv; ++r) {
      const T want = (r < k || c < k) ? arrow(r, c) : kStale<T>;
      EXPECT_EQ(rig.at(Tm, r, c), want) << "arrowhead T(" << r << "," << c << ")";
    }
  }

  // Continue the recurrence from step k: no stale entry may survive.
  for (int j = k; j < ncv; ++j) {
    const std::vector<T> w = random_vector<T>(n, 200u + static_cast<std::uint32_t>(j));
    expect_normalised(rig.step(j, w), w, rig.beta[static_cast<std::size_t>(j)]);
  }
  Tm = rig.matrix();
  for (int c = 0; c < ncv; ++c) {
    for (int r = 0; r < ncv; ++r) {
      T want = T(0);
      if (r < k || c < k) {
        want = arrow(r, c);
      } else if (r == c) {
        want = rig.alpha[static_cast<std::size_t>(c)];
      } else if (r == c + 1) {
        want = rig.beta[static_cast<std::size_t>(c)];
      } else if (c == r + 1) {
        want = rig.beta[static_cast<std::size_t>(r)];
      }
      EXPECT_EQ(rig.at(Tm, r, c), want) << "restarted T(" << r << "," << c << ")";
    }
  }
  EXPECT_EQ(rig.status().breakdown, 0);
}

TEST(LanczosKernelTests, ArrowheadLayoutThenStepsFloat) { check_arrowhead_then_steps<float>(); }
TEST(LanczosKernelTests, ArrowheadLayoutThenStepsDouble) { check_arrowhead_then_steps<double>(); }

/// The norm estimate reads the restarted block: a beta small against the kept
/// theta (but large against the fresh alpha alone) trips at step k.
TEST(LanczosKernelTests, GuardSeesArrowheadNorm) {
  constexpr int n = 16;
  constexpr int ncv = 5;
  constexpr int k = 2;
  StepRig<double> rig(n, ncv, std::vector<double>(ncv, 1e-3), std::vector<double>(ncv, 1e-9));
  auto d_theta = to_device(rig.handle, std::vector<double>{1e9, -1e9});
  auto d_s = to_device(rig.handle, std::vector<double>{0.0, 0.0});
  auto d_beta_m = to_device(rig.handle, std::vector<double>{1.0});
  device::lanczos_arrowhead<double>(rig.handle->stream().get(), ncv, k, d_theta.data(),
                                    d_s.data(), 1, d_beta_m.data(), rig.d_T.data());
  const std::vector<double> w = random_vector<double>(n, 9);
  expect_finite_unchanged(rig.step(k, w), w);
  EXPECT_EQ(rig.status().breakdown_step, k);
}

} // namespace
} // namespace calaman
