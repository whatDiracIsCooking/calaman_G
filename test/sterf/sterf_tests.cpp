// Oracle test for calaman.sterf: the eigenvalues of a symmetric tridiagonal
// (d, e) computed on the device must agree with reference LAPACKE_?sterf run on
// the host over the same input, in the same precision. Both run the identical
// root-free QL/QR algorithm, so INFO matches exactly and only rounding (device
// FMA contraction) separates the eigenvalues -- compared to the shared
// tolerance scaled by the input's max-abs norm, since ?sterf is backward stable
// in that norm. The device result must also be sorted ascending.
//
// Inputs: random of several orders, graded (entries spanning many decades),
// clustered (near-identical diagonal, tiny coupling), pre-split (zero
// off-diagonals, so the block search runs), huge and tiny scales (so the
// per-block ?lascl scaling runs both ways), the 1-2-1 Toeplitz, and n = 0, 1, 2.
//
// REQUIRES_GPU except SterfHostTests (an n < 0 rejection that launches
// nothing). Built only when calaman::lapack_reference exists (CMakeLists.txt).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.sterf;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;
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

lapack_int ref_sterf(lapack_int n, float *d, float *e) { return LAPACKE_ssterf(n, d, e); }
lapack_int ref_sterf(lapack_int n, double *d, double *e) { return LAPACKE_dsterf(n, d, e); }

// Run sterf on the device and against the reference over (d, e) and compare.
template<typename T>
void check(const std::vector<T> &d, const std::vector<T> &e, const std::string &ctx) {
  auto handle = shared_device();
  const int n = static_cast<int>(d.size());
  ASSERT_EQ(e.size(), n > 0 ? d.size() - 1 : 0) << ctx;

  T anorm{0};
  for (const T v : d) {
    anorm = std::max(anorm, std::abs(v));
  }
  for (const T v : e) {
    anorm = std::max(anorm, std::abs(v));
  }

  auto d_d = to_device(handle, d);
  auto d_e = to_device(handle, e);
  auto d_info = to_device(handle, std::vector<int>{-1});
  const auto status =
      calaman::sterf<T>(handle->stream().get(), n, d_d.data(), d_e.data(), d_info.data());
  ASSERT_TRUE(status.ok()) << ctx << ": sterf returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_d, d.size());
  const int g_info = from_device(handle, d_info, 1)[0];

  std::vector<T> r_d = d;
  std::vector<T> r_e = e.empty() ? std::vector<T>(1) : e;
  const lapack_int r_info = ref_sterf(n, r_d.data(), r_e.data());

  ASSERT_EQ(r_info, 0) << ctx << ": reference ?sterf did not converge";
  EXPECT_EQ(g_info, r_info) << ctx << ": info";

  const std::size_t un = static_cast<std::size_t>(n);
  const T tol = factorization_tol<T>(anorm, un, un) + std::numeric_limits<T>::min();
  for (std::size_t i = 0; i < un; ++i) {
    EXPECT_NEAR(got[i], r_d[i], tol) << ctx << ": eigenvalue " << i;
    if (i > 0) {
      EXPECT_LE(got[i - 1], got[i]) << ctx << ": not ascending at " << i;
    }
  }
}

template<typename T>
std::vector<T> make_random(std::size_t n, unsigned seed, double lo = -1.0, double hi = 1.0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(lo, hi);
  std::vector<T> v(n);
  for (auto &x : v) {
    x = static_cast<T>(dist(gen));
  }
  return v;
}

// ── cases ────────────────────────────────────────────────────────────────────

template<typename T>
void tiny_orders() {
  check<T>({}, {}, "n=0");
  check<T>({T{3.5}}, {}, "n=1");
  check<T>({T{2}, T{-1}}, {T{0.75}}, "n=2");
  check<T>({T{1}, T{1}}, {T{0}}, "n=2 split");
  check<T>({T{-4}, T{4}}, {T{1e-3}}, "n=2 swapped order");
}

template<typename T>
void random_orders() {
  unsigned seed = 193;
  for (std::size_t n : {3u, 4u, 7u, 16u, 50u, 128u, 300u}) {
    const auto d = make_random<T>(n, seed++);
    const auto e = make_random<T>(n - 1, seed++);
    check<T>(d, e, "random n=" + std::to_string(n));
  }
}

// Entries decaying (and growing) geometrically over many decades: the QL vs QR
// choice flips with the direction of grading.
template<typename T>
void graded() {
  const std::size_t n = 24;
  const double ratio = std::is_same_v<T, float> ? 0.5 : 0.1;
  std::vector<T> d(n), e(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = static_cast<T>(std::pow(ratio, static_cast<double>(i)));
  }
  for (std::size_t i = 0; i + 1 < n; ++i) {
    e[i] = static_cast<T>(0.5 * std::pow(ratio, i + 0.5));
  }
  check<T>(d, e, "graded decreasing");
  std::reverse(d.begin(), d.end());
  std::reverse(e.begin(), e.end());
  check<T>(d, e, "graded increasing");
}

template<typename T>
void clustered() {
  const std::size_t n = 40;
  std::vector<T> d(n), e(n - 1);
  for (std::size_t i = 0; i < n; ++i) {
    d[i] = T{1} + static_cast<T>(i % 3) * std::numeric_limits<T>::epsilon() * T{8};
  }
  for (auto &x : e) {
    x = static_cast<T>(1e-3);
  }
  check<T>(d, e, "clustered");
  // Wilkinson W21+: pairs of nearly equal eigenvalues.
  const std::size_t m = 21;
  std::vector<T> wd(m), we(m - 1, T{1});
  for (std::size_t i = 0; i < m; ++i) {
    wd[i] = static_cast<T>(std::abs(static_cast<int>(i) - 10));
  }
  check<T>(wd, we, "wilkinson21");
}

// Zero off-diagonals carve the matrix into independent blocks, including 1x1
// and 2x2 ones and a leading/trailing isolated entry.
template<typename T>
void pre_split() {
  const std::size_t n = 20;
  auto d = make_random<T>(n, 7);
  auto e = make_random<T>(n - 1, 8);
  for (std::size_t i : {0u, 3u, 5u, 6u, 11u, 18u}) {
    e[i] = T{0};
  }
  check<T>(d, e, "pre-split");
  std::vector<T> diag = make_random<T>(n, 9);
  check<T>(diag, std::vector<T>(n - 1, T{0}), "diagonal");
}

// Scales far from 1 drive the per-block ?lascl in both directions.
template<typename T>
void scaled() {
  const double big = std::is_same_v<T, float> ? 1e30 : 1e250;
  const double small = std::is_same_v<T, float> ? 1e-30 : 1e-250;
  const std::size_t n = 30;
  for (double s : {big, small}) {
    auto d = make_random<T>(n, 11);
    auto e = make_random<T>(n - 1, 12);
    for (auto &x : d) {
      x = static_cast<T>(x * s);
    }
    for (auto &x : e) {
      x = static_cast<T>(x * s);
    }
    check<T>(d, e, s > 1 ? "scaled huge" : "scaled tiny");
  }
}

// The 1-2-1 Toeplitz: eigenvalues 2 - 2cos(k*pi/(n+1)).
template<typename T>
void toeplitz() {
  const std::size_t n = 64;
  check<T>(std::vector<T>(n, T{2}), std::vector<T>(n - 1, T{-1}), "toeplitz 1-2-1");
}

TEST(SterfOracleTests, TinyOrdersFloat) { tiny_orders<float>(); }
TEST(SterfOracleTests, TinyOrdersDouble) { tiny_orders<double>(); }
TEST(SterfOracleTests, RandomFloat) { random_orders<float>(); }
TEST(SterfOracleTests, RandomDouble) { random_orders<double>(); }
TEST(SterfOracleTests, GradedFloat) { graded<float>(); }
TEST(SterfOracleTests, GradedDouble) { graded<double>(); }
TEST(SterfOracleTests, ClusteredFloat) { clustered<float>(); }
TEST(SterfOracleTests, ClusteredDouble) { clustered<double>(); }
TEST(SterfOracleTests, PreSplitFloat) { pre_split<float>(); }
TEST(SterfOracleTests, PreSplitDouble) { pre_split<double>(); }
TEST(SterfOracleTests, ScaledFloat) { scaled<float>(); }
TEST(SterfOracleTests, ScaledDouble) { scaled<double>(); }
TEST(SterfOracleTests, ToeplitzFloat) { toeplitz<float>(); }
TEST(SterfOracleTests, ToeplitzDouble) { toeplitz<double>(); }

// n < 0 is rejected at the front door, before any launch -- no card needed.
TEST(SterfHostTests, NegativeOrderIsInvalid) {
  const auto status = calaman::sterf<double>(nullptr, -1, nullptr, nullptr, nullptr);
  EXPECT_FALSE(status.ok());
}

} // namespace
} // namespace calaman
