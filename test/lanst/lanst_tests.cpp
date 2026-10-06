// Oracle test for calaman.lanst -- the ?lanst norm of a symmetric tridiagonal
// (d, e). The oracle is the reference ?lanst (LAPACK_?lanst) in the SAME precision on the
// host.
//
// LanstOracleTests stages (d, e) on the device and runs the kernel, so it is
// REQUIRES_GPU (labeled `gpu`). LanstHelperTests calls the header-only
// lanst_max_abs (lapack/lanst/lanst.h) on the HOST over sub-ranges -- the shape
// ?sterf/?steqr use it in -- and needs no card. Built only when
// calaman::lapack_reference exists; see this directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

#include "lapack/lanst/lanst.h"

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lanst;
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
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
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
T scalar_from_device(const DeviceBuffer<T> &device) {
  const auto handle = shared_device();
  HostBuffer<T> host(1);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return host.data()[0];
}

char norm_char(MatrixNorm which) {
  switch (which) {
  case MatrixNorm::max_abs:
    return 'M';
  case MatrixNorm::one:
    return '1';
  case MatrixNorm::inf:
    return 'I';
  case MatrixNorm::frobenius:
    return 'F';
  }
  return 'M';
}

// LAPACKE has no ?lanst wrapper, so call the Fortran routine through lapack.h's
// LAPACK_?lanst (which lapacke.h includes; it appends the hidden string length).
float ref_lanst(char norm, std::size_t n, const float *d, const float *e) {
  const lapack_int ni = static_cast<lapack_int>(n);
  return static_cast<float>(LAPACK_slanst(&norm, &ni, d, e));
}
double ref_lanst(char norm, std::size_t n, const double *d, const double *e) {
  const lapack_int ni = static_cast<lapack_int>(n);
  return LAPACK_dlanst(&norm, &ni, d, e);
}

// Mixed-sign values spanning a few binades, so neither d nor e dominates.
template<typename T>
std::vector<T> random_vector(std::size_t len, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  std::vector<T> v(len);
  for (auto &x : v) {
    x = static_cast<T>(dist(gen));
  }
  return v;
}

// The max norm compares and copies only, so it must match exactly; the sums
// differ from the reference in order (and, for 'F', by ?lassq's scaling), so
// they get the shared eps-relative tolerance. 'F' sums 2n-1 terms, but as
// chunks of at most 17 then a pairwise tree, so its error stays a few eps.
template<typename T>
T norm_tol(MatrixNorm which, T ref) {
  switch (which) {
  case MatrixNorm::max_abs:
    return T{0};
  case MatrixNorm::one:
  case MatrixNorm::inf:
    return factorization_tol<T>(ref, 3, 3);
  case MatrixNorm::frobenius:
    return factorization_tol<T>(ref, 4, 4);
  }
  return T{0};
}

template<typename T>
void expect_matches_reference(MatrixNorm which, std::size_t n) {
  const auto d = random_vector<T>(n, static_cast<std::uint32_t>(17 * n + 1));
  // One spare slot past n-1, holding a huge sentinel the kernel must not read.
  auto e = random_vector<T>(n == 0 ? 1 : n, static_cast<std::uint32_t>(29 * n + 3));
  e.back() = static_cast<T>(1e6);
  const T oracle = ref_lanst(norm_char(which), n, d.data(), e.data());

  auto d_d = to_device(d);
  auto d_e = to_device(e);
  DeviceBuffer<T> d_result(1, shared_device());
  const Status s = lanst<T>(shared_device()->stream().get(), which, n, d_d.data(), d_e.data(),
                            d_result.data());
  ASSERT_TRUE(s.ok()) << "lanst returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol(which, oracle))
      << "norm=" << norm_char(which) << " n=" << n;
}

template<typename T>
void run_sizes(MatrixNorm which) {
  // 1..3: the edge rows; 255..257: around the block's thread count (chunk 1 vs
  // 2); 1000/4099: multi-row chunks with a ragged last one.
  for (const std::size_t n : {1u, 2u, 3u, 7u, 64u, 255u, 256u, 257u, 1000u, 4099u}) {
    expect_matches_reference<T>(which, n);
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanst, per norm and type
// ========================================================================

TEST(LanstOracleTests, MaxAbsFloat) {
  run_sizes<float>(MatrixNorm::max_abs);
}
TEST(LanstOracleTests, MaxAbsDouble) {
  run_sizes<double>(MatrixNorm::max_abs);
}
TEST(LanstOracleTests, OneNormFloat) {
  run_sizes<float>(MatrixNorm::one);
}
TEST(LanstOracleTests, OneNormDouble) {
  run_sizes<double>(MatrixNorm::one);
}
TEST(LanstOracleTests, InfNormFloat) {
  run_sizes<float>(MatrixNorm::inf);
}
TEST(LanstOracleTests, InfNormDouble) {
  run_sizes<double>(MatrixNorm::inf);
}
TEST(LanstOracleTests, FrobeniusFloat) {
  run_sizes<float>(MatrixNorm::frobenius);
}
TEST(LanstOracleTests, FrobeniusDouble) {
  run_sizes<double>(MatrixNorm::frobenius);
}

// n == 0 writes 0 (DLANST returns 0) over a pre-seeded sentinel, for every norm.
TEST(LanstOracleTests, EmptyWritesZero) {
  for (const MatrixNorm which :
       {MatrixNorm::max_abs, MatrixNorm::one, MatrixNorm::inf, MatrixNorm::frobenius}) {
    expect_matches_reference<double>(which, 0);
    auto d_result = to_device(std::vector<float>{-12345.0f});
    auto dummy = to_device(std::vector<float>{42.0f});
    ASSERT_TRUE(lanst<float>(shared_device()->stream().get(), which, 0, dummy.data(),
                             dummy.data(), d_result.data())
                    .ok());
    EXPECT_EQ(scalar_from_device(d_result), 0.0f) << "norm=" << norm_char(which);
  }
}

// A NaN anywhere -- including the last off-diagonal, which a chunk boundary
// hands to its left neighbour -- propagates through the max norms, as DLANST's
// DISNAN guard does.
TEST(LanstOracleTests, NaNPropagatesThroughMaxNorms) {
  const std::size_t n = 600; // chunk 3: e[2] links thread 0 to thread 1
  for (const std::size_t pos : {std::size_t{2}, n - 2}) {
    auto d = random_vector<double>(n, 5);
    auto e = random_vector<double>(n - 1, 6);
    e[pos] = std::numeric_limits<double>::quiet_NaN();
    auto d_d = to_device(d);
    auto d_e = to_device(e);
    DeviceBuffer<double> d_result(1, shared_device());
    for (const MatrixNorm which : {MatrixNorm::max_abs, MatrixNorm::one}) {
      ASSERT_TRUE(lanst<double>(shared_device()->stream().get(), which, n, d_d.data(), d_e.data(),
                                d_result.data())
                      .ok());
      EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
          << "norm=" << norm_char(which) << " pos=" << pos;
    }
  }
}

// ========================================================================
// Host: the per-thread lanst_max_abs over sub-ranges, as ?sterf calls it
// ========================================================================

template<typename T>
void expect_helper_matches_reference() {
  const std::size_t n = 50;
  const auto d = random_vector<T>(n, 11);
  const auto e = random_vector<T>(n - 1, 12);
  // Every sub-block [l, l+len), the DSTERF `DLANST('M', LEND-L+1, D(L), E(L))`.
  for (std::size_t l = 0; l < n; ++l) {
    for (std::size_t len = 0; l + len <= n; ++len) {
      const T got = lanst_max_abs(len, d.data() + l, e.data() + l);
      const T want = ref_lanst('M', len, d.data() + l, e.data() + l);
      ASSERT_EQ(got, want) << "l=" << l << " len=" << len;
    }
  }
}

TEST(LanstHelperTests, SubRangesMatchReferenceFloat) {
  expect_helper_matches_reference<float>();
}
TEST(LanstHelperTests, SubRangesMatchReferenceDouble) {
  expect_helper_matches_reference<double>();
}

// n == 0 returns 0 and n == 1 reads only d[0] -- e may be null for both.
TEST(LanstHelperTests, EdgeOrders) {
  const double d = -3.5;
  EXPECT_EQ(lanst_max_abs<double>(0, nullptr, nullptr), 0.0);
  EXPECT_EQ(lanst_max_abs<double>(1, &d, nullptr), 3.5);
}

TEST(LanstHelperTests, NaNPropagates) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> d = {1.0, nan, 100.0};
  const std::vector<double> e = {2.0, 3.0};
  EXPECT_TRUE(std::isnan(lanst_max_abs(d.size(), d.data(), e.data())));
}

} // namespace
} // namespace calaman
