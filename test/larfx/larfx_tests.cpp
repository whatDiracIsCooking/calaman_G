// Oracle test for calaman.larfx: applying the elementary reflector
// H = I - tau*v*v^T to a matrix C must agree with the reference LAPACK for BOTH
// sides and BOTH of larfx's paths -- the fused order < 11 kernel and the
// order >= 11 deferral to larf -- computed in the SAME precision on the host.
//
// The oracle is netlib's own ?larfx, called through its Fortran ABI (dlarfx_ /
// slarfx_), not LAPACKE: this is the exact routine this PR mirrors, with its own
// m,n<=10 special-casing, so it is the faithful oracle for both paths at once.
// ?larfx takes no incv (unit stride on v, v(1) stored), matching calaman::larfx.
// The reference matrix is column-major, matching C's storage.
//
// C is m x n, non-square in every case so a side-L/side-R mix-up (which swaps
// the roles of the m- and n-length vectors) cannot pass by coincidence. v has
// length m for side L and n for side R; the device workspace w has the
// complementary length (n for L, m for R) -- referenced only on the order >= 11
// fallback, but always allocated here. Cases straddle order 10/11 so BOTH of
// larfx's paths are exercised: orders 3..8 take the fused kernel, order 12 the
// larf fallback.
//
// Inputs are small, so H*C stays well inside the mantissa and the only spread
// between the GPU and the reference is summation order, absorbed by the shared
// factorization_tol. tau == 0 is checked separately: H is the identity, so C
// must come back bitwise-unchanged.
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and runs
// the kernel / gemv / ger, so `ctest -LE gpu` excludes it. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass (docs/architecture.md §3).

#include <gtest/gtest.h>

#include <cstddef> // std::size_t for the Fortran prototypes below

// Netlib ?larfx through its Fortran ABI. gfortran passes scalars/arrays by
// reference and appends a hidden length for each CHARACTER argument (here the
// trailing size_t for `side`). ?larfx has no incv argument. The symbols live in
// LAPACK::LAPACK, linked by calaman::lapack_reference.
extern "C" {
void slarfx_(const char *side, const int *m, const int *n, const float *v, const float *tau,
             float *c, const int *ldc, float *work, std::size_t side_len);
void dlarfx_(const char *side, const int *m, const int *n, const double *v, const double *tau,
             double *c, const int *ldc, double *work, std::size_t side_len);
}

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.larfx;
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
using test::frobenius_norm;
using test::shared_device;

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

/// @brief Download `device` (length n) back to a host vector
template<typename T>
std::vector<T> from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<T> &device,
                           std::size_t n) {
  HostBuffer<T> host(n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  std::vector<T> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = host.data()[i];
  }
  return out;
}

// The reference oracle: netlib ?larfx (Fortran ABI), same precision as the
// device path, column-major to match C's storage. No incv argument.
void ref_larfx(char side, int m, int n, const float *v, float tau, float *c, int ldc,
               float *work) {
  slarfx_(&side, &m, &n, v, &tau, c, &ldc, work, 1);
}
void ref_larfx(char side, int m, int n, const double *v, double tau, double *c, int ldc,
               double *work) {
  dlarfx_(&side, &m, &n, v, &tau, c, &ldc, work, 1);
}

// Small exactly-representable ramps of mixed sign for C and v.
template<typename T>
T c_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 11) - 5);
}
template<typename T>
T v_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 7) - 3);
}

/// @brief calaman::larfx on the device must match the reference for one case
///
/// C is m x n column-major with ldc == m. v has length m (side L) or n (side R);
/// the workspace w and the reference `work` have the complementary length. The
/// final C is compared element-by-element against LAPACK's, to the shared
/// factorization tolerance scaled by ||C_out||. The order (m for L, n for R)
/// selects which of larfx's two paths runs -- the caller chooses it per case.
template<typename T>
void expect_matches_reference(Side side, int m, int n, T tau) {
  const bool left = side == Side::L;
  const int vlen = left ? m : n;
  const int wlen = left ? n : m;
  const std::size_t cn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_c(cn);
  HostBuffer<T> host_v(static_cast<std::size_t>(vlen));
  std::vector<T> ref_c(cn);
  std::vector<T> ref_v(static_cast<std::size_t>(vlen));
  for (std::size_t i = 0; i < cn; ++i) {
    host_c.data()[i] = c_at<T>(i);
    ref_c[i] = host_c.data()[i];
  }
  for (int i = 0; i < vlen; ++i) {
    host_v.data()[static_cast<std::size_t>(i)] = v_at<T>(static_cast<std::size_t>(i));
    ref_v[static_cast<std::size_t>(i)] = host_v.data()[static_cast<std::size_t>(i)];
  }

  // Reference: overwrite ref_c with H applied, in host memory.
  std::vector<T> ref_work(static_cast<std::size_t>(wlen));
  const char side_char = left ? 'L' : 'R';
  ref_larfx(side_char, m, n, ref_v.data(), tau, ref_c.data(), m, ref_work.data());

  auto d_c = to_device(handle, host_c, cn);
  auto d_v = to_device(handle, host_v, static_cast<std::size_t>(vlen));
  DeviceBuffer<T> d_w(static_cast<std::size_t>(wlen), handle);

  const auto status = larfx<T>(blas, side, m, n, d_v.data(), tau, d_c.data(), m, d_w.data());
  const auto got = from_device(handle, d_c, cn);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS)
      << "side=" << side_char << " m=" << m << " n=" << n;
  const T tol = factorization_tol<T>(frobenius_norm<T>(ref_c), static_cast<std::size_t>(m),
                                     static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < cn; ++i) {
    EXPECT_NEAR(got[i], ref_c[i], tol)
        << "side=" << side_char << " m=" << m << " n=" << n << " i=" << i;
  }
}

// Side L: order = m. m in 3..8 exercises the fused kernel; m = 12 the fallback.
TEST(LarfxOracleTests, MatchesReferenceLeftFusedFloat) {
  expect_matches_reference<float>(Side::L, 5, 3, 0.7F);
  expect_matches_reference<float>(Side::L, 7, 4, 1.3F);
  expect_matches_reference<float>(Side::L, 3, 8, 0.4F);
}

TEST(LarfxOracleTests, MatchesReferenceLeftFusedDouble) {
  expect_matches_reference<double>(Side::L, 5, 3, 0.7);
  expect_matches_reference<double>(Side::L, 7, 4, 1.3);
  expect_matches_reference<double>(Side::L, 3, 8, 0.4);
}

TEST(LarfxOracleTests, MatchesReferenceRightFusedFloat) {
  expect_matches_reference<float>(Side::R, 5, 3, 0.7F);
  expect_matches_reference<float>(Side::R, 7, 4, 1.3F);
  expect_matches_reference<float>(Side::R, 8, 3, 0.4F);
}

TEST(LarfxOracleTests, MatchesReferenceRightFusedDouble) {
  expect_matches_reference<double>(Side::R, 5, 3, 0.7);
  expect_matches_reference<double>(Side::R, 7, 4, 1.3);
  expect_matches_reference<double>(Side::R, 8, 3, 0.4);
}

// The order >= 11 fallback to larf (gemv + ger). Side L order = m = 12; side R
// order = n = 12. The complementary dimension stays small so the reference is
// cheap, and both are non-square to keep the side check honest.
TEST(LarfxOracleTests, MatchesReferenceLargeOrderFallbackFloat) {
  expect_matches_reference<float>(Side::L, 12, 4, 0.6F);
  expect_matches_reference<float>(Side::R, 4, 12, 0.9F);
}

TEST(LarfxOracleTests, MatchesReferenceLargeOrderFallbackDouble) {
  expect_matches_reference<double>(Side::L, 12, 4, 0.6);
  expect_matches_reference<double>(Side::R, 4, 12, 0.9);
}

// Order exactly on the 10/11 boundary: order 10 is the last fused case, order 11
// the first fallback. A boundary off-by-one would land one of these on the wrong
// path and perturb C beyond the oracle tolerance.
TEST(LarfxOracleTests, MatchesReferenceOrderBoundaryDouble) {
  expect_matches_reference<double>(Side::L, 10, 4, 0.5); // fused, order 10
  expect_matches_reference<double>(Side::L, 11, 4, 0.5); // fallback, order 11
  expect_matches_reference<double>(Side::R, 4, 10, 0.5); // fused, order 10
  expect_matches_reference<double>(Side::R, 4, 11, 0.5); // fallback, order 11
}

TEST(LarfxOracleTests, TauZeroIsNoop) {
  // tau == 0: H is the identity, so C must come back exactly as uploaded, for
  // both sides and both paths (orders straddling 10/11). larfx returns before
  // enqueuing anything; the check is bitwise against the ORIGINAL C, so a stray
  // update with a nonzero scalar would perturb it below any oracle tolerance and
  // still fail.
  struct Case {
    Side side;
    int m;
    int n;
  };
  for (const Case c : {Case{Side::L, 6, 4}, Case{Side::R, 4, 6}, Case{Side::L, 12, 4},
                       Case{Side::R, 4, 12}}) {
    const int m = c.m;
    const int n = c.n;
    const std::size_t cn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const int vlen = c.side == Side::L ? m : n;
    const int wlen = c.side == Side::L ? n : m;

    auto handle = shared_device();
    wwr::wwrblasHandle_t blas{};
    ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
    ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

    HostBuffer<double> host_c(cn);
    HostBuffer<double> host_v(static_cast<std::size_t>(vlen));
    for (std::size_t i = 0; i < cn; ++i) {
      host_c.data()[i] = c_at<double>(i);
    }
    for (int i = 0; i < vlen; ++i) {
      host_v.data()[static_cast<std::size_t>(i)] = v_at<double>(static_cast<std::size_t>(i));
    }

    auto d_c = to_device(handle, host_c, cn);
    auto d_v = to_device(handle, host_v, static_cast<std::size_t>(vlen));
    DeviceBuffer<double> d_w(static_cast<std::size_t>(wlen), handle);

    const auto status = larfx<double>(blas, c.side, m, n, d_v.data(), 0.0, d_c.data(), m,
                                      d_w.data());
    const auto got = from_device(handle, d_c, cn);
    wwr::wwrblasDestroy(blas);

    EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
    for (std::size_t i = 0; i < cn; ++i) {
      EXPECT_DOUBLE_EQ(got[i], c_at<double>(i))
          << "side=" << (c.side == Side::L ? 'L' : 'R') << " m=" << m << " n=" << n << " i=" << i;
    }
  }
}

} // namespace
} // namespace calaman
