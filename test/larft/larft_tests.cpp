// Oracle test for calaman.larft: forming the triangular factor T of a block
// reflector must agree with the reference LAPACK for all four (direct, storev)
// cases, computed in the SAME precision on the host.
//
// The oracle is LAPACKE_?larft (this LAPACK build ships the C binding), called
// column-major to match T's storage. T is defined purely algebraically from V
// (unit diagonal implied) and tau -- the recursive identity does not assume any
// relationship between the two -- so an arbitrary small ramp for V and tau is a
// valid input, and both paths read the same referenced entries of V (the
// diagonal and the unreferenced corner are touched by neither).
//
// Only the DEFINED triangle of T is compared: upper for Direct::F, lower for
// Direct::B (LAPACK leaves "the rest of the array" unused). Every case uses a
// non-square V (n > k) and at least one square one (n == k) so the gemm tail,
// which does nothing when n == k, is exercised both ways.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages V, tau and T on the
// device and runs the trmm/gemm/geam kernels, so `ctest -LE gpu` excludes it.
// Built only when calaman::lapack_reference exists; its CMakeLists.txt returns
// early otherwise, so its absence is a missing tier, not a silent pass
// (docs/architecture.md §3).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.larft;
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

// The reference oracle: LAPACKE ?larft, same precision as the device path,
// column-major to match T's storage.
lapack_int ref_larft(char direct, char storev, int n, int k, const float *v, int ldv,
                     const float *tau, float *t, int ldt) {
  return LAPACKE_slarft(LAPACK_COL_MAJOR, direct, storev, n, k, v, ldv, tau, t, ldt);
}
lapack_int ref_larft(char direct, char storev, int n, int k, const double *v, int ldv,
                     const double *tau, double *t, int ldt) {
  return LAPACKE_dlarft(LAPACK_COL_MAJOR, direct, storev, n, k, v, ldv, tau, t, ldt);
}

// Small exactly-representable ramps for V and tau. tau is kept positive and modest
// so T stays well inside the mantissa for both precisions.
template<typename T>
T v_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 5) - 2);
}
template<typename T>
T tau_at(std::size_t i) {
  return static_cast<T>(0.5) + static_cast<T>(static_cast<int>(i % 3)) * static_cast<T>(0.1);
}

/// @brief calaman::larft on the device must match the reference for one case
///
/// V is n-by-k (StoreV::C, ldv == n) or k-by-n (StoreV::R, ldv == k); T is k-by-k
/// with ldt == k. Both paths get the same ramp V and tau; the comparison covers
/// only the triangle `direct` defines (upper for F, lower for B), to the shared
/// factorization tolerance scaled by ||T||.
template<typename T>
void expect_matches_reference(Direct direct, StoreV storev, int n, int k) {
  const char dc = direct == Direct::F ? 'F' : 'B';
  const char sc = storev == StoreV::C ? 'C' : 'R';
  const bool colv = storev == StoreV::C;
  const int ldv = colv ? n : k;
  const int vcols = colv ? k : n;
  const std::size_t vn = static_cast<std::size_t>(ldv) * static_cast<std::size_t>(vcols);
  const std::size_t tn = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_v(vn);
  HostBuffer<T> host_tau(static_cast<std::size_t>(k));
  HostBuffer<T> host_t(tn);
  std::vector<T> ref_v(vn);
  std::vector<T> ref_tau(static_cast<std::size_t>(k));
  std::vector<T> ref_t(tn, T{0});
  for (std::size_t i = 0; i < vn; ++i) {
    host_v.data()[i] = v_at<T>(i);
    ref_v[i] = host_v.data()[i];
  }
  for (int i = 0; i < k; ++i) {
    host_tau.data()[static_cast<std::size_t>(i)] = tau_at<T>(static_cast<std::size_t>(i));
    ref_tau[static_cast<std::size_t>(i)] = host_tau.data()[static_cast<std::size_t>(i)];
  }
  for (std::size_t i = 0; i < tn; ++i) {
    host_t.data()[i] = T{0};
  }

  // Reference: fill ref_t with the triangular factor, in host memory.
  const lapack_int info =
      ref_larft(dc, sc, n, k, ref_v.data(), ldv, ref_tau.data(), ref_t.data(), k);
  ASSERT_EQ(info, 0) << "direct=" << dc << " storev=" << sc << " n=" << n << " k=" << k;

  auto d_v = to_device(handle, host_v, vn);
  auto d_tau = to_device(handle, host_tau, static_cast<std::size_t>(k));
  auto d_t = to_device(handle, host_t, tn);

  const auto status =
      larft<T>(blas, direct, storev, n, k, d_v.data(), ldv, d_tau.data(), d_t.data(), k);
  const auto got = from_device(handle, d_t, tn);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS)
      << "direct=" << dc << " storev=" << sc << " n=" << n << " k=" << k;
  const T tol = factorization_tol<T>(frobenius_norm<T>(ref_t), static_cast<std::size_t>(k),
                                     static_cast<std::size_t>(k));
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < k; ++i) {
      const bool defined = direct == Direct::F ? (i <= j) : (i >= j);
      if (!defined) {
        continue;
      }
      const std::size_t idx =
          static_cast<std::size_t>(j) * static_cast<std::size_t>(k) + static_cast<std::size_t>(i);
      EXPECT_NEAR(got[idx], ref_t[idx], tol) << "direct=" << dc << " storev=" << sc << " n=" << n
                                             << " k=" << k << " i=" << i << " j=" << j;
    }
  }
}

// The (n, k) grid every case sweeps: a base case (k == 1), a square one (n == k,
// no gemm tail), odd and even k, and a deeper recursion, all with k <= n.
template<typename T>
void sweep(Direct direct, StoreV storev) {
  expect_matches_reference<T>(direct, storev, 1, 1);
  expect_matches_reference<T>(direct, storev, 6, 1);
  expect_matches_reference<T>(direct, storev, 5, 5);
  expect_matches_reference<T>(direct, storev, 6, 4);
  expect_matches_reference<T>(direct, storev, 8, 3);
  expect_matches_reference<T>(direct, storev, 9, 6);
}

TEST(LarftOracleTests, ForwardColumnwiseFloat) {
  sweep<float>(Direct::F, StoreV::C);
}
TEST(LarftOracleTests, ForwardColumnwiseDouble) {
  sweep<double>(Direct::F, StoreV::C);
}
TEST(LarftOracleTests, ForwardRowwiseFloat) {
  sweep<float>(Direct::F, StoreV::R);
}
TEST(LarftOracleTests, ForwardRowwiseDouble) {
  sweep<double>(Direct::F, StoreV::R);
}
TEST(LarftOracleTests, BackwardColumnwiseFloat) {
  sweep<float>(Direct::B, StoreV::C);
}
TEST(LarftOracleTests, BackwardColumnwiseDouble) {
  sweep<double>(Direct::B, StoreV::C);
}
TEST(LarftOracleTests, BackwardRowwiseFloat) {
  sweep<float>(Direct::B, StoreV::R);
}
TEST(LarftOracleTests, BackwardRowwiseDouble) {
  sweep<double>(Direct::B, StoreV::R);
}

} // namespace
} // namespace calaman
