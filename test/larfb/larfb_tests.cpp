// Oracle test for calaman.larfb: applying a block reflector H (or H^T) to a
// general matrix must agree with the reference LAPACK for all sixteen
// (side, trans, direct, storev) cases, computed in the SAME precision on the
// host.
//
// The oracle is LAPACKE_?larfb (this LAPACK build ships the C binding), called
// column-major to match C's storage. ?larfb's update is purely algebraic in V,
// T and C -- the block identity assumes no relationship among them -- so an
// arbitrary small ramp for each is a valid input, and both paths read the same
// referenced entries of V (its strict trapezoid; the unit diagonal is implied)
// and the defined triangle of T. C is overwritten in full, so the whole m-by-n
// result is compared to the shared factorization tolerance scaled by ||C||.
//
// Every case sweeps a square V (m == k or n == k, no gemm tail) and non-square
// ones so the rectangular-tail gemm is exercised both ways, plus a k == 1 base.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages V, T, C and the workspace
// on the device and runs the trmm/gemm/geam kernels, so `ctest -LE gpu` excludes
// it. Built only when calaman::lapack_reference exists; its CMakeLists.txt
// returns early otherwise, so its absence is a missing tier, not a silent pass
// (docs/architecture.md section 3).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.larfb;
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

// The reference oracle: LAPACKE ?larfb, same precision as the device path,
// column-major to match C's storage.
lapack_int ref_larfb(char side, char trans, char direct, char storev, int m, int n, int k,
                     const float *v, int ldv, const float *t, int ldt, float *c, int ldc) {
  return LAPACKE_slarfb(LAPACK_COL_MAJOR, side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c,
                        ldc);
}
lapack_int ref_larfb(char side, char trans, char direct, char storev, int m, int n, int k,
                     const double *v, int ldv, const double *t, int ldt, double *c, int ldc) {
  return LAPACKE_dlarfb(LAPACK_COL_MAJOR, side, trans, direct, storev, m, n, k, v, ldv, t, ldt, c,
                        ldc);
}

// Small exactly-representable ramps. V and C span a modest range; T is kept
// small so the rank-k update stays well inside the mantissa for both precisions.
template<typename T>
T v_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 5) - 2);
}
template<typename T>
T c_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 7) - 3);
}
template<typename T>
T t_at(std::size_t i) {
  return static_cast<T>(0.25) + static_cast<T>(static_cast<int>(i % 4)) * static_cast<T>(0.1);
}

/// @brief calaman::larfb on the device must match the reference for one case
///
/// V is len-by-k (StoreV::C, ldv == len) or k-by-len (StoreV::R, ldv == k), with
/// len = m (Side::L) or n (Side::R); T is k-by-k (ldt == k); C is m-by-n
/// (ldc == m). Both paths get the same ramps; the whole overwritten C is
/// compared to the shared factorization tolerance scaled by ||C||.
template<typename T>
void expect_matches_reference(Side side, Trans trans, Direct direct, StoreV storev, int m, int n,
                              int k) {
  const char sd = side == Side::L ? 'L' : 'R';
  const char tr = trans == Trans::N ? 'N' : 'T';
  const char dc = direct == Direct::F ? 'F' : 'B';
  const char sc = storev == StoreV::C ? 'C' : 'R';
  const bool left = side == Side::L;
  const bool colv = storev == StoreV::C;

  const int len = left ? m : n;
  const int ldv = colv ? len : k;
  const int vcols = colv ? k : len;
  const int ldwork = left ? n : m;
  const std::size_t vn = static_cast<std::size_t>(ldv) * static_cast<std::size_t>(vcols);
  const std::size_t tn = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
  const std::size_t cn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  const std::size_t wn = static_cast<std::size_t>(ldwork) * static_cast<std::size_t>(k);

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_v(vn);
  HostBuffer<T> host_t(tn);
  HostBuffer<T> host_c(cn);
  HostBuffer<T> host_w(wn);
  std::vector<T> ref_v(vn);
  std::vector<T> ref_t(tn);
  std::vector<T> ref_c(cn);
  for (std::size_t i = 0; i < vn; ++i) {
    host_v.data()[i] = v_at<T>(i);
    ref_v[i] = host_v.data()[i];
  }
  for (std::size_t i = 0; i < tn; ++i) {
    host_t.data()[i] = t_at<T>(i);
    ref_t[i] = host_t.data()[i];
  }
  for (std::size_t i = 0; i < cn; ++i) {
    host_c.data()[i] = c_at<T>(i);
    ref_c[i] = host_c.data()[i];
  }
  for (std::size_t i = 0; i < wn; ++i) {
    host_w.data()[i] = T{0};
  }

  // Reference: overwrite ref_c with H*C / C*H (or the transpose), in host memory.
  const lapack_int info =
      ref_larfb(sd, tr, dc, sc, m, n, k, ref_v.data(), ldv, ref_t.data(), k, ref_c.data(), m);
  ASSERT_EQ(info, 0) << "side=" << sd << " trans=" << tr << " direct=" << dc << " storev=" << sc
                     << " m=" << m << " n=" << n << " k=" << k;

  auto d_v = to_device(handle, host_v, vn);
  auto d_t = to_device(handle, host_t, tn);
  auto d_c = to_device(handle, host_c, cn);
  auto d_w = to_device(handle, host_w, wn);

  const auto status = larfb<T>(blas, side, trans, direct, storev, m, n, k, d_v.data(), ldv,
                               d_t.data(), k, d_c.data(), m, d_w.data(), ldwork);
  const auto got = from_device(handle, d_c, cn);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS)
      << "side=" << sd << " trans=" << tr << " direct=" << dc << " storev=" << sc << " m=" << m
      << " n=" << n << " k=" << k;
  const T tol = factorization_tol<T>(frobenius_norm<T>(ref_c), static_cast<std::size_t>(m),
                                     static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < cn; ++i) {
    EXPECT_NEAR(got[i], ref_c[i], tol)
        << "side=" << sd << " trans=" << tr << " direct=" << dc << " storev=" << sc << " m=" << m
        << " n=" << n << " k=" << k << " i=" << i;
  }
}

// The (m, n, k) grid every case sweeps: a k == 1 base, a square V (the leading
// dimension of C equals k, so the rectangular-tail gemm is skipped), and
// non-square ones (k < m and k < n) that exercise the tail. k never exceeds the
// side's bound (m for L, n for R); the grid keeps k <= min(m, n) so one set
// serves both sides.
template<typename T>
void sweep(Side side, Trans trans, Direct direct, StoreV storev) {
  expect_matches_reference<T>(side, trans, direct, storev, 5, 4, 1);
  expect_matches_reference<T>(side, trans, direct, storev, 4, 4, 4);
  expect_matches_reference<T>(side, trans, direct, storev, 6, 5, 3);
  expect_matches_reference<T>(side, trans, direct, storev, 7, 6, 4);
}

#define LARFB_CASE(name, side, trans, direct, storev)                                              \
  TEST(LarfbOracleTests, name##Float) {                                                            \
    sweep<float>(side, trans, direct, storev);                                                     \
  }                                                                                                \
  TEST(LarfbOracleTests, name##Double) {                                                           \
    sweep<double>(side, trans, direct, storev);                                                    \
  }

LARFB_CASE(LeftNoTransForwardColumnwise, Side::L, Trans::N, Direct::F, StoreV::C)
LARFB_CASE(LeftTransForwardColumnwise, Side::L, Trans::T, Direct::F, StoreV::C)
LARFB_CASE(RightNoTransForwardColumnwise, Side::R, Trans::N, Direct::F, StoreV::C)
LARFB_CASE(RightTransForwardColumnwise, Side::R, Trans::T, Direct::F, StoreV::C)
LARFB_CASE(LeftNoTransBackwardColumnwise, Side::L, Trans::N, Direct::B, StoreV::C)
LARFB_CASE(RightTransBackwardColumnwise, Side::R, Trans::T, Direct::B, StoreV::C)
LARFB_CASE(LeftNoTransForwardRowwise, Side::L, Trans::N, Direct::F, StoreV::R)
LARFB_CASE(RightTransForwardRowwise, Side::R, Trans::T, Direct::F, StoreV::R)
LARFB_CASE(LeftNoTransBackwardRowwise, Side::L, Trans::N, Direct::B, StoreV::R)
LARFB_CASE(RightNoTransBackwardRowwise, Side::R, Trans::N, Direct::B, StoreV::R)
LARFB_CASE(LeftTransBackwardRowwise, Side::L, Trans::T, Direct::B, StoreV::R)
LARFB_CASE(RightTransBackwardColumnwiseAlt, Side::R, Trans::N, Direct::B, StoreV::C)

#undef LARFB_CASE

} // namespace
} // namespace calaman
