// Oracle test for calaman.larfg: the generated Householder reflector must
// agree with the reference LAPACK -- LAPACKE_slarfg / LAPACKE_dlarfg -- computed
// in the SAME precision on the host. larfg builds H = I - tau*v*v^T with
// v = [1; v_tail] so that H*[alpha; x] = [beta; 0]; the device path must produce
// the same tau, the same beta (overwriting alpha), and the same scaled tail v.
//
// larfg OVERWRITES alpha (with beta) and x (with v_tail), so every case uploads a
// fresh [alpha; x] and reads both back. The handle is left in its default (host)
// pointer mode, which is what larfg requires: nrm2's result and scal's scale are
// host-side, and the reduction blocks.
//
// Edge cases the acceptance calls out, each its own expectation below:
//   - zero tail  -> tau == 0, beta == alpha, tail untouched (identity reflector)
//   - n == 1     -> empty tail, same identity case
//   - both signs -> alpha > 0 and alpha < 0, pinning beta = -sign(alpha)*||v||
//
// REQUIRES_GPU (see CMakeLists.txt): every non-degenerate case allocates device
// memory and runs the BLAS routines, so the suite is excluded by `ctest -LE gpu`.
// Built only when calaman::lapack_reference exists; its CMakeLists.txt returns
// early otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.larfg;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceHandle, DeviceAbort>;

/// @brief Upload `host` to a fresh device buffer on `handle`'s stream
template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const HostBuffer<T> &host,
                          std::size_t n) {
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, host, handle->stream().get());
  return device;
}

// The reference oracle: reference LAPACK's ?larfg, dispatched by type, same
// precision as the device path. alpha is overwritten with beta, x with v_tail.
void ref_larfg(int n, float *alpha, float *x, int incx, float *tau) {
  LAPACKE_slarfg(n, alpha, x, incx, tau);
}
void ref_larfg(int n, double *alpha, double *x, int incx, double *tau) {
  LAPACKE_dlarfg(n, alpha, x, incx, tau);
}

/// @brief A relative tolerance tight enough to catch a wrong sign, tau, or scale
///        yet generous for the one sqrt + divide done in each path's summation
template<typename T>
T larfg_tol(T ref) {
  const T a = ref < T{0} ? -ref : ref;
  return (T{64} * eps<T>()) * (a + T{1});
}

/// @brief One (n, alpha, seed) case: larfg on the device must match the reference
///
/// Fills a fresh [alpha; x] (x has n-1 elements), runs larfg on the host oracle
/// and on the device, then checks tau, beta (the overwritten alpha) and every
/// element of v_tail agree. alpha is the caller's chosen sign; the tail is a
/// small deterministic ramp so float and double both stay well inside the
/// mantissa and the only divergence is summation order.
template<typename T>
void expect_matches_reference(int n, T alpha0, unsigned seed) {
  ASSERT_GE(n, 1);
  const int tail = n - 1;
  const std::size_t tn = static_cast<std::size_t>(tail);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  // Reference and device share identical inputs. Keep the originals in `input`;
  // ref_x is a copy the reference OVERWRITES in place with its v_tail.
  std::vector<T> input(tn);
  for (std::size_t i = 0; i < tn; ++i) {
    input[i] = static_cast<T>(dist(rng));
  }
  std::vector<T> ref_x = input;
  T ref_alpha = alpha0;
  T ref_tau = T{0};
  ref_larfg(n, &ref_alpha, ref_x.data(), 1, &ref_tau);
  const T ref_beta = ref_alpha; // overwritten in place

  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_alpha(1);
  host_alpha.data()[0] = alpha0;
  auto d_alpha = to_device(handle, host_alpha, 1);

  // The tail buffer is size max(1,tail) so n == 1 still allocates a valid buffer;
  // larfg ignores it when tail == 0.
  const std::size_t alloc = tn == 0 ? 1 : tn;
  HostBuffer<T> host_x(alloc);
  for (std::size_t i = 0; i < tn; ++i) {
    host_x.data()[i] = input[i]; // the ORIGINAL x, not the reference's v_tail
  }
  auto d_x = to_device(handle, host_x, alloc);

  T got_tau = static_cast<T>(-12345);
  T got_beta = static_cast<T>(-12345);
  const auto status =
      larfg<T>(blas, n, d_alpha.data(), d_x.data(), 1, &got_tau, &got_beta);
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " alpha=" << alpha0;

  // Read beta (the overwritten alpha) and the scaled tail back from the device.
  HostBuffer<T> out_alpha(1);
  wwr::extension::copy(out_alpha, d_alpha, handle->stream().get());
  HostBuffer<T> out_x(alloc);
  wwr::extension::copy(out_x, d_x, handle->stream().get());
  ASSERT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);
  wwr::wwrblasDestroy(blas);

  const T tol = larfg_tol(ref_beta);
  EXPECT_NEAR(got_tau, ref_tau, tol) << "tau n=" << n << " alpha=" << alpha0;
  EXPECT_NEAR(got_beta, ref_beta, tol) << "beta(host) n=" << n << " alpha=" << alpha0;
  EXPECT_NEAR(out_alpha.data()[0], ref_beta, tol)
      << "beta(device) n=" << n << " alpha=" << alpha0;
  for (std::size_t i = 0; i < tn; ++i) {
    EXPECT_NEAR(out_x.data()[i], ref_x[i], tol)
        << "v[" << i << "] n=" << n << " alpha=" << alpha0;
  }
}

TEST(LarfgOracleTests, MatchesReferenceFloat) {
  expect_matches_reference<float>(7, 2.5F, 1);
  expect_matches_reference<float>(7, -2.5F, 2);  // negative alpha: beta flips sign
  expect_matches_reference<float>(64, 0.3F, 3);
  expect_matches_reference<float>(64, -4.0F, 4);
  expect_matches_reference<float>(128, 1.0F, 5);
}

TEST(LarfgOracleTests, MatchesReferenceDouble) {
  expect_matches_reference<double>(7, 2.5, 11);
  expect_matches_reference<double>(7, -2.5, 12); // negative alpha: beta flips sign
  expect_matches_reference<double>(64, 0.3, 13);
  expect_matches_reference<double>(64, -4.0, 14);
  expect_matches_reference<double>(500, 3.0, 15);
}

// A zero tail is already [alpha; 0]: the reflector is the identity, so tau == 0,
// beta == alpha, and the tail (a sentinel) must survive untouched.
template<typename T>
void expect_zero_tail_is_identity(int n, T alpha0) {
  ASSERT_GE(n, 2);
  const int tail = n - 1;
  const std::size_t tn = static_cast<std::size_t>(tail);

  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_alpha(1);
  host_alpha.data()[0] = alpha0;
  auto d_alpha = to_device(handle, host_alpha, 1);

  HostBuffer<T> host_x(tn);
  for (std::size_t i = 0; i < tn; ++i) {
    host_x.data()[i] = T{0}; // zero tail
  }
  auto d_x = to_device(handle, host_x, tn);

  T got_tau = static_cast<T>(-12345);
  T got_beta = static_cast<T>(-12345);
  const auto status =
      larfg<T>(blas, n, d_alpha.data(), d_x.data(), 1, &got_tau, &got_beta);
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> out_alpha(1);
  wwr::extension::copy(out_alpha, d_alpha, handle->stream().get());
  HostBuffer<T> out_x(tn);
  wwr::extension::copy(out_x, d_x, handle->stream().get());
  ASSERT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(got_tau, T{0});           // identity reflector
  EXPECT_EQ(got_beta, alpha0);        // beta == alpha
  EXPECT_EQ(out_alpha.data()[0], alpha0); // alpha untouched on the device
  for (std::size_t i = 0; i < tn; ++i) {
    EXPECT_EQ(out_x.data()[i], T{0}); // tail untouched
  }
}

TEST(LarfgOracleTests, ZeroTailIsIdentity) {
  expect_zero_tail_is_identity<float>(5, 3.0F);
  expect_zero_tail_is_identity<float>(5, -3.0F);
  expect_zero_tail_is_identity<double>(5, 3.0);
  expect_zero_tail_is_identity<double>(5, -3.0);
}

// n == 1 is the empty-tail case: no x at all, so the reflector is the identity.
// larfg must leave alpha as beta and report tau == 0 without touching the tail.
template<typename T>
void expect_single_element(T alpha0) {
  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_alpha(1);
  host_alpha.data()[0] = alpha0;
  auto d_alpha = to_device(handle, host_alpha, 1);
  HostBuffer<T> host_x(1); // never read (tail == 0)
  host_x.data()[0] = static_cast<T>(-777);
  auto d_x = to_device(handle, host_x, 1);

  T got_tau = static_cast<T>(-12345);
  T got_beta = static_cast<T>(-12345);
  const auto status =
      larfg<T>(blas, 1, d_alpha.data(), d_x.data(), 1, &got_tau, &got_beta);
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> out_alpha(1);
  wwr::extension::copy(out_alpha, d_alpha, handle->stream().get());
  ASSERT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);
  wwr::wwrblasDestroy(blas);

  // Cross-check against the reference's n == 1 behaviour.
  T ref_alpha = alpha0;
  T ref_tau = T{0};
  ref_larfg(1, &ref_alpha, host_x.data(), 1, &ref_tau);

  EXPECT_EQ(got_tau, T{0});
  EXPECT_EQ(got_tau, ref_tau);
  EXPECT_EQ(got_beta, alpha0);
  EXPECT_EQ(got_beta, ref_alpha);
  EXPECT_EQ(out_alpha.data()[0], alpha0);
}

TEST(LarfgOracleTests, SingleElement) {
  expect_single_element<float>(4.0F);
  expect_single_element<float>(-4.0F);
  expect_single_element<double>(4.0);
  expect_single_element<double>(-4.0);
}

TEST(LarfgOracleTests, NonPositiveNIsNoop) {
  // n <= 0: larfg enqueues nothing, returns success, and leaves tau/beta
  // untouched -- the sentinels must survive.
  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<double> host_alpha(1);
  host_alpha.data()[0] = 2.0;
  auto d_alpha = to_device(handle, host_alpha, 1);
  HostBuffer<double> host_x(1);
  host_x.data()[0] = 1.0;
  auto d_x = to_device(handle, host_x, 1);

  double tau = -12345.0;
  double beta = -54321.0;
  const auto status = larfg<double>(blas, 0, d_alpha.data(), d_x.data(), 1, &tau, &beta);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_DOUBLE_EQ(tau, -12345.0);  // untouched
  EXPECT_DOUBLE_EQ(beta, -54321.0); // untouched
}

} // namespace
} // namespace calaman
