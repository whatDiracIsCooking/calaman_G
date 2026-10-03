// Oracle test for calaman.orghr: the device generation of the orthogonal Q from
// ?gehrd's Householder reflectors must agree with reference LAPACK -- ?orghr --
// computed in the SAME precision on the host. The reflectors are an INPUT (as in
// LAPACK's standalone ?orghr), produced here by reference ?gehrd, so this suite
// does not depend on a device ?gehrd existing.
//
// ?orghr and ?gehrd both have LAPACKE C bindings, so the oracle calls
// LAPACKE_?gehrd / LAPACKE_?orghr directly (LAPACK_COL_MAJOR), the laqp2 / geqp3
// pattern -- no Fortran-symbol route needed. Both the device path and the oracle
// finish with ?orgqr on the SAME shifted panel, so for well-scaled inputs only
// rounding separates the numbers; the shared factorization tolerance absorbs it.
//
// Two checks per case: Q matches the reference element-wise (same deterministic
// ?orgqr on the same shuffled reflectors), and an independent invariant that Q is
// genuinely orthogonal -- ||Q^T Q - I||_F -- which catches a self-consistent but
// non-orthogonal output. Cases cover square float/double at several orders, a
// windowed [ilo,ihi] sub-block (the gebal -> gehrd -> orghr path), the n <= 1 and
// empty-window quick returns, and the argument-validation contract.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the panel on the device and
// runs the shuffle kernel plus wwr::orgqr, so `ctest -LE gpu` excludes it. Built
// only when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.solver;
import wwr.extension.memory_buffer;
import calaman.orghr;
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

// Reference ?gehrd / ?orghr via LAPACKE (column-major), same precision as device.
lapack_int ref_gehrd(int n, int ilo, int ihi, float *a, int lda, float *tau) {
  return LAPACKE_sgehrd(LAPACK_COL_MAJOR, n, ilo, ihi, a, lda, tau);
}
lapack_int ref_gehrd(int n, int ilo, int ihi, double *a, int lda, double *tau) {
  return LAPACKE_dgehrd(LAPACK_COL_MAJOR, n, ilo, ihi, a, lda, tau);
}
lapack_int ref_orghr(int n, int ilo, int ihi, float *a, int lda, const float *tau) {
  return LAPACKE_sorghr(LAPACK_COL_MAJOR, n, ilo, ihi, a, lda, tau);
}
lapack_int ref_orghr(int n, int ilo, int ihi, double *a, int lda, const double *tau) {
  return LAPACKE_dorghr(LAPACK_COL_MAJOR, n, ilo, ihi, a, lda, tau);
}

// ||Q^T Q - I||_F for an n-by-n column-major Q with leading dimension n.
template<typename T>
T orthogonality_residual(const std::vector<T> &q, int n) {
  const auto ld = static_cast<std::size_t>(n);
  std::vector<T> r(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      T s{};
      for (int k = 0; k < n; ++k) {
        s += q[static_cast<std::size_t>(i) * ld + k] * q[static_cast<std::size_t>(j) * ld + k];
      }
      r[static_cast<std::size_t>(j) * ld + i] = s - (i == j ? T{1} : T{0});
    }
  }
  return frobenius_norm(r);
}

// One case: calaman::orghr on the device must match reference ?orghr for the Q of
// a random n-by-n A reduced over the window [ilo, ihi].
template<typename T>
void expect_matches_reference(int n, int ilo, int ihi, unsigned seed) {
  const int lda = n;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);

  std::vector<T> a(static_cast<std::size_t>(lda) * n);
  for (auto &x : a) {
    x = static_cast<T>(dist(rng));
  }

  // ?gehrd assumes A is already upper triangular outside the window [ilo, ihi]
  // (the block structure gebal leaves). Zero the strictly-below-diagonal entries
  // there, as the gehd2 suite does. For the full window this clears nothing.
  for (int j = 0; j < n; ++j) {
    for (int i = j + 1; i < n; ++i) {
      if (j < ilo - 1 || i > ihi - 1) {
        a[static_cast<std::size_t>(j) * lda + i] = T{0};
      }
    }
  }

  // Reference ?gehrd -> packed reflectors + tau (length max(n-1,1)).
  std::vector<T> packed = a;
  std::vector<T> tau(static_cast<std::size_t>(std::max(n - 1, 1)), T{0});
  ASSERT_EQ(ref_gehrd(n, ilo, ihi, packed.data(), lda, tau.data()), 0)
      << "gehrd n=" << n << " ilo=" << ilo << " ihi=" << ihi;

  // Oracle ?orghr on a copy of the packed reflectors -> Q.
  std::vector<T> ref_q = packed;
  ASSERT_EQ(ref_orghr(n, ilo, ihi, ref_q.data(), lda, tau.data()), 0)
      << "orghr n=" << n << " ilo=" << ilo << " ihi=" << ihi;

  // Device orghr over the SAME packed reflectors and tau.
  auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  auto d_a = to_device(handle, packed);
  auto d_tau = to_device(handle, tau);
  auto d_info = to_device(handle, std::vector<int>(1, -1));
  const std::size_t bytes = orghr_bufferSize<T>(solver, n, lda, ilo, ihi);
  auto d_work = to_device(handle, std::vector<std::byte>(bytes == 0 ? 1 : bytes, std::byte{0}));

  const auto status = orghr<T>(solver, n, ilo, ihi, d_a.data(), lda, d_tau.data(),
                               d_work.data(), bytes, d_info.data());
  ASSERT_TRUE(status.ok()) << "orghr status " << status.name() << " n=" << n << " ilo=" << ilo
                           << " ihi=" << ihi;

  const auto got_q = from_device(handle, d_a, static_cast<std::size_t>(lda) * n);
  const auto got_info = from_device(handle, d_info, 1);
  wwr::wwrsolverDnDestroy(solver);
  EXPECT_EQ(got_info[0], 0) << "devInfo n=" << n << " ilo=" << ilo << " ihi=" << ihi;

  const T norm = frobenius_norm(ref_q);
  const T tol =
      factorization_tol<T>(norm, static_cast<std::size_t>(n), static_cast<std::size_t>(n));

  // 1. Q matches the reference element-wise (same ?orgqr on the same shuffle).
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * lda + static_cast<std::size_t>(i);
      EXPECT_NEAR(got_q[k], ref_q[k], tol)
          << "Q(" << i << "," << j << ") n=" << n << " ilo=" << ilo << " ihi=" << ihi;
    }
  }

  // 2. Independent invariant: Q is orthogonal.
  EXPECT_LE(orthogonality_residual(got_q, n), tol)
      << "orthogonality n=" << n << " ilo=" << ilo << " ihi=" << ihi;
}

} // namespace

TEST(OrghrOracleTests, SquareFloat) {
  expect_matches_reference<float>(4, 1, 4, 1);
  expect_matches_reference<float>(7, 1, 7, 2);
}
TEST(OrghrOracleTests, SquareDouble) {
  expect_matches_reference<double>(4, 1, 4, 3);
  expect_matches_reference<double>(7, 1, 7, 4);
  expect_matches_reference<double>(12, 1, 12, 5);
  expect_matches_reference<double>(16, 1, 16, 6);
}

// A windowed sub-block: ilo > 1 and ihi < n, the gebal -> gehrd -> orghr path.
// The leading ilo and trailing n-ihi rows/columns of Q must come back as the
// identity, which the element-wise comparison pins.
TEST(OrghrOracleTests, WindowedDouble) {
  expect_matches_reference<double>(8, 2, 7, 10);
  expect_matches_reference<double>(10, 3, 8, 11);
}
TEST(OrghrOracleTests, WindowedFloat) {
  expect_matches_reference<float>(8, 2, 7, 12);
}

// n <= 1, and an empty window (ihi == ilo): Q is the identity, no reflectors.
TEST(OrghrOracleTests, IdentityQuickReturn) {
  expect_matches_reference<double>(1, 1, 1, 20);
  expect_matches_reference<double>(5, 3, 3, 21); // nh = 0: empty window
}

// Argument validation mirrors LAPACK's INFO < 0 contract: an illegal ilo/ihi/lda
// or a null pointer / undersized buffer returns a non-ok Status and touches
// nothing.
TEST(OrghrOracleTests, RejectsIllegalArguments) {
  auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  const int n = 5;
  auto d_a = to_device(handle, std::vector<double>(static_cast<std::size_t>(n) * n, 1.0));
  auto d_tau = to_device(handle, std::vector<double>(static_cast<std::size_t>(n - 1), 0.0));
  auto d_info = to_device(handle, std::vector<int>(1, 0));
  const std::size_t bytes = orghr_bufferSize<double>(solver, n, n, 1, n);
  auto d_work = to_device(handle, std::vector<std::byte>(bytes == 0 ? 1 : bytes, std::byte{0}));

  // ilo = 0 (< 1), ihi = n+1 (> n), lda < n.
  EXPECT_FALSE(orghr<double>(solver, n, 0, n, d_a.data(), n, d_tau.data(), d_work.data(), bytes,
                             d_info.data())
                   .ok());
  EXPECT_FALSE(orghr<double>(solver, n, 1, n + 1, d_a.data(), n, d_tau.data(), d_work.data(),
                             bytes, d_info.data())
                   .ok());
  EXPECT_FALSE(orghr<double>(solver, n, 1, n, d_a.data(), n - 1, d_tau.data(), d_work.data(),
                             bytes, d_info.data())
                   .ok());
  // Null output / undersized buffer.
  EXPECT_FALSE(orghr<double>(solver, n, 1, n, nullptr, n, d_tau.data(), d_work.data(), bytes,
                             d_info.data())
                   .ok());
  EXPECT_FALSE(orghr<double>(solver, n, 1, n, d_a.data(), n, d_tau.data(), d_work.data(), 0,
                             d_info.data())
                   .ok());

  wwr::wwrsolverDnDestroy(solver);
}

} // namespace calaman
