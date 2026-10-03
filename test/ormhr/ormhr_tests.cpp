// Oracle test for calaman.ormhr: the device multiply of a general matrix C by the
// orthogonal Q from ?gehrd -- Q*C, Q^T*C, C*Q, C*Q^T -- must agree with reference
// LAPACK (?ormhr) computed in the SAME precision on the host. ?ormhr has a LAPACKE
// C binding (LAPACKE_?ormhr), so the oracle calls it directly, column-major, as a
// faithful reference; the reflectors themselves come from LAPACKE_?gehrd on a
// separate nq-by-nq matrix (nq = m for Side::L, n for Side::R), so the suite feeds
// ormhr exactly the packed A/tau a real gehrd leaves behind.
//
// Both paths are the SAME ormqr apply over the SAME shifted reflectors, so only
// rounding (device FMA contraction, BLAS summation order) separates the numbers --
// the shared factorization tolerance absorbs it. Every case checks all four
// (Side in {L,R}) x (Trans in {N,T}) combinations element-wise against the oracle.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/tau/C/work on the device
// and runs the ormqr kernel, so `ctest -LE gpu` excludes it. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise, so
// its absence is a missing tier, not a silent pass (docs/architecture.md §3).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.common;
import calaman.ormhr;
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
using test::eps;
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

// Reference ?gehrd / ?ormhr via LAPACKE, column-major, same precision as the
// device path. gehrd writes the packed reflectors into a and the scalars into tau.
int ref_gehrd(int n, int ilo, int ihi, float *a, int lda, float *tau) {
  return LAPACKE_sgehrd(LAPACK_COL_MAJOR, n, ilo, ihi, a, lda, tau);
}
int ref_gehrd(int n, int ilo, int ihi, double *a, int lda, double *tau) {
  return LAPACKE_dgehrd(LAPACK_COL_MAJOR, n, ilo, ihi, a, lda, tau);
}
int ref_ormhr(char side, char trans, int m, int n, int ilo, int ihi, const float *a, int lda,
              const float *tau, float *c, int ldc) {
  return LAPACKE_sormhr(LAPACK_COL_MAJOR, side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc);
}
int ref_ormhr(char side, char trans, int m, int n, int ilo, int ihi, const double *a, int lda,
              const double *tau, double *c, int ldc) {
  return LAPACKE_dormhr(LAPACK_COL_MAJOR, side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc);
}

// One case: ormhr on the device must match reference ?ormhr for every (side,trans).
// nq = m for Side::L, n for Side::R; the reflectors A/tau come from a reference
// gehrd on a random nq-by-nq matrix over [ilo, ihi]. C is a random m-by-n matrix.
template<typename T>
void expect_matches_reference(Side side, int m, int n, int ilo, int ihi, unsigned seed) {
  const char cside = side == Side::L ? 'L' : 'R';
  const int nq = side == Side::L ? m : n;
  const int lda = std::max(nq, 1);
  const int ldc = std::max(m, 1);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);

  // Random nq-by-nq matrix reduced to Hessenberg: gehrd packs Q's reflectors into
  // a and the scalars into tau (length nq-1). The window [ilo, ihi] is passed
  // through unchanged, so ormhr receives exactly a real gehrd's output.
  std::vector<T> a(static_cast<std::size_t>(lda) * std::max(nq, 1));
  for (auto &x : a) {
    x = static_cast<T>(dist(rng));
  }
  std::vector<T> tau(static_cast<std::size_t>(std::max(nq - 1, 1)), T{0});
  const int gehrd_info = ref_gehrd(nq, ilo, ihi, a.data(), lda, tau.data());
  ASSERT_EQ(gehrd_info, 0) << "gehrd side=" << cside << " nq=" << nq;

  // Random C, shared across the four (side, trans) applies below.
  std::vector<T> c0(static_cast<std::size_t>(ldc) * std::max(n, 1));
  for (auto &x : c0) {
    x = static_cast<T>(dist(rng));
  }

  auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  auto d_a = to_device(handle, a);
  auto d_tau = to_device(handle, tau);

  const int lwork = ormhr_work_len<T>(solver, side, m, n, ilo, ihi, lda, ldc);
  std::vector<T> work_init(static_cast<std::size_t>(std::max(lwork, 1)), T{0});
  auto d_work = to_device(handle, work_init);
  auto d_info = to_device(handle, std::vector<int>{0});

  const T norm_c = frobenius_norm(c0);
  const T tol = factorization_tol<T>(std::max(norm_c, T{1}), static_cast<std::size_t>(nq),
                                     static_cast<std::size_t>(std::max(m, n)));

  for (const Trans trans : {Trans::N, Trans::T}) {
    const char ctrans = trans == Trans::N ? 'N' : 'T';

    // Reference ?ormhr on a fresh copy of C.
    std::vector<T> ref_c = c0;
    const int info = ref_ormhr(cside, ctrans, m, n, ilo, ihi, a.data(), lda, tau.data(),
                               ref_c.data(), ldc);
    ASSERT_EQ(info, 0) << "ormhr ref side=" << cside << " trans=" << ctrans;

    // Device ormhr on a fresh copy of C.
    auto d_c = to_device(handle, c0);
    const auto status =
        ormhr<T>(solver, side, trans, m, n, ilo, ihi, d_a.data(), lda, d_tau.data(), d_c.data(),
                 ldc, d_work.data(), lwork, d_info.data());
    ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS)
        << "side=" << cside << " trans=" << ctrans << " m=" << m << " n=" << n;

    const auto got_c = from_device(handle, d_c, static_cast<std::size_t>(ldc) * std::max(n, 1));
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        const std::size_t k = static_cast<std::size_t>(j) * ldc + static_cast<std::size_t>(i);
        EXPECT_NEAR(got_c[k], ref_c[k], tol)
            << "C(" << i << "," << j << ") side=" << cside << " trans=" << ctrans << " m=" << m
            << " n=" << n << " ilo=" << ilo << " ihi=" << ihi;
      }
    }
  }

  wwr::wwrsolverDnDestroy(solver);
}

} // namespace

// Square C, full window (ilo=1, ihi=nq): Q is the product of every reflector.
TEST(OrmhrOracleTests, SquareFloat) {
  expect_matches_reference<float>(Side::L, 5, 5, 1, 5, 1);
  expect_matches_reference<float>(Side::R, 5, 5, 1, 5, 2);
  expect_matches_reference<float>(Side::L, 8, 8, 1, 8, 3);
}
TEST(OrmhrOracleTests, SquareDouble) {
  expect_matches_reference<double>(Side::L, 6, 6, 1, 6, 10);
  expect_matches_reference<double>(Side::R, 6, 6, 1, 6, 11);
  expect_matches_reference<double>(Side::L, 12, 12, 1, 12, 12);
}

// Rectangular C: m != n exercises the two sides' different nq (m for L, n for R)
// and the shifted submatrix sizes. Q is still square (nq-by-nq).
TEST(OrmhrOracleTests, RectangularDouble) {
  expect_matches_reference<double>(Side::L, 9, 4, 1, 9, 20); // nq = 9
  expect_matches_reference<double>(Side::R, 4, 9, 1, 9, 21); // nq = 9
  expect_matches_reference<double>(Side::L, 7, 3, 1, 7, 22);
}
TEST(OrmhrOracleTests, RectangularFloat) {
  expect_matches_reference<float>(Side::R, 3, 7, 1, 7, 23);
}

// A windowed sub-block (ilo > 1 and/or ihi < nq), the gebal -> gehrd -> ormhr
// path: only reflectors ilo..ihi-1 contribute, exactly the shift ormhr performs.
TEST(OrmhrOracleTests, WindowedDouble) {
  expect_matches_reference<double>(Side::L, 8, 8, 2, 7, 30);
  expect_matches_reference<double>(Side::R, 8, 7, 2, 6, 31); // nq = n = 7
  expect_matches_reference<double>(Side::L, 10, 6, 2, 9, 32);
}
TEST(OrmhrOracleTests, WindowedFloat) {
  expect_matches_reference<float>(Side::L, 8, 8, 2, 7, 33);
}

// nh = ihi - ilo == 0 is a quick return: Q = I, C comes back unchanged. ihi == ilo
// leaves no reflectors; the reference ?ormhr agrees C is untouched.
TEST(OrmhrOracleTests, NoReflectorQuickReturn) {
  expect_matches_reference<double>(Side::L, 5, 5, 3, 3, 40); // empty window
  expect_matches_reference<double>(Side::L, 4, 4, 1, 1, 41); // ihi == ilo == 1
}

// Argument validation mirrors LAPACK's ?ormhr INFO < 0 contract: an illegal
// side/trans/m/n/ilo/ihi/lda/ldc returns wwrErrorInvalidValue and touches nothing.
TEST(OrmhrOracleTests, RejectsIllegalArguments) {
  auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  const int m = 5;
  const int n = 5; // Side::L => nq = m = 5
  auto d_a = to_device(handle, std::vector<double>(static_cast<std::size_t>(m) * m, 1.0));
  auto d_tau = to_device(handle, std::vector<double>(static_cast<std::size_t>(m - 1), 0.0));
  auto d_c = to_device(handle, std::vector<double>(static_cast<std::size_t>(m) * n, 1.0));
  auto d_work = to_device(handle, std::vector<double>(static_cast<std::size_t>(m), 0.0));
  auto d_info = to_device(handle, std::vector<int>{0});

  auto call = [&](Side side, Trans trans, int mm, int nn, int ilo, int ihi, int lda, int ldc) {
    return ormhr<double>(solver, side, trans, mm, nn, ilo, ihi, d_a.data(), lda, d_tau.data(),
                         d_c.data(), ldc, d_work.data(), m, d_info.data());
  };

  // Trans::C (not a real operation), m < 0, ilo = 0 (< 1), ihi = nq+1 (> nq),
  // lda < nq, ldc < m.
  EXPECT_EQ(call(Side::L, Trans::C, m, n, 1, m, m, m), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(Side::L, Trans::N, -1, n, 1, 1, m, m), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(Side::L, Trans::N, m, n, 0, m, m, m), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(Side::L, Trans::N, m, n, 1, m + 1, m, m), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(Side::L, Trans::N, m, n, 1, m, m - 1, m), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(Side::L, Trans::N, m, n, 1, m, m, m - 1), wwr::wwrErrorInvalidValue);

  wwr::wwrsolverDnDestroy(solver);
}

} // namespace calaman
