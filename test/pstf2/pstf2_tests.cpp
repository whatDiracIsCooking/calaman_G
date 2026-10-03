// Oracle test for calaman.pstf2 -- the level-2 pivoted Cholesky panel of a
// symmetric positive semidefinite matrix. LAPACKE ships no ?pstf2 binding, so
// the oracle is LAPACKE_?pstrf run on the WHOLE matrix: ?pstrf and ?pstf2
// produce the same unique complete-pivoting factorization, so the pivot
// permutation PIV, the computed RANK, and the backward error of the
// reconstructed P^T A P must all agree.
//
// A pivoted Cholesky's factor is unique on well-separated diagonals, so the
// robust invariant is the backward error ||P^T A P - U^H U|| / ||A|| rebuilt from
// the device's own factor and PIV, not a bitwise-equal factor. PIV and RANK are
// compared against the oracle directly (both make the same greedy choice).
//
// Three regimes exercise the branches the issue calls out: a full-rank SPD input
// (RANK == n, auto TOL), a genuinely rank-deficient PSD input B B^T of known rank
// r (the TOL / RANK / positive-INFO branch), and a NaN diagonal (the abort path;
// LAPACKE rejects NaN input via its nancheck, so only our graceful stop is
// asserted there). Each runs for both UPLO and both precisions.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/work on the device and
// runs the pivot and factor kernels, so `ctest -LE gpu` excludes it. Built only
// when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.pstf2;
import calaman.common;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;
import calaman.test.shared.tolerance;

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

// Reference pivoted Cholesky: LAPACKE_?pstrf over the whole matrix, column-major,
// with the same TOL the device call uses. Returns LAPACKE's info; *rank gets the
// reference rank.
int ref_pstrf(char uplo, int n, float *a, int lda, int *piv, int *rank, float tol) {
  lapack_int rk = 0;
  const int info = LAPACKE_spstrf(LAPACK_COL_MAJOR, uplo, n, a, lda, piv, &rk, tol);
  *rank = static_cast<int>(rk);
  return info;
}
int ref_pstrf(char uplo, int n, double *a, int lda, int *piv, int *rank, double tol) {
  lapack_int rk = 0;
  const int info = LAPACKE_dpstrf(LAPACK_COL_MAJOR, uplo, n, a, lda, piv, &rk, tol);
  *rank = static_cast<int>(rk);
  return info;
}

// A random symmetric PSD matrix A = B B^T, B being n-by-k, stored FULL column-major
// (both triangles filled, so pstf2 and the oracle read a consistent matrix). With
// k < n the rank is k; the optional diagonal `shift` lifts it to full-rank SPD.
template<typename T>
std::vector<T> symmetric_psd(int n, int k, unsigned seed, T shift) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> b(static_cast<std::size_t>(n) * k);
  for (auto &x : b) {
    x = static_cast<T>(dist(rng));
  }
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      T s{};
      for (int c = 0; c < k; ++c) {
        s += b[static_cast<std::size_t>(c) * n + i] * b[static_cast<std::size_t>(c) * n + j];
      }
      a[static_cast<std::size_t>(j) * n + i] = s;
    }
    a[static_cast<std::size_t>(i) * n + i] += shift;
  }
  return a;
}

// Reconstruct M = U^H U (upper) or L L^H (lower) from the device's packed factor,
// with factor rows/columns at or beyond `rank` treated as zero (the trailing
// factor the rank-revealing stop zeroed). M is the factorization of P^T A P.
template<typename T>
std::vector<T> reconstruct(const std::vector<T> &fac, bool upper, int n, int lda, int rank) {
  std::vector<T> m(static_cast<std::size_t>(n) * n, T{0});
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      T s{};
      const int kmax = std::min(std::min(r, c) + 1, rank);
      for (int kk = 0; kk < kmax; ++kk) {
        if (upper) {
          // U[k,r], U[k,c] sit in the stored upper triangle, column-major.
          s += fac[static_cast<std::size_t>(r) * lda + kk] *
               fac[static_cast<std::size_t>(c) * lda + kk];
        } else {
          // L[r,k], L[c,k] sit in the stored lower triangle, column-major.
          s += fac[static_cast<std::size_t>(kk) * lda + r] *
               fac[static_cast<std::size_t>(kk) * lda + c];
        }
      }
      m[static_cast<std::size_t>(c) * n + r] = s;
    }
  }
  return m;
}

// pstf2 on the device must match ?pstrf for one symmetric PSD matrix.
template<typename T>
void expect_matches_reference(Uplo uplo, int n, int k, unsigned seed, T shift, T tol,
                              int expected_rank) {
  const int lda = n;
  const char cuplo = uplo == Uplo::U ? 'U' : 'L';
  const auto a_orig = symmetric_psd<T>(n, k, seed, shift);

  // Reference ?pstrf on a copy.
  std::vector<T> ref_a = a_orig;
  std::vector<int> ref_piv(static_cast<std::size_t>(n), 0);
  int ref_rank = 0;
  const int ref_info = ref_pstrf(cuplo, n, ref_a.data(), lda, ref_piv.data(), &ref_rank, tol);
  ASSERT_GE(ref_info, 0) << "LAPACKE rejected the input (info=" << ref_info << ")";
  ASSERT_EQ(ref_rank, expected_rank) << "oracle rank n=" << n << " k=" << k;

  // Device pstf2.
  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = to_device(handle, a_orig);
  std::vector<T> work_init(static_cast<std::size_t>(n) + 3, T{0});
  auto d_work = to_device(handle, work_init);

  std::vector<int> piv(static_cast<std::size_t>(n), 0);
  int rank = 0;
  int info = 0;
  const auto status = pstf2<T>(blas, uplo, n, d_a.data(), lda, piv.data(), &rank, &info, tol,
                               d_work.data());
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " k=" << k;
  wwr::wwrblasDestroy(blas);

  const auto got_a = from_device(handle, d_a, static_cast<std::size_t>(lda) * n);

  // 1. Rank and the rank-revealing info convention.
  EXPECT_EQ(rank, ref_rank) << "rank n=" << n << " k=" << k << " uplo=" << cuplo;
  EXPECT_EQ(info, rank == n ? 0 : rank + 1) << "info n=" << n << " k=" << k;

  // 2. Pivot permutation matches the oracle's (same greedy choice).
  for (int j = 0; j < n; ++j) {
    EXPECT_EQ(piv[static_cast<std::size_t>(j)], ref_piv[static_cast<std::size_t>(j)])
        << "piv[" << j << "] n=" << n << " k=" << k << " uplo=" << cuplo << " seed=" << seed;
  }

  // 3. Backward error of the reconstructed P^T A P. A factor beyond rank was
  // zeroed, so for an exactly-rank-r input the dropped trailing Schur complement
  // is ~roundoff and the reconstruction is exact to tolerance. P^T A P is the
  // symmetric original permuted by the device's own 1-based piv.
  const T norm_a = frobenius_norm(a_orig);
  const T tol_recon = T{8} * factorization_tol<T>(norm_a, static_cast<std::size_t>(n),
                                                  static_cast<std::size_t>(n));
  const auto m = reconstruct(got_a, uplo == Uplo::U, n, lda, rank);
  std::vector<T> residual(static_cast<std::size_t>(n) * n, T{0});
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      const int pr = piv[static_cast<std::size_t>(r)] - 1;
      const int pc = piv[static_cast<std::size_t>(c)] - 1;
      const T pap = a_orig[static_cast<std::size_t>(pc) * n + pr];
      residual[static_cast<std::size_t>(c) * n + r] = pap - m[static_cast<std::size_t>(c) * n + r];
    }
  }
  EXPECT_LE(frobenius_norm(residual), tol_recon)
      << "backward error n=" << n << " k=" << k << " uplo=" << cuplo << " seed=" << seed;
}

// Full-rank SPD: A = B B^T + I, auto TOL (-1), RANK must be n and INFO 0.
TEST(Pstf2OracleTests, FullRankDouble) {
  expect_matches_reference<double>(Uplo::L, 6, 12, 1, 1.0, -1.0, 6);
  expect_matches_reference<double>(Uplo::U, 6, 12, 1, 1.0, -1.0, 6);
  expect_matches_reference<double>(Uplo::L, 10, 20, 2, 1.0, -1.0, 10);
  expect_matches_reference<double>(Uplo::U, 10, 20, 2, 1.0, -1.0, 10);
}
TEST(Pstf2OracleTests, FullRankFloat) {
  expect_matches_reference<float>(Uplo::L, 6, 12, 3, 1.0f, -1.0f, 6);
  expect_matches_reference<float>(Uplo::U, 6, 12, 3, 1.0f, -1.0f, 6);
  expect_matches_reference<float>(Uplo::L, 8, 16, 4, 1.0f, -1.0f, 8);
}

// Rank-deficient PSD: A = B B^T with B n-by-r of known rank r (no shift). An
// explicit TOL sits between the ~roundoff trailing tail and the O(1) real pivots,
// so RANK must be r and INFO = r + 1.
TEST(Pstf2OracleTests, RankDeficientDouble) {
  expect_matches_reference<double>(Uplo::L, 8, 3, 10, 0.0, 1e-7, 3);
  expect_matches_reference<double>(Uplo::U, 8, 3, 10, 0.0, 1e-7, 3);
  expect_matches_reference<double>(Uplo::L, 10, 5, 11, 0.0, 1e-7, 5);
  expect_matches_reference<double>(Uplo::U, 10, 5, 11, 0.0, 1e-7, 5);
}
TEST(Pstf2OracleTests, RankDeficientFloat) {
  expect_matches_reference<float>(Uplo::L, 8, 3, 12, 0.0f, 1e-3f, 3);
  expect_matches_reference<float>(Uplo::U, 8, 3, 12, 0.0f, 1e-3f, 3);
}

// NaN abort: a NaN on the first diagonal. LAPACKE rejects NaN input (its
// nancheck returns a negative info), so only the device's graceful stop is
// asserted -- rank 0, info 1 (== rank + 1), Status success.
template<typename T>
void expect_nan_abort(Uplo uplo, int n, unsigned seed) {
  const int lda = n;
  auto a = symmetric_psd<T>(n, 2 * n, seed, T{1});
  a[0] = std::numeric_limits<T>::quiet_NaN(); // A(0,0)

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = to_device(handle, a);
  std::vector<T> work_init(static_cast<std::size_t>(n) + 3, T{0});
  auto d_work = to_device(handle, work_init);

  std::vector<int> piv(static_cast<std::size_t>(n), 0);
  int rank = -1;
  int info = -1;
  const auto status = pstf2<T>(blas, uplo, n, d_a.data(), lda, piv.data(), &rank, &info, T{-1},
                               d_work.data());
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(rank, 0) << "NaN abort rank n=" << n;
  EXPECT_EQ(info, 1) << "NaN abort info n=" << n;
}

TEST(Pstf2OracleTests, NanAbortDouble) {
  expect_nan_abort<double>(Uplo::L, 6, 20);
  expect_nan_abort<double>(Uplo::U, 6, 21);
}
TEST(Pstf2OracleTests, NanAbortFloat) {
  expect_nan_abort<float>(Uplo::L, 5, 22);
}

// Smallest shapes: n == 1 (one step, no trailing update) and n == 2.
TEST(Pstf2OracleTests, SmallShapes) {
  expect_matches_reference<double>(Uplo::L, 1, 2, 30, 1.0, -1.0, 1);
  expect_matches_reference<double>(Uplo::U, 1, 2, 30, 1.0, -1.0, 1);
  expect_matches_reference<double>(Uplo::L, 2, 4, 31, 1.0, -1.0, 2);
  expect_matches_reference<double>(Uplo::U, 2, 4, 31, 1.0, -1.0, 2);
}

} // namespace
} // namespace calaman
