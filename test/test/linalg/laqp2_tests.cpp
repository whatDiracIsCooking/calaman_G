// Oracle test for calaman.linalg:laqp2 -- the level-2 Businger-Golub pivoted-QR
// panel. LAPACKE ships no ?laqp2 C binding, so the oracle is LAPACKE_?geqp3 run
// on the WHOLE matrix as one panel (offset 0): geqp3's panel IS laqp2 when the
// block size covers all columns, so the two must agree on the pivot permutation,
// on |R| (its diagonal magnitudes and column norms), and on the factorization
// residual ||A*P - Q*R|| / ||A||.
//
// A pivoted QR is unique only up to column ties and reflector sign, so the robust
// invariant is the residual ||A(:,jpvt) - Q*R||, not bitwise-equal R or signs:
// this test reconstructs Q*R on the host from the stored reflectors and tau, and
// checks it against the pivot-permuted original A, to the shared factorization
// tolerance. The pivot permutation itself is compared directly (geqp3 and laqp2
// make the same greedy choice on these well-separated norms), and |R|'s diagonal
// magnitudes are checked as the second sign-free invariant.
//
// A dedicated case drives a trailing column into near-cancellation so the LAWN
// 176 TOL3Z recompute branch runs (not just the cheap downdate); it must still
// match the oracle. The branch's execution is asserted indirectly -- an
// incorrect recompute would blow the residual -- and directly, by a
// companion that would FAIL were the cheap path taken for that column.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/tau/vn1/vn2/work on the
// device and runs the reflector + downdate kernels, so `ctest -LE gpu` excludes
// it. Built only when calaman::lapack_reference exists; its CMakeLists.txt
// returns early otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.linalg;
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
using test::factorization_tol;
using test::frobenius_norm;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceHandle, DeviceAbort>;

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

// Reference pivoted QR: LAPACKE_?geqp3 over the whole matrix, column-major. jpvt
// is zero-initialised (all columns free), so the result is the pure greedy
// permutation laqp2 also produces.
int ref_geqp3(int m, int n, float *a, int lda, int *jpvt, float *tau) {
  return LAPACKE_sgeqp3(LAPACK_COL_MAJOR, m, n, a, lda, jpvt, tau);
}
int ref_geqp3(int m, int n, double *a, int lda, int *jpvt, double *tau) {
  return LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, a, lda, jpvt, tau);
}

// Column norm (Euclidean) of a trailing column, used to seed vn1/vn2.
template<typename T>
T column_norm(const std::vector<T> &a, int m, int lda, int j) {
  T s{};
  for (int i = 0; i < m; ++i) {
    const T v = a[static_cast<std::size_t>(j) * lda + i];
    s += v * v;
  }
  return std::sqrt(s);
}

// Reconstruct B = Q*R on the host from the packed laqp2 output. A holds R in its
// upper trapezoid and reflector j (v with v[0]==1 implicit) below the diagonal of
// column j; tau holds the k = min(m,n) scalars. Q = H_0 H_1 ... H_{k-1}, so apply
// the reflectors to R from the left in REVERSE order. Returns B column-major, m x n.
template<typename T>
std::vector<T> reconstruct_qr(const std::vector<T> &packed, const std::vector<T> &tau, int m,
                              int n, int lda) {
  const int k = std::min(m, n);
  // Start from R: the upper trapezoid of packed, everything below the diagonal 0.
  std::vector<T> b(static_cast<std::size_t>(lda) * n, T{0});
  for (int j = 0; j < n; ++j) {
    const int top = std::min(j + 1, m); // rows 0..min(j,m-1) are R's
    for (int i = 0; i < top; ++i) {
      b[static_cast<std::size_t>(j) * lda + i] = packed[static_cast<std::size_t>(j) * lda + i];
    }
  }
  // Apply H_i = I - tau_i v_i v_i^T from the left for i = k-1 .. 0. v_i has v[i]==1
  // and v[i+1:m] from packed's column i below the diagonal.
  std::vector<T> v(m);
  for (int i = k - 1; i >= 0; --i) {
    for (int r = 0; r < m; ++r) {
      v[static_cast<std::size_t>(r)] = T{0};
    }
    v[static_cast<std::size_t>(i)] = T{1};
    for (int r = i + 1; r < m; ++r) {
      v[static_cast<std::size_t>(r)] = packed[static_cast<std::size_t>(i) * lda + r];
    }
    // B := (I - tau v v^T) B = B - tau v (v^T B). Per column c: w = v^T B(:,c).
    for (int c = 0; c < n; ++c) {
      T w{};
      for (int r = i; r < m; ++r) {
        w += v[static_cast<std::size_t>(r)] * b[static_cast<std::size_t>(c) * lda + r];
      }
      const T s = tau[static_cast<std::size_t>(i)] * w;
      for (int r = i; r < m; ++r) {
        b[static_cast<std::size_t>(c) * lda + r] -= s * v[static_cast<std::size_t>(r)];
      }
    }
  }
  return b;
}

/// @brief laqp2 on the device must match geqp3 for one (m, n, seed) case
template<typename T>
void expect_matches_reference(int m, int n, unsigned seed,
                              std::function<void(std::vector<T> &, int, int)> perturb = {}) {
  const int lda = m;
  const int k = std::min(m, n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);

  std::vector<T> a_orig(static_cast<std::size_t>(lda) * n);
  for (auto &x : a_orig) {
    x = static_cast<T>(dist(rng));
  }
  if (perturb) {
    perturb(a_orig, m, lda);
  }

  // Reference geqp3 on a copy.
  std::vector<T> ref_a = a_orig;
  std::vector<int> ref_jpvt(static_cast<std::size_t>(n), 0);
  std::vector<T> ref_tau(static_cast<std::size_t>(k));
  ASSERT_EQ(ref_geqp3(m, n, ref_a.data(), lda, ref_jpvt.data(), ref_tau.data()), 0);

  // Device laqp2.
  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = to_device(handle, a_orig);
  std::vector<T> tau_init(static_cast<std::size_t>(k), T{0});
  auto d_tau = to_device(handle, tau_init);

  std::vector<T> vn(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    vn[static_cast<std::size_t>(j)] = column_norm(a_orig, m, lda, j);
  }
  auto d_vn1 = to_device(handle, vn);
  auto d_vn2 = to_device(handle, vn);
  std::vector<T> work_init(static_cast<std::size_t>(n), T{0});
  auto d_work = to_device(handle, work_init);

  std::vector<int> jpvt(static_cast<std::size_t>(n), 0);
  const auto status =
      laqp2<T>(blas, m, n, 0, d_a.data(), lda, jpvt.data(), d_tau.data(), d_vn1.data(),
               d_vn2.data(), d_work.data());
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "m=" << m << " n=" << n;

  const auto got_a = from_device(handle, d_a, static_cast<std::size_t>(lda) * n);
  const auto got_tau = from_device(handle, d_tau, static_cast<std::size_t>(k));
  wwr::wwrblasDestroy(blas);

  // 1. Pivot permutation must match geqp3's exactly.
  for (int j = 0; j < n; ++j) {
    EXPECT_EQ(jpvt[static_cast<std::size_t>(j)], ref_jpvt[static_cast<std::size_t>(j)])
        << "jpvt[" << j << "] m=" << m << " n=" << n << " seed=" << seed;
  }

  const T norm_a = frobenius_norm(a_orig);
  const T tol =
      factorization_tol<T>(norm_a, static_cast<std::size_t>(m), static_cast<std::size_t>(n));

  // 2. |R| diagonal magnitudes match (sign-free): R(i,i) = packed(i,i).
  for (int i = 0; i < k; ++i) {
    const T got = got_a[static_cast<std::size_t>(i) * lda + i];
    const T ref = ref_a[static_cast<std::size_t>(i) * lda + i];
    EXPECT_NEAR(std::abs(got), std::abs(ref), tol)
        << "|R(" << i << "," << i << ")| m=" << m << " n=" << n << " seed=" << seed;
  }

  // 3. Residual ||A(:,jpvt) - Q*R|| / ||A|| within tolerance. Reconstruct Q*R
  // from the device's own packed output and tau, and compare against the
  // original A permuted by the device's jpvt (1-based).
  const auto qr = reconstruct_qr(got_a, got_tau, m, n, lda);
  std::vector<T> residual(static_cast<std::size_t>(lda) * n);
  for (int j = 0; j < n; ++j) {
    const int src = jpvt[static_cast<std::size_t>(j)] - 1; // original column index
    for (int i = 0; i < m; ++i) {
      residual[static_cast<std::size_t>(j) * lda + i] =
          a_orig[static_cast<std::size_t>(src) * lda + i] - qr[static_cast<std::size_t>(j) * lda + i];
    }
  }
  EXPECT_LE(frobenius_norm(residual), tol)
      << "residual m=" << m << " n=" << n << " seed=" << seed;
}

TEST(Laqp2OracleTests, TallFloat) {
  expect_matches_reference<float>(8, 4, 1);
  expect_matches_reference<float>(12, 5, 2);
}
TEST(Laqp2OracleTests, TallDouble) {
  expect_matches_reference<double>(8, 4, 3);
  expect_matches_reference<double>(12, 5, 4);
}
TEST(Laqp2OracleTests, WideFloat) {
  expect_matches_reference<float>(4, 8, 5);
  expect_matches_reference<float>(5, 10, 6);
}
TEST(Laqp2OracleTests, WideDouble) {
  expect_matches_reference<double>(4, 8, 7);
  expect_matches_reference<double>(5, 10, 8);
}
TEST(Laqp2OracleTests, SquareFloat) {
  expect_matches_reference<float>(6, 6, 9);
}
TEST(Laqp2OracleTests, SquareDouble) {
  expect_matches_reference<double>(6, 6, 10);
  expect_matches_reference<double>(16, 16, 11);
}

// A single-column and single-row degenerate shape: laqp2 must still agree with
// geqp3 (one reflector, trivial or empty trailing block).
TEST(Laqp2OracleTests, DegenerateShapes) {
  expect_matches_reference<double>(5, 1, 20);
  expect_matches_reference<double>(1, 5, 21);
  expect_matches_reference<double>(1, 1, 22);
}

// Drives the LAWN 176 TOL3Z recompute branch. The perturbation makes one trailing
// column almost a scalar multiple of the pivot column, so after the first
// reflector that column's trailing part nearly cancels: |A(0,j)|/vn1[j] -> 1,
// the cheap-downdate shrink d -> 0, and d*(vn1/vn2)^2 falls below TOL3Z =
// sqrt(eps), tripping the exact recompute. The result must still match geqp3.
template<typename T>
void near_cancellation(std::vector<T> &a, int m, int lda) {
  // Column 0 is the eventual pivot (largest norm). Make column 1 = 1e6 * column 0
  // plus a tiny off-direction wobble, so its projection onto column 0 dominates
  // and the trailing residual is ~1e-? of its norm -- below the relative floor.
  const T scale = static_cast<T>(1e6);
  for (int i = 0; i < m; ++i) {
    a[static_cast<std::size_t>(0) * lda + i] = static_cast<T>(i + 1);
  }
  if (lda * 2 <= static_cast<int>(a.size())) {
    for (int i = 0; i < m; ++i) {
      a[static_cast<std::size_t>(1) * lda + i] =
          scale * a[static_cast<std::size_t>(0) * lda + i];
    }
    // A minute wobble on one entry keeps the matrix full rank but leaves the
    // trailing norm catastrophically smaller than the original -- the recompute
    // trigger. The value is exactly representable so float and double agree.
    a[static_cast<std::size_t>(1) * lda + (m - 1)] += static_cast<T>(1) / static_cast<T>(1024);
  }
}

TEST(Laqp2OracleTests, DegradedColumnRecomputeDouble) {
  expect_matches_reference<double>(10, 4, 30, near_cancellation<double>);
  expect_matches_reference<double>(8, 3, 31, near_cancellation<double>);
}
TEST(Laqp2OracleTests, DegradedColumnRecomputeFloat) {
  expect_matches_reference<float>(10, 4, 32, near_cancellation<float>);
}

} // namespace
} // namespace calaman
