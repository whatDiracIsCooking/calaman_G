// Oracle test for calaman.linalg:geqp3 -- the top-level QR-with-column-pivoting
// driver, all-free unblocked path. The oracle is LAPACKE_?geqp3 with jpvt = 0
// (every column free), so the two must agree on the pivot permutation jpvt, on
// |R| (its diagonal magnitudes), and on the factorization residual
// ||A*P - Q*R|| / ||A||.
//
// A pivoted QR is unique only up to column ties and reflector sign, so -- like
// the laqp2 suite this mirrors -- the robust invariant is the residual
// ||A(:,jpvt) - Q*R||, not bitwise-equal R or signs: this reconstructs Q*R on
// the host from the stored reflectors and tau and checks it against the
// pivot-permuted original A, to the shared factorization tolerance. jpvt is
// compared directly and |R|'s diagonal magnitudes are the second sign-free
// invariant. Tall, wide, square and rank-deficient inputs are all exercised.
//
// Unlike laqp2's suite, the test does NOT pre-seed vn1/vn2: geqp3 seeds the
// column norms itself, so the device buffers are passed as bare scratch.
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

// Reference pivoted QR: LAPACKE_?geqp3 over the whole matrix, column-major, with
// jpvt zero-initialised (all columns free).
int ref_geqp3(int m, int n, float *a, int lda, int *jpvt, float *tau) {
  return LAPACKE_sgeqp3(LAPACK_COL_MAJOR, m, n, a, lda, jpvt, tau);
}
int ref_geqp3(int m, int n, double *a, int lda, int *jpvt, double *tau) {
  return LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, a, lda, jpvt, tau);
}

// Reconstruct B = Q*R on the host from the packed geqp3 output. A holds R in its
// upper trapezoid and reflector j (v with v[0]==1 implicit) below the diagonal of
// column j; tau holds the k = min(m,n) scalars. Q = H_0 H_1 ... H_{k-1}, so apply
// the reflectors to R from the left in REVERSE order. Returns B column-major, m x n.
template<typename T>
std::vector<T> reconstruct_qr(const std::vector<T> &packed, const std::vector<T> &tau, int m,
                              int n, int lda) {
  const int k = std::min(m, n);
  std::vector<T> b(static_cast<std::size_t>(lda) * n, T{0});
  for (int j = 0; j < n; ++j) {
    const int top = std::min(j + 1, m); // rows 0..min(j,m-1) are R's
    for (int i = 0; i < top; ++i) {
      b[static_cast<std::size_t>(j) * lda + i] = packed[static_cast<std::size_t>(j) * lda + i];
    }
  }
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

/// @brief geqp3 on the device must match the reference for one (m, n, seed) case
///
/// @p unique_pivots is false for rank-deficient inputs, where equal-norm
/// dependent columns make the greedy pivot choice genuinely non-unique: the
/// direct jpvt comparison is skipped and only the sign-free invariants (residual,
/// |R| diagonal magnitudes) -- which hold for any valid pivoted QR -- are checked.
template<typename T>
void expect_matches_reference(int m, int n, unsigned seed,
                              std::function<void(std::vector<T> &, int, int)> perturb = {},
                              bool unique_pivots = true, int nb = 0) {
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

  // Device geqp3. vn1/vn2 are bare scratch here -- geqp3 seeds them itself.
  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = to_device(handle, a_orig);
  std::vector<T> tau_init(static_cast<std::size_t>(k), T{0});
  auto d_tau = to_device(handle, tau_init);

  std::vector<T> scratch(static_cast<std::size_t>(n), T{0});
  auto d_vn1 = to_device(handle, scratch);
  auto d_vn2 = to_device(handle, scratch);
  // work must hold the blocked driver's extended scratch (laqp2 scratch | F |
  // auxv | the int mask); geqp3_work_size gives the length for this m, n.
  std::vector<T> work_init(geqp3_work_size(m, n), T{0});
  auto d_work = to_device(handle, work_init);

  std::vector<int> jpvt(static_cast<std::size_t>(n), 0);
  const auto status =
      geqp3<T>(blas, m, n, d_a.data(), lda, jpvt.data(), d_tau.data(), d_vn1.data(),
               d_vn2.data(), d_work.data(), nb);
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "m=" << m << " n=" << n;

  const auto got_a = from_device(handle, d_a, static_cast<std::size_t>(lda) * n);
  const auto got_tau = from_device(handle, d_tau, static_cast<std::size_t>(k));
  wwr::wwrblasDestroy(blas);

  const T norm_a = frobenius_norm(a_orig);
  const T tol =
      factorization_tol<T>(norm_a, static_cast<std::size_t>(m), static_cast<std::size_t>(n));

  // 1. Pivot permutation. With well-separated norms geqp3 and the reference make
  // the same greedy choice, so jpvt matches exactly. Under genuine column ties
  // (the rank-deficient cases) the choice is non-unique, so only the sign-free
  // invariants below are asserted.
  if (unique_pivots) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(jpvt[static_cast<std::size_t>(j)], ref_jpvt[static_cast<std::size_t>(j)])
          << "jpvt[" << j << "] m=" << m << " n=" << n << " seed=" << seed;
    }
  }

  // 2. |R| diagonal magnitudes (sign-free): R(i,i) = packed(i,i). When the pivots
  // match, compare in order; otherwise compare the sorted multiset, which any
  // valid pivoted QR of this A shares.
  std::vector<T> got_diag(static_cast<std::size_t>(k));
  std::vector<T> ref_diag(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    got_diag[static_cast<std::size_t>(i)] = std::abs(got_a[static_cast<std::size_t>(i) * lda + i]);
    ref_diag[static_cast<std::size_t>(i)] = std::abs(ref_a[static_cast<std::size_t>(i) * lda + i]);
  }
  if (!unique_pivots) {
    std::sort(got_diag.begin(), got_diag.end());
    std::sort(ref_diag.begin(), ref_diag.end());
  }
  for (int i = 0; i < k; ++i) {
    EXPECT_NEAR(got_diag[static_cast<std::size_t>(i)], ref_diag[static_cast<std::size_t>(i)], tol)
        << "|R| diag " << i << " m=" << m << " n=" << n << " seed=" << seed;
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

// Make a rank-deficient matrix: fill columns [rank:n) as fixed linear
// combinations of the first `rank` columns, so A has exactly `rank` independent
// columns and geqp3 must pivot the dependent ones last. The reference and the
// device make the same greedy choice on these well-separated norms.
template<typename T>
std::function<void(std::vector<T> &, int, int)> make_rank_deficient(int rank) {
  return [rank](std::vector<T> &a, int m, int lda) {
    const int n = static_cast<int>(a.size() / static_cast<std::size_t>(lda));
    for (int j = rank; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        a[static_cast<std::size_t>(j) * lda + i] = T{0};
      }
      // Dependent column j is a sum of the independent columns with distinct,
      // exactly representable weights, so float and double agree.
      for (int c = 0; c < rank; ++c) {
        const T w = static_cast<T>(c + 1) / static_cast<T>(4);
        for (int i = 0; i < m; ++i) {
          a[static_cast<std::size_t>(j) * lda + i] +=
              w * a[static_cast<std::size_t>(c) * lda + i];
        }
      }
    }
  };
}

TEST(Geqp3OracleTests, TallFloat) {
  expect_matches_reference<float>(8, 4, 1);
  expect_matches_reference<float>(12, 5, 2);
}
TEST(Geqp3OracleTests, TallDouble) {
  expect_matches_reference<double>(8, 4, 3);
  expect_matches_reference<double>(12, 5, 4);
}
TEST(Geqp3OracleTests, WideFloat) {
  expect_matches_reference<float>(4, 8, 5);
  expect_matches_reference<float>(5, 10, 6);
}
TEST(Geqp3OracleTests, WideDouble) {
  expect_matches_reference<double>(4, 8, 7);
  expect_matches_reference<double>(5, 10, 8);
}
TEST(Geqp3OracleTests, SquareFloat) {
  expect_matches_reference<float>(6, 6, 9);
}
TEST(Geqp3OracleTests, SquareDouble) {
  expect_matches_reference<double>(6, 6, 10);
  expect_matches_reference<double>(16, 16, 11);
}

// Rank-deficient: fewer independent columns than n, tall and wide and square.
// Equal-norm dependent columns make the greedy pivot choice non-unique, so
// unique_pivots = false -- the residual and sorted |R| diagonals still pin it.
TEST(Geqp3OracleTests, RankDeficientDouble) {
  expect_matches_reference<double>(10, 6, 40, make_rank_deficient<double>(3), false);
  expect_matches_reference<double>(6, 6, 41, make_rank_deficient<double>(4), false);
  expect_matches_reference<double>(5, 9, 42, make_rank_deficient<double>(4), false);
}
TEST(Geqp3OracleTests, RankDeficientFloat) {
  expect_matches_reference<float>(10, 6, 43, make_rank_deficient<float>(3), false);
  expect_matches_reference<float>(8, 8, 44, make_rank_deficient<float>(5), false);
}

// Degenerate shapes: single column / single row, and the empty matrices
// m == 0, n == 0, min(m,n) == 0. geqp3 must return cleanly and agree with the
// reference (or be a no-op where there is nothing to factor).
TEST(Geqp3OracleTests, DegenerateShapes) {
  expect_matches_reference<double>(5, 1, 50);
  expect_matches_reference<double>(1, 5, 51);
  expect_matches_reference<double>(1, 1, 52);
}

// Crossover fallback: a block width driven PAST min(m,n) forces geqp3 to skip
// the blocked :laqps loop and factor the whole matrix as one :laqp2 panel. The
// same inputs that take the blocked route by default (min(m,n) > the crossover)
// must still match the reference when pushed onto the unblocked route, proving
// the two paths agree. nb = 1000 is far past any min(m,n) here.
TEST(Geqp3OracleTests, CrossoverFallbackDouble) {
  expect_matches_reference<double>(12, 8, 60, {}, true, 1000);
  expect_matches_reference<double>(16, 16, 61, {}, true, 1000);
  expect_matches_reference<double>(10, 12, 62, {}, true, 1000);
}
TEST(Geqp3OracleTests, CrossoverFallbackFloat) {
  expect_matches_reference<float>(12, 8, 63, {}, true, 1000);
}

// Larger matrices: min(m,n) well past kBlockSize, so geqp3 runs MULTIPLE :laqps
// blocks before the tail -- exercising the block-to-block hand-off (F reset per
// block, the running jpvt across blocks, the per-block deferred gemm).
TEST(Geqp3OracleTests, MultiBlockDouble) {
  expect_matches_reference<double>(80, 70, 70);
  expect_matches_reference<double>(64, 64, 71);
}
TEST(Geqp3OracleTests, MultiBlockFloat) {
  expect_matches_reference<float>(80, 70, 72);
}

// Empty matrices: m == 0 or n == 0 means min(m,n) == 0 -- nothing to factor.
// geqp3 short-circuits to success and writes nothing. Driven directly (the
// oracle harness assumes a nonempty matrix), checking only the return status.
TEST(Geqp3OracleTests, EmptyMatrices) {
  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  std::vector<double> one(1, 0.0);
  auto d_a = to_device(handle, one);
  auto d_tau = to_device(handle, one);
  auto d_vn1 = to_device(handle, one);
  auto d_vn2 = to_device(handle, one);
  auto d_work = to_device(handle, one);
  std::vector<int> jpvt(1, 0);

  // m == 0 (n > 0), n == 0 (m > 0), and both zero: all min(m,n) == 0.
  for (const auto [m, n] : std::vector<std::pair<int, int>>{{0, 4}, {3, 0}, {0, 0}}) {
    const auto status = geqp3<double>(blas, m, n, d_a.data(), 1, jpvt.data(), d_tau.data(),
                                      d_vn1.data(), d_vn2.data(), d_work.data());
    EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "m=" << m << " n=" << n;
  }

  wwr::wwrblasDestroy(blas);
}

} // namespace
} // namespace calaman
