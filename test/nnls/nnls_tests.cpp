// Tests for calaman.nnls -- non-negative least squares (Lawson-Hanson active
// set). Unlike the factorization suites there is NO LAPACKE oracle (LAPACKE has
// no nnls), so correctness is pinned by the problem's own first-order optimality
// instead of a reference call:
//
//   * KKT: at the returned x, with r = b - Ax and w = A^T r, every x_j >= 0,
//     every w_j <= 0 (dual feasibility), and x_j > 0 => w_j == 0
//     (complementary slackness) -- to a scaled tolerance. A wrong solver breaks
//     one of these by an O(||A||) amount.
//   * Recovery: when b = A x* for a known x* >= 0 and A has full column rank, the
//     unconstrained least-squares solution IS x* and is feasible, so nnls must
//     return x*.
//   * Zero solution: when A^T b <= 0 (here A >= 0 entrywise and b <= 0), x = 0 is
//     already optimal and nnls converges at iteration 0.
//
// Tall, square, wide and rank-deficient A are all exercised (the rank-deficient
// case drives geqp3's rank-truncated subproblem path), plus the iteration cap as
// an outcome and the argument-checking contract.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/b/x/work on the device
// and runs the active-set kernels, so `ctest -LE gpu` excludes it. Needs no
// oracle, so -- unlike geqp3 -- the build does not condition on
// calaman::lapack_reference.

#include <gtest/gtest.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.nnls;
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

// y = A x, A column-major m-by-n (A[j*m + i] = A(i,j)).
template<typename T>
std::vector<T> matvec(int m, int n, const std::vector<T> &A, const std::vector<T> &x) {
  std::vector<T> y(static_cast<std::size_t>(m), T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      y[static_cast<std::size_t>(i)] += A[static_cast<std::size_t>(j) * m + i] * x[static_cast<std::size_t>(j)];
    }
  }
  return y;
}

// w = A^T r.
template<typename T>
std::vector<T> matvec_t(int m, int n, const std::vector<T> &A, const std::vector<T> &r) {
  std::vector<T> w(static_cast<std::size_t>(n), T{0});
  for (int j = 0; j < n; ++j) {
    T acc{};
    for (int i = 0; i < m; ++i) {
      acc += A[static_cast<std::size_t>(j) * m + i] * r[static_cast<std::size_t>(i)];
    }
    w[static_cast<std::size_t>(j)] = acc;
  }
  return w;
}

template<typename T>
struct Solved {
  std::vector<T> x;
  NnlsInfo<T> info;
  wwr::wwrblasStatus_t status;
};

// Stage A (m-by-n column-major), b (m) on the device, allocate the worst-case
// workspace from nnls_bufferSize, run nnls, and read x (n) back.
template<typename T>
Solved<T> solve(std::shared_ptr<DeviceHandle> handle, int m, int n, const std::vector<T> &A,
                const std::vector<T> &b, NnlsOptions<T> opts = {}) {
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  wwr::wwrsolverDnHandle_t solver{};
  EXPECT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  auto d_A = to_device(handle, A);
  auto d_b = to_device(handle, b);
  std::vector<T> x_init(static_cast<std::size_t>(n), T{0});
  auto d_x = to_device(handle, x_init);

  const std::size_t bytes = nnls_bufferSize<T>(solver, m, n);
  const std::size_t count = (bytes + sizeof(T) - 1) / sizeof(T);
  std::vector<T> work_init(count, T{0});
  auto d_work = to_device(handle, work_init);

  NnlsInfo<T> info{};
  const auto status =
      nnls<T>(blas, solver, m, n, d_A.data(), m, d_b.data(), d_x.data(),
              static_cast<void *>(d_work.data()), bytes, opts, &info);

  auto x = from_device(handle, d_x, static_cast<std::size_t>(n));
  wwr::wwrsolverDnDestroy(solver);
  wwr::wwrblasDestroy(blas);
  return {std::move(x), info, status};
}

// Assert the Karush-Kuhn-Tucker conditions at x for min ||Ax - b|| s.t. x >= 0.
template<typename T>
void expect_kkt(int m, int n, const std::vector<T> &A, const std::vector<T> &b,
                const std::vector<T> &x) {
  const auto Ax = matvec(m, n, A, x);
  std::vector<T> r(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    r[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)] - Ax[static_cast<std::size_t>(i)];
  }
  const auto w = matvec_t(m, n, A, r);

  const T norm_a = frobenius_norm(A);
  const T norm_b = frobenius_norm(b);
  const T xnorm = frobenius_norm(x);
  const T scale = norm_a * (norm_a * (T{1} + xnorm) + norm_b) + T{1};
  const T tol = T{1024} * eps<T>() * scale;

  for (int j = 0; j < n; ++j) {
    const T xj = x[static_cast<std::size_t>(j)];
    const T wj = w[static_cast<std::size_t>(j)];
    EXPECT_GE(xj, -tol) << "feasibility x[" << j << "]";
    EXPECT_LE(wj, tol) << "dual feasibility w[" << j << "]";
    if (xj > tol) {
      EXPECT_LE(std::abs(wj), tol) << "complementarity w[" << j << "], x[" << j << "]=" << xj;
    }
  }
}

template<typename T>
std::vector<T> random_matrix(int m, int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> A(static_cast<std::size_t>(m) * n);
  for (auto &v : A) {
    v = static_cast<T>(dist(rng));
  }
  return A;
}

// ---- KKT on random problems, every shape ---------------------------------

template<typename T>
void kkt_case(int m, int n, unsigned seed) {
  auto handle = shared_device();
  const auto A = random_matrix<T>(m, n, seed);
  std::mt19937 rng(seed ^ 0x9e3779b9u);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> b(static_cast<std::size_t>(m));
  for (auto &v : b) {
    v = static_cast<T>(dist(rng));
  }

  const auto out = solve<T>(handle, m, n, A, b);
  ASSERT_EQ(out.status, wwr::WWRBLAS_STATUS_SUCCESS) << "m=" << m << " n=" << n;
  EXPECT_EQ(out.info.reason, NnlsStopReason::Converged) << "m=" << m << " n=" << n;
  expect_kkt(m, n, A, b, out.x);
}

TEST(NnlsKktTests, TallDouble) {
  kkt_case<double>(8, 4, 1);
  kkt_case<double>(20, 7, 2);
}
TEST(NnlsKktTests, TallFloat) {
  kkt_case<float>(8, 4, 3);
  kkt_case<float>(20, 7, 4);
}
TEST(NnlsKktTests, SquareDouble) {
  kkt_case<double>(6, 6, 5);
  kkt_case<double>(12, 12, 6);
}
TEST(NnlsKktTests, SquareFloat) { kkt_case<float>(6, 6, 7); }
TEST(NnlsKktTests, WideDouble) {
  kkt_case<double>(4, 8, 8);
  kkt_case<double>(5, 12, 9);
}
TEST(NnlsKktTests, WideFloat) { kkt_case<float>(4, 8, 10); }

// Rank-deficient A: columns [rank:n) are linear combinations of the first
// `rank`, so the passive-set subproblem is rank-deficient and geqp3's
// rank-truncated solve is exercised. KKT still characterizes the optimum.
template<typename T>
std::vector<T> make_rank_deficient(int m, int n, int rank, unsigned seed) {
  auto A = random_matrix<T>(m, n, seed);
  for (int j = rank; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      A[static_cast<std::size_t>(j) * m + i] = T{0};
    }
    for (int c = 0; c < rank; ++c) {
      const T wgt = static_cast<T>(c + 1) / static_cast<T>(4);
      for (int i = 0; i < m; ++i) {
        A[static_cast<std::size_t>(j) * m + i] += wgt * A[static_cast<std::size_t>(c) * m + i];
      }
    }
  }
  return A;
}

template<typename T>
void rank_deficient_case(int m, int n, int rank, unsigned seed) {
  auto handle = shared_device();
  const auto A = make_rank_deficient<T>(m, n, rank, seed);
  std::mt19937 rng(seed ^ 0x85ebca6bu);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> b(static_cast<std::size_t>(m));
  for (auto &v : b) {
    v = static_cast<T>(dist(rng));
  }
  const auto out = solve<T>(handle, m, n, A, b);
  ASSERT_EQ(out.status, wwr::WWRBLAS_STATUS_SUCCESS) << "m=" << m << " n=" << n;
  EXPECT_EQ(out.info.reason, NnlsStopReason::Converged);
  expect_kkt(m, n, A, b, out.x);
}

TEST(NnlsKktTests, RankDeficientDouble) {
  rank_deficient_case<double>(10, 6, 3, 20);
  rank_deficient_case<double>(8, 8, 5, 21);
  rank_deficient_case<double>(6, 9, 4, 22);
}
TEST(NnlsKktTests, RankDeficientFloat) {
  rank_deficient_case<float>(10, 6, 3, 23);
}

// ---- Recovery of a known non-negative solution ---------------------------

template<typename T>
void recovery_case(int m, int n, unsigned seed) {
  auto handle = shared_device();
  const auto A = random_matrix<T>(m, n, seed);

  // A known x* >= 0 with some exact zeros; b = A x* is attained exactly, so the
  // unconstrained least-squares solution is x* and is feasible.
  std::mt19937 rng(seed ^ 0xc2b2ae35u);
  std::uniform_real_distribution<double> dist(0.0, 2.0);
  std::vector<T> xstar(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    xstar[static_cast<std::size_t>(j)] = (j % 3 == 0) ? T{0} : static_cast<T>(dist(rng));
  }
  const auto b = matvec(m, n, A, xstar);

  const auto out = solve<T>(handle, m, n, A, b);
  ASSERT_EQ(out.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(out.info.reason, NnlsStopReason::Converged);

  const T xnorm = frobenius_norm(xstar);
  const T norm_a = frobenius_norm(A);
  const T tol = T{4096} * eps<T>() * norm_a * (T{1} + xnorm);
  std::vector<T> diff(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    diff[static_cast<std::size_t>(j)] = out.x[static_cast<std::size_t>(j)] - xstar[static_cast<std::size_t>(j)];
    EXPECT_GE(out.x[static_cast<std::size_t>(j)], -tol);
  }
  EXPECT_LE(frobenius_norm(diff), tol) << "recovery m=" << m << " n=" << n << " seed=" << seed;
  // b is attained, so the residual is ~0.
  EXPECT_LE(out.info.residual_norm, tol * (T{1} + frobenius_norm(b)));
}

TEST(NnlsRecoveryTests, TallDouble) {
  recovery_case<double>(15, 5, 100);
  recovery_case<double>(24, 6, 101);
}
TEST(NnlsRecoveryTests, TallFloat) { recovery_case<float>(15, 5, 102); }
TEST(NnlsRecoveryTests, SquareDouble) { recovery_case<double>(8, 8, 103); }

// ---- A^T b <= 0 forces x = 0 ---------------------------------------------

template<typename T>
void zero_case(int m, int n, unsigned seed) {
  auto handle = shared_device();
  // A entrywise non-negative and b entrywise non-positive => (A^T b)_j <= 0 for
  // all j, so x = 0 is optimal and nnls converges at iteration 0.
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> pos(0.1, 2.0);
  std::vector<T> A(static_cast<std::size_t>(m) * n);
  for (auto &v : A) {
    v = static_cast<T>(pos(rng));
  }
  std::vector<T> b(static_cast<std::size_t>(m));
  for (auto &v : b) {
    v = static_cast<T>(-pos(rng));
  }

  const auto out = solve<T>(handle, m, n, A, b);
  ASSERT_EQ(out.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(out.info.reason, NnlsStopReason::Converged);
  EXPECT_EQ(out.info.iterations, 0);

  const T tol = T{16} * eps<T>() * frobenius_norm(A);
  for (int j = 0; j < n; ++j) {
    EXPECT_LE(std::abs(out.x[static_cast<std::size_t>(j)]), tol) << "x[" << j << "]";
  }
}

TEST(NnlsZeroTests, Double) {
  zero_case<double>(8, 5, 200);
  zero_case<double>(6, 6, 201);
}
TEST(NnlsZeroTests, Float) { zero_case<float>(8, 5, 202); }

// ---- The iteration cap is an outcome, not an error -----------------------

TEST(NnlsIterationCapTests, Double) {
  auto handle = shared_device();
  const int m = 20;
  const int n = 8;
  const auto A = random_matrix<double>(m, n, 300);
  // A solution with several positive components needs several outer iterations;
  // cap at 1 so the budget runs out first.
  std::vector<double> xstar(static_cast<std::size_t>(n), 1.0);
  const auto b = matvec(m, n, A, xstar);

  NnlsOptions<double> opts{};
  opts.max_iterations = 1;
  const auto out = solve<double>(handle, m, n, A, b, opts);
  EXPECT_EQ(out.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(out.info.reason, NnlsStopReason::MaxIterations);
  EXPECT_EQ(out.info.iterations, 1);
}

// ---- Argument-checking contract ------------------------------------------

TEST(NnlsArgumentTests, RejectsBadArguments) {
  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  const int m = 6;
  const int n = 3;
  const auto A = random_matrix<double>(m, n, 400);
  std::vector<double> b(static_cast<std::size_t>(m), 1.0);
  auto d_A = to_device(handle, A);
  auto d_b = to_device(handle, b);
  std::vector<double> x_init(static_cast<std::size_t>(n), 0.0);
  auto d_x = to_device(handle, x_init);
  const std::size_t bytes = nnls_bufferSize<double>(solver, m, n);
  std::vector<double> work_init((bytes + sizeof(double) - 1) / sizeof(double), 0.0);
  auto d_work = to_device(handle, work_init);
  void *work = static_cast<void *>(d_work.data());

  // Bad dimensions.
  EXPECT_EQ(nnls<double>(blas, solver, 0, n, d_A.data(), m, d_b.data(), d_x.data(), work, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(nnls<double>(blas, solver, m, 0, d_A.data(), m, d_b.data(), d_x.data(), work, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  // lda < m.
  EXPECT_EQ(nnls<double>(blas, solver, m, n, d_A.data(), m - 1, d_b.data(), d_x.data(), work, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  // Null pointers.
  EXPECT_EQ(nnls<double>(blas, solver, m, n, nullptr, m, d_b.data(), d_x.data(), work, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(nnls<double>(blas, solver, m, n, d_A.data(), m, nullptr, d_x.data(), work, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(nnls<double>(blas, solver, m, n, d_A.data(), m, d_b.data(), nullptr, work, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(nnls<double>(blas, solver, m, n, d_A.data(), m, d_b.data(), d_x.data(), nullptr, bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  // Undersized workspace.
  EXPECT_EQ(nnls<double>(blas, solver, m, n, d_A.data(), m, d_b.data(), d_x.data(), work, 0),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);

  wwr::wwrsolverDnDestroy(solver);
  wwr::wwrblasDestroy(blas);
}

} // namespace
} // namespace calaman
