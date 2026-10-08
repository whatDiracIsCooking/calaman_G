// Reference suite for calaman.lanczos's lanczos_solve. The operator is a dense
// A = Q diag(lambda) Q^T on the device, applied through a symv-backed
// lanczos_matvec adapter; the oracle is LAPACKE_?syevd on the same A. Checked,
// for smallest / largest / both_ends, float and double:
//
//   * the selected eigenvalues match the reference's to the shared tolerance;
//   * the Ritz vectors: ||A X - X Theta||_F and ||X^T X - I||_F to it;
//   * well-separated outliers converge in one cycle (iterations == 0, matvecs
//     == ncv); a spectrum clustered at the wanted end needs thick restarts, each
//     costing ncv - k matvecs (k = lanczos_restart_keep).
//
// Plus: ncv == n (every Ritz pair exact), max_iterations exhaustion (success,
// reason MaxIterations, eigenvalues still written), the true-residual pass, breakdown recovery
// (an invariant subspace from e_0, and one per distinct eigenvalue of a
// low-rank-spectrum A), a repeated eigenvalue found once (the documented
// single-vector limitation), a zero start vector rejected, and a
// linear_operator model taking the matvec lambda's exact path. REQUIRES_GPU.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

#include "shared/expect_converged.h"

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.wrappers.blas;
import wwr.extension.memory_buffer;
import calaman.lanczos;
import calaman.error_handling;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;
using test::frobenius_norm;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

constexpr int kInvalidValue = static_cast<int>(wwr::WWRBLAS_STATUS_INVALID_VALUE);

// ── the matvec adapter ───────────────────────────────────────────────────────

/// y = A x for a dense symmetric n x n device matrix (lower triangle read).
/// lanczos_solve calls the matvec in HOST pointer mode, so host scalars.
template<typename T>
auto symv_matvec(wwr::wwrblasHandle_t blas, int n, const T *d_a) {
  return [blas, n, d_a](wwr::wwrStream_t, const T *x, T *y) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::symv<T, int>(blas, wwr::WWRBLAS_FILL_MODE_LOWER, n, &one, d_a, n, x, 1, &zero, y,
                             1);
  };
}

/// The same A as a linear_operator model: one symv per column, counting its
/// applies and noting any made outside HOST pointer mode.
template<typename T>
struct SymvOperator {
  wwr::wwrblasHandle_t blas;
  int n;
  const T *d_a;
  int applies = 0;
  bool host_mode = true;

  Status apply(wwr::wwrStream_t stream, int k, const T *X, T *Y) {
    ++applies;
    wwr::wwrblasPointerMode_t mode{};
    const Status got = wwr::wwrblasGetPointerMode(blas, &mode);
    host_mode = host_mode && got.ok() && mode == wwr::WWRBLAS_POINTER_MODE_HOST;
    const auto matvec = symv_matvec<T>(blas, n, d_a);
    const auto nz = static_cast<std::size_t>(n);
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      const Status st = matvec(stream, X + j * nz, Y + j * nz);
      if (!st.ok()) {
        return st;
      }
    }
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};
static_assert(linear_operator<SymvOperator<float>, float>);
static_assert(linear_operator<SymvOperator<double>, double>);

// ── host side ────────────────────────────────────────────────────────────────

lapack_int ref_syevd(lapack_int n, float *a, float *w) {
  return LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a, n, w);
}
lapack_int ref_syevd(lapack_int n, double *a, double *w) {
  return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a, n, w);
}

/// A = Q diag(lambda) Q^T, Q a random orthogonal (geqrf + orgqr), formed in
/// double and rounded to T. Column-major, both triangles.
template<typename T>
std::vector<T> from_spectrum(const std::vector<double> &lambda, std::uint32_t seed) {
  const int n = static_cast<int>(lambda.size());
  const auto nz = static_cast<std::size_t>(n);
  std::vector<double> q(nz * nz);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (double &x : q) {
    x = dist(gen);
  }
  std::vector<double> tau(nz);
  EXPECT_EQ(LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, q.data(), n, tau.data()), 0);
  EXPECT_EQ(LAPACKE_dorgqr(LAPACK_COL_MAJOR, n, n, n, q.data(), n, tau.data()), 0);
  std::vector<T> a(nz * nz);
  for (std::size_t j = 0; j < nz; ++j) {
    for (std::size_t i = 0; i < nz; ++i) {
      double sum = 0.0;
      for (std::size_t k = 0; k < nz; ++k) {
        sum += q[i + k * nz] * lambda[k] * q[j + k * nz];
      }
      a[i + j * nz] = static_cast<T>(sum);
    }
  }
  return a;
}

/// Three well-separated outliers at each end over a bulk spread in [-1, 1].
std::vector<double> outlier_spectrum(int n) {
  std::vector<double> lambda = {-10.0, -8.5, -7.0, 7.0, 8.5, 10.0};
  const int bulk = n - static_cast<int>(lambda.size());
  for (int i = 0; i < bulk; ++i) {
    lambda.push_back(-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(bulk - 1));
  }
  return lambda;
}

const char *name(LanczosWhich which) {
  switch (which) {
  case LanczosWhich::smallest:
    return "smallest";
  case LanczosWhich::largest:
    return "largest";
  case LanczosWhich::both_ends:
    return "both_ends";
  }
  return "?";
}

// ── device rig ───────────────────────────────────────────────────────────────

struct Handles {
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
  explicit Handles(wwr::wwrStream_t stream) {
    EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrblasSetStream(blas, stream), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrsolverDnSetStream(solver, stream), wwr::WWRSOLVER_STATUS_SUCCESS);
  }
  ~Handles() {
    wwr::wwrblasDestroy(blas);
    wwr::wwrsolverDnDestroy(solver);
  }
  Handles(const Handles &) = delete;
  Handles &operator=(const Handles &) = delete;
};

template<typename T>
struct Rig {
  std::shared_ptr<DeviceHandle> handle = shared_device();
  wwr::wwrStream_t stream = handle->stream().get();
  Handles h{stream};
  int n;
  int nev;
  int ncv;
  DeviceBuffer<T> d_a;
  DeviceBuffer<T> d_x;
  DeviceBuffer<T> d_w;
  std::unique_ptr<DeviceBuffer<std::byte>> work;
  LanczosSlices<T> s;

  Rig(const std::vector<T> &a, int n_, int nev_, int ncv_)
      : n(n_), nev(nev_), ncv(ncv_), d_a(a.size(), handle),
        d_x(static_cast<std::size_t>(n_) * static_cast<std::size_t>(nev_), handle),
        d_w(static_cast<std::size_t>(nev_), handle) {
    upload(d_a.data(), a);
    std::size_t lwork = 0;
    EXPECT_TRUE(lanczos_bufferSize<T>(h.solver, n, nev, ncv, &lwork).ok());
    work = std::make_unique<DeviceBuffer<std::byte>>(lwork, handle);
    EXPECT_TRUE(make_lanczos_slices<T>(h.solver, n, nev, ncv, work->data(), &s, nullptr).ok());
  }

  void upload(T *dst, const std::vector<T> &src) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(dst, src.data(), sizeof(T) * src.size(),
                                  wwr::wwrMemcpyHostToDevice, stream),
              wwr::wwrSuccess);
    ASSERT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  }
  std::vector<T> download(const T *src, std::size_t count) {
    std::vector<T> out(count);
    EXPECT_EQ(
        wwr::wwrMemcpyAsync(out.data(), src, sizeof(T) * count, wwr::wwrMemcpyDeviceToHost, stream),
        wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
    return out;
  }

  Status solve(LanczosWhich which, const LanczosOptions<T> &options, LanczosInfo *info) {
    return lanczos_solve<T>(h.blas, h.solver, stream, n, nev, ncv, which, s,
                            symv_matvec<T>(h.blas, n, d_a.data()), d_w.data(), d_x.data(), info,
                            options);
  }
  Status solve(LanczosWhich which, const LanczosOptions<T> &options, SymvOperator<T> &op,
               LanczosInfo *info) {
    return lanczos_solve<T>(h.blas, h.solver, stream, n, nev, ncv, which, s, op, d_w.data(),
                            d_x.data(), info, options);
  }
  std::vector<T> eigenvalues() { return download(d_w.data(), static_cast<std::size_t>(nev)); }
};

/// A finished solve: the report and the downloaded eigenvalues.
template<typename T>
struct Outcome {
  LanczosInfo info;
  std::vector<T> eigenvalues;
};

/// ||A X - X diag(theta)||_F and ||X^T X - I||_F, in double on the host.
template<typename T>
std::pair<double, double> ritz_errors(const std::vector<T> &a, const std::vector<T> &x,
                                      const std::vector<T> &theta, int n) {
  const auto nz = static_cast<std::size_t>(n);
  const std::size_t k = theta.size();
  double res = 0.0;
  double orth = 0.0;
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t i = 0; i < nz; ++i) {
      double ax = 0.0;
      for (std::size_t l = 0; l < nz; ++l) {
        ax += static_cast<double>(a[i + l * nz]) * static_cast<double>(x[l + c * nz]);
      }
      const double r = ax - static_cast<double>(theta[c]) * static_cast<double>(x[i + c * nz]);
      res += r * r;
    }
    for (std::size_t d = 0; d < k; ++d) {
      double dot = 0.0;
      for (std::size_t i = 0; i < nz; ++i) {
        dot += static_cast<double>(x[i + c * nz]) * static_cast<double>(x[i + d * nz]);
      }
      const double e = dot - (c == d ? 1.0 : 0.0);
      orth += e * e;
    }
  }
  return {std::sqrt(res), std::sqrt(orth)};
}

/// The reference eigenvalues lanczos_solve should return for @p which.
template<typename T>
std::vector<T> reference_selection(const std::vector<T> &a, int n, int nev, LanczosWhich which) {
  std::vector<T> w(static_cast<std::size_t>(n));
  std::vector<T> a_copy = a;
  EXPECT_EQ(ref_syevd(n, a_copy.data(), w.data()), 0);
  std::vector<T> expected;
  for (const int i : ritz_select(which, n, nev)) {
    expected.push_back(w[static_cast<std::size_t>(i)]);
  }
  return expected;
}

template<typename T>
LanczosOptions<T> tight_options() {
  LanczosOptions<T> options;
  options.tolerance = T(1000) * test::eps<T>();
  return options;
}

/// One converged solve, checked against LAPACKE_?syevd on @p a; the outcome is
/// returned for the caller's restart/matvec assertions.
template<typename T>
Outcome<T> solve_and_check(const std::vector<T> &a, int n, int nev, int ncv, LanczosWhich which,
                           const LanczosOptions<T> &options) {
  const std::vector<T> expected = reference_selection(a, n, nev, which);
  Rig<T> rig(a, n, nev, ncv);
  Outcome<T> result;
  const Status st = rig.solve(which, options, &result.info);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(result.info);
  result.eigenvalues = rig.eigenvalues();

  const T tol = factorization_tol<T>(frobenius_norm(a), n, n);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.eigenvalues[i], expected[i], tol) << i;
  }
  const auto x = rig.download(rig.d_x.data(), static_cast<std::size_t>(n) * nev);
  const auto [res, orth] = ritz_errors(a, x, result.eigenvalues, n);
  EXPECT_LE(res, static_cast<double>(tol));
  EXPECT_LE(orth, static_cast<double>(factorization_tol<T>(T(1), n, n)));
  return result;
}

/// One converged single-cycle solve: iterations == 0, matvecs == ncv.
template<typename T>
void check_solve(const std::vector<T> &a, int n, int nev, int ncv, LanczosWhich which) {
  SCOPED_TRACE(::testing::Message() << "which=" << name(which) << " n=" << n << " nev=" << nev
                                    << " ncv=" << ncv << " sizeof(T)=" << sizeof(T));
  const auto result = solve_and_check<T>(a, n, nev, ncv, which, tight_options<T>());
  EXPECT_EQ(result.info.iterations, 0);
  EXPECT_EQ(result.info.matvecs, ncv);
}

template<typename T>
void check_outliers() {
  constexpr int n = 200;
  const auto a = from_spectrum<T>(outlier_spectrum(n), 20261006u);
  check_solve<T>(a, n, 3, 40, LanczosWhich::smallest);
  check_solve<T>(a, n, 3, 40, LanczosWhich::largest);
  check_solve<T>(a, n, 6, 40, LanczosWhich::both_ends);
}

TEST(LanczosSolveReferenceTests, OneCycleConvergesDouble) {
  check_outliers<double>();
}
TEST(LanczosSolveReferenceTests, OneCycleConvergesFloat) {
  check_outliers<float>();
}

// ncv == n: no gap needed, the cycle spans R^n and every Ritz pair is exact.
template<typename T>
void check_full_basis() {
  constexpr int n = 24;
  std::vector<double> lambda;
  for (int i = 0; i < n; ++i) {
    lambda.push_back(-3.0 + 0.35 * static_cast<double>(i) + 0.01 * static_cast<double>(i % 3));
  }
  const auto a = from_spectrum<T>(lambda, 7u);
  check_solve<T>(a, n, 2, n, LanczosWhich::smallest);
  check_solve<T>(a, n, 2, n, LanczosWhich::largest);
  check_solve<T>(a, n, 4, n, LanczosWhich::both_ends);
}

TEST(LanczosSolveReferenceTests, FullBasisIsExactDouble) {
  check_full_basis<double>();
}
TEST(LanczosSolveReferenceTests, FullBasisIsExactFloat) {
  check_full_basis<float>();
}

// ── thick restart ────────────────────────────────────────────────────────────

/// The wanted ends clustered: the extremes sit 0.02 apart over an evenly spaced
/// bulk in [-1, 1], so one cycle of ncv = 16 cannot resolve them.
std::vector<double> clustered_spectrum(int n) {
  std::vector<double> lambda;
  for (int i = 0; i < n; ++i) {
    lambda.push_back(-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1));
  }
  return lambda;
}

/// Restarts forced: converged, iterations >= 2, and every restart cost ncv - k matvecs.
template<typename T>
void check_restarted(const std::vector<T> &a, int n, int nev, int ncv, LanczosWhich which) {
  SCOPED_TRACE(::testing::Message() << "which=" << name(which) << " n=" << n << " nev=" << nev
                                    << " ncv=" << ncv << " sizeof(T)=" << sizeof(T));
  const auto result = solve_and_check<T>(a, n, nev, ncv, which, tight_options<T>());
  EXPECT_GE(result.info.iterations, 2);
  EXPECT_EQ(result.info.matvecs,
            ncv + result.info.iterations * (ncv - lanczos_restart_keep(nev, ncv)));
}

template<typename T>
void check_clustered() {
  constexpr int n = 100;
  const auto a = from_spectrum<T>(clustered_spectrum(n), 31u);
  check_restarted<T>(a, n, 3, 16, LanczosWhich::smallest);
  check_restarted<T>(a, n, 3, 16, LanczosWhich::largest);
  check_restarted<T>(a, n, 4, 16, LanczosWhich::both_ends);
}

TEST(LanczosSolveReferenceTests, ClusteredEndRestartsDouble) {
  check_clustered<double>();
}
TEST(LanczosSolveReferenceTests, ClusteredEndRestartsFloat) {
  check_clustered<float>();
}

// max_iterations bounds the work: too few restarts to converge is still success,
// with reason MaxIterations and the nev eigenvalues written -- ascending, and
// each no lower than the true one it approximates from above (Cauchy interlacing).
TEST(LanczosSolveReferenceTests, MaxIterationsIsSuccess) {
  constexpr int n = 100;
  constexpr int nev = 3;
  constexpr int ncv = 16;
  const auto a = from_spectrum<double>(clustered_spectrum(n), 31u);
  const auto expected = reference_selection(a, n, nev, LanczosWhich::smallest);
  Rig<double> rig(a, n, nev, ncv);
  auto options = tight_options<double>();

  for (const int budget : {1, 0}) { // 0: a single cycle
    SCOPED_TRACE(::testing::Message() << "max_iterations=" << budget);
    options.max_iterations = budget;
    LanczosInfo info;
    const Status st = rig.solve(LanczosWhich::smallest, options, &info);
    EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
    EXPECT_FALSE(converged(info));
    EXPECT_EQ(static_cast<int>(info.reason), static_cast<int>(LanczosStopReason::MaxIterations));
    EXPECT_EQ(info.iterations, budget);
    EXPECT_EQ(info.matvecs, ncv + budget * (ncv - lanczos_restart_keep(nev, ncv)));
    const auto w = rig.eigenvalues();
    for (std::size_t i = 0; i < w.size(); ++i) {
      EXPECT_TRUE(std::isfinite(w[i])) << i;
      EXPECT_GE(w[i], expected[i] - 1e-12) << i;
      if (i > 0) {
        EXPECT_LT(w[i - 1], w[i]) << i;
      }
    }
  }
}

// verify_residuals: the estimates pass, then nev more matvecs confirm them.
template<typename T>
void check_verified() {
  constexpr int n = 100;
  constexpr int nev = 3;
  constexpr int ncv = 16;
  const auto a = from_spectrum<T>(clustered_spectrum(n), 31u);
  auto options = tight_options<T>();
  const auto plain = solve_and_check<T>(a, n, nev, ncv, LanczosWhich::smallest, options);
  options.verify_residuals = true;
  const auto verified = solve_and_check<T>(a, n, nev, ncv, LanczosWhich::smallest, options);
  // Same iteration up to the first check; a check that fails keeps restarting.
  EXPECT_GE(verified.info.iterations, plain.info.iterations);
  EXPECT_GE(verified.info.matvecs, plain.info.matvecs + nev);
}

TEST(LanczosSolveReferenceTests, TrueResidualPassDouble) {
  check_verified<double>();
}
TEST(LanczosSolveReferenceTests, TrueResidualPassFloat) {
  check_verified<float>();
}

// ── linear_operator model ────────────────────────────────────────────────────

// A linear_operator model and the matvec lambda over the same A take the same
// path: restarts and the true-residual pass included, the same iterations and
// matvecs, one k = 1 apply per matvec, every one in HOST pointer mode.
template<typename T>
void check_operator_model() {
  constexpr int n = 100;
  constexpr int nev = 3;
  constexpr int ncv = 16;
  const auto a = from_spectrum<T>(clustered_spectrum(n), 31u);
  const auto expected = reference_selection(a, n, nev, LanczosWhich::smallest);
  auto options = tight_options<T>();
  options.verify_residuals = true;
  Rig<T> rig(a, n, nev, ncv);

  LanczosInfo by_matvec;
  ASSERT_TRUE(rig.solve(LanczosWhich::smallest, options, &by_matvec).ok());
  SymvOperator<T> op{rig.h.blas, n, rig.d_a.data()};
  LanczosInfo by_operator;
  const Status st = rig.solve(LanczosWhich::smallest, options, op, &by_operator);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(by_operator);
  EXPECT_EQ(by_operator.iterations, by_matvec.iterations);
  EXPECT_EQ(by_operator.matvecs, by_matvec.matvecs);
  EXPECT_EQ(op.applies, by_operator.matvecs);
  EXPECT_TRUE(op.host_mode);

  const auto w = rig.eigenvalues();
  const T tol = factorization_tol<T>(frobenius_norm(a), n, n);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(w[i], expected[i], tol) << i;
  }
}

TEST(LanczosSolveReferenceTests, OperatorModelMatchesMatvecDouble) {
  check_operator_model<double>();
}
TEST(LanczosSolveReferenceTests, OperatorModelMatchesMatvecFloat) {
  check_operator_model<float>();
}

// ── breakdown recovery ───────────────────────────────────────────────────────

// A diagonal A started from e_0: A e_0 = lambda_0 e_0 exactly, so beta_0 = 0 and
// step 0 breaks down. Recovery injects a fresh vector orthogonal to e_0 and runs
// steps 1..ncv-1 again, so nev beyond the invariant subspace still converges.
TEST(LanczosSolveReferenceTests, EarlyBreakdownRecovers) {
  constexpr int n = 16;
  constexpr int ncv = 8;
  const auto nz = static_cast<std::size_t>(n);
  std::vector<double> a(nz * nz, 0.0);
  for (std::size_t i = 0; i < nz; ++i) {
    a[i + i * nz] = -4.0 + static_cast<double>(i);
  }
  std::vector<double> e0(nz, 0.0);
  e0[0] = 3.0; // need not be normalised

  for (const int nev : {1, 2}) {
    SCOPED_TRACE(::testing::Message() << "nev=" << nev);
    Rig<double> rig(a, n, nev, ncv);
    DeviceBuffer<double> d_start(nz, rig.handle);
    rig.upload(d_start.data(), e0);
    auto options = tight_options<double>();
    options.start_vector = d_start.data();
    LanczosInfo info;
    const Status st = rig.solve(LanczosWhich::smallest, options, &info);
    ASSERT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
    EXPECT_CONVERGED(info);
    // The frozen tail after the breakdown, then steps 1..ncv-1 again.
    EXPECT_GE(info.matvecs, ncv + ncv - 1);
    const auto w = rig.eigenvalues();
    const auto x = rig.download(rig.d_x.data(), nz * static_cast<std::size_t>(nev));
    for (int c = 0; c < nev; ++c) {
      const auto cz = static_cast<std::size_t>(c);
      EXPECT_NEAR(w[cz], -4.0 + c, 1e-12);
      EXPECT_NEAR(std::abs(x[cz + cz * nz]), 1.0, 1e-12);
    }
  }
}

/// A diagonal A (both triangles of a dense n x n) with @p lambda on the diagonal.
template<typename T>
std::vector<T> diagonal(const std::vector<double> &lambda) {
  const std::size_t n = lambda.size();
  std::vector<T> a(n * n, T(0));
  for (std::size_t i = 0; i < n; ++i) {
    a[i + i * n] = static_cast<T>(lambda[i]);
  }
  return a;
}

/// Solve the diagonal @p lambda for its nev smallest pairs from @p start.
template<typename T>
Outcome<T> solve_diagonal_from(const std::vector<double> &lambda, const std::vector<T> &start,
                               int nev, int ncv) {
  const int n = static_cast<int>(lambda.size());
  Rig<T> rig(diagonal<T>(lambda), n, nev, ncv);
  DeviceBuffer<T> d_start(start.size(), rig.handle);
  rig.upload(d_start.data(), start);
  auto options = tight_options<T>();
  options.start_vector = d_start.data();
  Outcome<T> result;
  const Status st = rig.solve(LanczosWhich::smallest, options, &result.info);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(result.info);
  result.eigenvalues = rig.eigenvalues();
  return result;
}

// -2 twice, started from e_0: step 0 breaks down on span{e_0}, and the injected
// vector carries an e_1 component, so recovery finds the second copy of -2 --
// what syevd reports.
template<typename T>
void check_recovery_finds_copy() {
  constexpr int n = 16;
  std::vector<double> lambda = {-2.0, -2.0};
  for (int i = 2; i < n; ++i) {
    lambda.push_back(static_cast<double>(i));
  }
  std::vector<T> e0(static_cast<std::size_t>(n), T(0));
  e0[0] = T(1);
  const auto result = solve_diagonal_from<T>(lambda, e0, 2, 8);
  EXPECT_GE(result.info.matvecs, 8 + 7);
  ASSERT_EQ(result.eigenvalues.size(), 2u);
  const auto expected = reference_selection(diagonal<T>(lambda), n, 2, LanczosWhich::smallest);
  const T tol = T(1000) * test::eps<T>();
  EXPECT_NEAR(result.eigenvalues[0], expected[0], tol);
  EXPECT_NEAR(result.eigenvalues[1], expected[1], tol);
}

TEST(LanczosSolveReferenceTests, BreakdownRecoveryFindsCopyDouble) {
  check_recovery_finds_copy<double>();
}
TEST(LanczosSolveReferenceTests, BreakdownRecoveryFindsCopyFloat) {
  check_recovery_finds_copy<float>();
}

// ── the single-vector limitation ─────────────────────────────────────────────

// -5 with multiplicity 3 at the wanted end, then -4.5, -4 over a bulk in [-1, 1].
// A start vector meets a multiple eigenvalue's eigenspace along ONE direction
// (its projection), so the Krylov space never holds a second copy. Modelled
// exactly: A diagonal, the start vector zero on two of -5's three coordinates
// (no rounding can seed them). syevd reports -5, -5, -5; the solve reports -5
// once and the next distinct eigenvalues -- converged, not full multiplicity.
template<typename T>
void check_repeated() {
  constexpr int n = 120;
  constexpr int nev = 3;
  std::vector<double> lambda = {-5.0, -5.0, -5.0, -4.5, -4.0};
  const int bulk = n - static_cast<int>(lambda.size());
  for (int i = 0; i < bulk; ++i) {
    lambda.push_back(-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(bulk - 1));
  }
  std::mt19937 gen(9u);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> start(static_cast<std::size_t>(n));
  for (T &x : start) {
    x = static_cast<T>(dist(gen));
  }
  start[1] = T(0);
  start[2] = T(0);

  const auto result = solve_diagonal_from<T>(lambda, start, nev, 20);
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(nev));
  const auto reference = reference_selection(diagonal<T>(lambda), n, 5, LanczosWhich::smallest);
  EXPECT_EQ(reference[1], T(-5)); // the oracle sees the full multiplicity
  EXPECT_EQ(reference[2], T(-5));
  const T tol = T(1000) * test::eps<T>() * T(5);
  EXPECT_NEAR(result.eigenvalues[0], reference[0], tol);
  EXPECT_NEAR(result.eigenvalues[1], reference[3], tol); // -4.5: the copies are skipped
  EXPECT_NEAR(result.eigenvalues[2], reference[4], tol); // -4
}

TEST(LanczosSolveReferenceTests, RepeatedEigenvalueFoundOnceDouble) {
  check_repeated<double>();
}
TEST(LanczosSolveReferenceTests, RepeatedEigenvalueFoundOnceFloat) {
  check_repeated<float>();
}

TEST(LanczosSolveReferenceTests, RejectsZeroStartVector) {
  constexpr int n = 16;
  const auto a = from_spectrum<double>(outlier_spectrum(n), 3u);
  Rig<double> rig(a, n, 1, 5);
  DeviceBuffer<double> d_start(static_cast<std::size_t>(n), rig.handle);
  rig.upload(d_start.data(), std::vector<double>(static_cast<std::size_t>(n), 0.0));
  LanczosOptions<double> options;
  options.start_vector = d_start.data();
  LanczosInfo info;
  EXPECT_EQ(rig.solve(LanczosWhich::smallest, options, &info).code, kInvalidValue);
  EXPECT_EQ(info.matvecs, 0);
}

} // namespace
} // namespace calaman
