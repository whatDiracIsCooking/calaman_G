// Reference suite for calaman.lanczos's single-cycle lanczos_solve. The operator
// is a dense A = Q diag(lambda) Q^T on the device, applied through a
// symv-backed LanczosMatvecFn adapter; the oracle is LAPACKE_?syevd on the same
// A. Checked, for smallest / largest / both_ends, float and double:
//
//   * one cycle converges: converged, restarts == 0, matvecs == ncv;
//   * the selected eigenvalues match the reference's to the shared tolerance;
//   * the Ritz vectors: ||A X - X Theta||_F and ||X^T X - I||_F to it.
//
// Plus the edge cases: ncv == n (the Krylov space is all of R^n, so every Ritz
// pair is exact), an invariant subspace found at step 0 (the early-breakdown
// path: a diagonal A from e_0), a too-small invariant subspace reported as
// non-converged, and a zero start vector rejected. REQUIRES_GPU.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

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
constexpr int kInternalError = static_cast<int>(wwr::WWRBLAS_STATUS_INTERNAL_ERROR);

// ── the matvec adapter ───────────────────────────────────────────────────────

/// y = A x for a dense symmetric n x n device matrix (lower triangle read).
/// lanczos_solve calls the matvec in HOST pointer mode, so host scalars.
template<typename T>
LanczosMatvecFn<T> symv_matvec(wwr::wwrblasHandle_t blas, int n, const T *d_a) {
  return [blas, n, d_a](wwr::wwrStream_t, const T *x, T *y) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::symv<T, int>(blas, wwr::WWRBLAS_FILL_MODE_LOWER, n, &one, d_a, n, x, 1, &zero, y,
                             1);
  };
}

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
  std::unique_ptr<DeviceBuffer<std::byte>> work;
  LanczosSlices<T> s;

  Rig(const std::vector<T> &a, int n_, int nev_, int ncv_)
      : n(n_), nev(nev_), ncv(ncv_), d_a(a.size(), handle),
        d_x(static_cast<std::size_t>(n_) * static_cast<std::size_t>(nev_), handle) {
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

  Status solve(LanczosWhich which, const LanczosOptions<T> &options, LanczosResult<T> *result) {
    return lanczos_solve<T>(h.blas, h.solver, stream, n, nev, ncv, which, s,
                            symv_matvec<T>(h.blas, n, d_a.data()), d_x.data(), result, options);
  }
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

/// One converged single-cycle solve, checked against LAPACKE_?syevd on @p a.
template<typename T>
void check_solve(const std::vector<T> &a, int n, int nev, int ncv, LanczosWhich which) {
  SCOPED_TRACE(::testing::Message() << "which=" << name(which) << " n=" << n << " nev=" << nev
                                    << " ncv=" << ncv << " sizeof(T)=" << sizeof(T));
  std::vector<T> w(static_cast<std::size_t>(n));
  std::vector<T> a_copy = a;
  ASSERT_EQ(ref_syevd(n, a_copy.data(), w.data()), 0);
  std::vector<T> expected;
  for (const int i : lanczos_select(which, n, nev)) {
    expected.push_back(w[static_cast<std::size_t>(i)]);
  }

  Rig<T> rig(a, n, nev, ncv);
  LanczosOptions<T> options;
  options.tolerance = T(1000) * test::eps<T>();
  LanczosResult<T> result;
  const Status st = rig.solve(which, options, &result);
  ASSERT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.restarts, 0);
  EXPECT_EQ(result.matvecs, ncv);
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(nev));

  const T tol = factorization_tol<T>(frobenius_norm(a), n, n);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.eigenvalues[i], expected[i], tol) << i;
  }
  const auto x = rig.download(rig.d_x.data(), static_cast<std::size_t>(n) * nev);
  const auto [res, orth] = ritz_errors(a, x, result.eigenvalues, n);
  EXPECT_LE(res, static_cast<double>(tol));
  EXPECT_LE(orth, static_cast<double>(factorization_tol<T>(T(1), n, n)));
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

// A diagonal A started from e_0: A e_0 = lambda_0 e_0 exactly, so beta_0 = 0 and
// step 0 breaks down. The active dimension is 1: lambda_0 is exact, and a
// request for more pairs than the invariant subspace holds is not converged.
TEST(LanczosSolveReferenceTests, EarlyBreakdownInvariantSubspace) {
  constexpr int n = 16;
  constexpr int ncv = 8;
  const auto nz = static_cast<std::size_t>(n);
  std::vector<double> a(nz * nz, 0.0);
  for (std::size_t i = 0; i < nz; ++i) {
    a[i + i * nz] = -4.0 + static_cast<double>(i);
  }
  std::vector<double> e0(nz, 0.0);
  e0[0] = 3.0; // need not be normalised

  {
    Rig<double> rig(a, n, 1, ncv);
    DeviceBuffer<double> d_start(nz, rig.handle);
    rig.upload(d_start.data(), e0);
    LanczosOptions<double> options;
    options.start_vector = d_start.data();
    LanczosResult<double> result;
    const Status st = rig.solve(LanczosWhich::smallest, options, &result);
    ASSERT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.matvecs, ncv); // the steps after the breakdown still run
    ASSERT_EQ(result.eigenvalues.size(), 1u);
    EXPECT_EQ(result.eigenvalues[0], -4.0);
    const auto x = rig.download(rig.d_x.data(), nz);
    EXPECT_EQ(std::abs(x[0]), 1.0);
    for (std::size_t i = 1; i < nz; ++i) {
      EXPECT_EQ(x[i], 0.0) << i;
    }
  }

  {
    Rig<double> rig(a, n, 2, ncv);
    DeviceBuffer<double> d_start(nz, rig.handle);
    rig.upload(d_start.data(), e0);
    LanczosOptions<double> options;
    options.start_vector = d_start.data();
    LanczosResult<double> result;
    EXPECT_EQ(rig.solve(LanczosWhich::smallest, options, &result).code, kInternalError);
    EXPECT_FALSE(result.converged);
    ASSERT_EQ(result.eigenvalues.size(), 1u); // the one pair the subspace holds
    EXPECT_EQ(result.eigenvalues[0], -4.0);

    options.fail_on_non_convergence = false;
    EXPECT_TRUE(rig.solve(LanczosWhich::smallest, options, &result).ok());
    EXPECT_FALSE(result.converged);
  }
}

TEST(LanczosSolveReferenceTests, RejectsZeroStartVector) {
  constexpr int n = 16;
  const auto a = from_spectrum<double>(outlier_spectrum(n), 3u);
  Rig<double> rig(a, n, 1, 5);
  DeviceBuffer<double> d_start(static_cast<std::size_t>(n), rig.handle);
  rig.upload(d_start.data(), std::vector<double>(static_cast<std::size_t>(n), 0.0));
  LanczosOptions<double> options;
  options.start_vector = d_start.data();
  LanczosResult<double> result;
  EXPECT_EQ(rig.solve(LanczosWhich::smallest, options, &result).code, kInvalidValue);
  EXPECT_EQ(result.matvecs, 0);
}

} // namespace
} // namespace calaman
