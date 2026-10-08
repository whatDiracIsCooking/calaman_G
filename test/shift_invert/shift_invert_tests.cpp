// Oracle suite for calaman.shift_invert -- DenseShiftInvert, the linear_operator
// Y = (A - sigma I)^{-1} X. The oracle is LAPACKE_?sysv on the shifted matrix.
// sigma sits midway between two interior eigenvalues of a random symmetric A,
// so A - sigma I is indefinite (2x2 pivots) and its condition number kappa is
// known exactly from LAPACKE_dsyev: the forward error against the reference is
// held to the shared tolerance scaled by kappa, and the backward error
// ||(A - sigma I) Y - X|| to the shared tolerance unscaled.
//
// REQUIRES_GPU (see CMakeLists.txt). Built only when calaman::lapack_reference
// exists; its CMakeLists.txt returns early otherwise.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.shift_invert;
import calaman.common;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;
import calaman.test.shared.tolerance;

namespace calaman {
namespace {

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;
using test::shared_device;
using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;
template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

static_assert(linear_operator<DenseShiftInvert<float>, float>);
static_assert(linear_operator<DenseShiftInvert<double>, double>);

template<typename T>
DeviceBuffer<T> upload(std::shared_ptr<DeviceHandle> h, const std::vector<T> &host) {
  const std::size_t n = host.empty() ? 1 : host.size();
  HostBuffer<T> staging(n);
  std::memcpy(staging.data(), host.data(), host.size() * sizeof(T));
  DeviceBuffer<T> device(n, h);
  wwr::extension::copy(device, staging, h->stream().get());
  wwr::wwrStreamSynchronize(h->stream().get());
  return device;
}

template<typename T>
std::vector<T> download(std::shared_ptr<DeviceHandle> h, const DeviceBuffer<T> &device,
                        std::size_t n) {
  HostBuffer<T> staging(n == 0 ? 1 : n);
  wwr::extension::copy(staging, device, h->stream().get());
  wwr::wwrStreamSynchronize(h->stream().get());
  std::vector<T> out(n);
  std::memcpy(out.data(), staging.data(), n * sizeof(T));
  return out;
}

struct Handles {
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};

  explicit Handles(const std::shared_ptr<DeviceHandle> &h) {
    EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrblasSetStream(blas, h->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrsolverDnSetStream(solver, h->stream().get()), wwr::WWRSOLVER_STATUS_SUCCESS);
  }
  ~Handles() {
    wwr::wwrblasDestroy(blas);
    wwr::wwrsolverDnDestroy(solver);
  }
  Handles(const Handles &) = delete;
  Handles &operator=(const Handles &) = delete;
};

// The model's workspace: sized by the bufferSize query, carved by make_*_slices.
template<typename T>
struct Workspace {
  DeviceBuffer<double> buffer;
  DenseShiftInvertSlices<T> slices;
};

template<typename T>
std::unique_ptr<Workspace<T>> make_workspace(std::shared_ptr<DeviceHandle> h, const Handles &hs,
                                             int n) {
  std::size_t bytes = 0;
  EXPECT_EQ(dense_shift_invert_bufferSize<T>(hs.solver, n, &bytes), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_GT(bytes, 0u);
  auto ws = std::unique_ptr<Workspace<T>>(
      new Workspace<T>{DeviceBuffer<double>(bytes / sizeof(double) + 1, h), {}});
  std::size_t carved = 0;
  EXPECT_EQ(
      make_dense_shift_invert_slices<T>(hs.solver, n, ws->buffer.data(), &ws->slices, &carved),
      wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(carved, bytes) << "sizing and carving disagree";
  return ws;
}

// Random symmetric A = M + M^T, column-major at leading dimension lda; the
// padding rows hold a sentinel the model must never read.
std::vector<double> sym_indef(int n, int lda, unsigned seed) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> m(static_cast<std::size_t>(n) * n);
  for (auto &x : m) {
    x = d(g);
  }
  std::vector<double> a(static_cast<std::size_t>(lda) * n, 1.0e30);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<std::size_t>(j) * lda + i] =
          m[static_cast<std::size_t>(j) * n + i] + m[static_cast<std::size_t>(i) * n + j];
    }
  }
  return a;
}

lapack_int lpk_sysv(char u, int n, int nrhs, float *a, int lda, lapack_int *ip, float *b, int ldb) {
  return LAPACKE_ssysv(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, ip, b, ldb);
}
lapack_int lpk_sysv(char u, int n, int nrhs, double *a, int lda, lapack_int *ip, double *b,
                    int ldb) {
  return LAPACKE_dsysv(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, ip, b, ldb);
}

template<typename T>
T fro(const std::vector<T> &v) {
  double s = 0;
  for (const T x : v) {
    s += static_cast<double>(x) * static_cast<double>(x);
  }
  return static_cast<T>(std::sqrt(s));
}

// One prepare, then apply for each k in @p ks: Y against LAPACKE_?sysv on the
// shifted matrix (forward, kappa-scaled) and against X (backward).
template<typename T>
void expect_matches_sysv(Uplo uplo, int n, std::initializer_list<int> ks, unsigned seed) {
  const int lda = n + 3;
  const char cuplo = uplo == Uplo::U ? 'U' : 'L';
  const auto a_dbl = sym_indef(n, lda, seed);
  std::vector<T> a(a_dbl.begin(), a_dbl.end());

  // The spectrum of the T-rounded A, in double, places sigma and gives kappa.
  std::vector<double> a_eig(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a_eig[static_cast<std::size_t>(j) * n + i] =
          static_cast<double>(a[static_cast<std::size_t>(j) * lda + i]);
    }
  }
  std::vector<double> w(static_cast<std::size_t>(n));
  ASSERT_EQ(LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'L', n, a_eig.data(), n, w.data()), 0);
  const int mid = n / 2;
  const double sigma_dbl = n == 1 ? w[0] + 0.5 : 0.5 * (w[mid - 1] + w[mid]);
  const T sigma = static_cast<T>(sigma_dbl);
  double gap_min = std::numeric_limits<double>::max();
  double gap_max = 0;
  for (const double lam : w) {
    gap_min = std::min(gap_min, std::abs(lam - static_cast<double>(sigma)));
    gap_max = std::max(gap_max, std::abs(lam - static_cast<double>(sigma)));
  }
  ASSERT_GT(gap_min, 0.0);
  const T kappa = static_cast<T>(gap_max / gap_min);

  // The shifted matrix, dense at ld n, for the reference solve and the residual.
  std::vector<T> shifted(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      shifted[static_cast<std::size_t>(j) * n + i] =
          a[static_cast<std::size_t>(j) * lda + i] - (i == j ? sigma : T{0});
    }
  }
  const T norm_shifted = fro(shifted);

  auto h = shared_device();
  const Handles hs{h};
  auto ws = make_workspace<T>(h, hs, n);
  auto d_a = upload(h, a);
  DenseShiftInvert<T> op{hs.blas, hs.solver, uplo, n, d_a.data(), lda, sigma, ws->slices};
  EXPECT_EQ(op.sigma(), sigma);
  EXPECT_EQ(op.dim(), n);
  ASSERT_EQ(op.prepare(h->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS)
      << "n=" << n << " uplo=" << cuplo << " seed=" << seed;

  unsigned rhs_seed = seed * 31u;
  for (const int k : ks) {
    const std::size_t nk = static_cast<std::size_t>(n) * k;
    std::vector<T> x(nk);
    {
      std::mt19937 g(++rhs_seed);
      std::uniform_real_distribution<double> d(-1.0, 1.0);
      for (auto &v : x) {
        v = static_cast<T>(d(g));
      }
    }
    auto d_x = upload(h, x);
    DeviceBuffer<T> d_y(nk, h);
    ASSERT_EQ(op.apply(h->stream().get(), k, d_x.data(), d_y.data()), wwr::WWRBLAS_STATUS_SUCCESS)
        << "n=" << n << " k=" << k;
    const auto y = download(h, d_y, nk);
    EXPECT_EQ(download(h, d_x, nk), x) << "apply wrote its input";

    std::vector<T> ref_a = shifted;
    std::vector<T> y_ref = x;
    std::vector<lapack_int> ipiv(static_cast<std::size_t>(n));
    ASSERT_EQ(lpk_sysv(cuplo, n, k, ref_a.data(), n, ipiv.data(), y_ref.data(), n), 0);

    std::vector<T> diff(nk);
    std::vector<T> resid(nk);
    for (int c = 0; c < k; ++c) {
      for (int i = 0; i < n; ++i) {
        const std::size_t ic = static_cast<std::size_t>(c) * n + i;
        diff[ic] = y[ic] - y_ref[ic];
        double s = 0;
        for (int l = 0; l < n; ++l) {
          s += static_cast<double>(shifted[static_cast<std::size_t>(l) * n + i]) *
               static_cast<double>(y[static_cast<std::size_t>(c) * n + l]);
        }
        resid[ic] = static_cast<T>(s - static_cast<double>(x[ic]));
      }
    }
    EXPECT_LE(fro(diff), factorization_tol<T>(kappa * fro(y_ref), n, n))
        << "vs LAPACKE n=" << n << " k=" << k << " uplo=" << cuplo << " kappa=" << kappa;
    EXPECT_LE(fro(resid), factorization_tol<T>(norm_shifted * fro(y) + fro(x), n, n))
        << "backward error n=" << n << " k=" << k << " uplo=" << cuplo;
  }
}

template<typename T>
void run_oracle(unsigned base) {
  for (const Uplo uplo : {Uplo::L, Uplo::U}) {
    expect_matches_sysv<T>(uplo, 1, {1, 2}, base + 1);
    expect_matches_sysv<T>(uplo, 5, {1, 3}, base + 2);
    expect_matches_sysv<T>(uplo, 16, {1, 4, 8}, base + 3);
    expect_matches_sysv<T>(uplo, 33, {2, 5}, base + 4);
    expect_matches_sysv<T>(uplo, 64, {1, 3, 16}, base + 5);
  }
}

TEST(ShiftInvertOracleTests, Float) {
  run_oracle<float>(100);
}
TEST(ShiftInvertOracleTests, Double) {
  run_oracle<double>(200);
}

// A shift on an eigenvalue of a diagonal A makes A - sigma I exactly singular:
// prepare reports it, and apply refuses to run against the failed factor.
template<typename T>
void expect_singular_shift(Uplo uplo) {
  const int n = 6;
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    a[static_cast<std::size_t>(i) * n + i] = static_cast<T>(i + 1);
  }
  auto h = shared_device();
  const Handles hs{h};
  auto ws = make_workspace<T>(h, hs, n);
  auto d_a = upload(h, a);
  DenseShiftInvert<T> op{hs.blas, hs.solver, uplo, n, d_a.data(), n, T{3}, ws->slices};
  EXPECT_FALSE(op.prepare(h->stream().get()).ok());

  DeviceBuffer<T> d_x(n, h);
  DeviceBuffer<T> d_y(n, h);
  EXPECT_EQ(op.apply(h->stream().get(), 1, d_x.data(), d_y.data()),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
}

TEST(ShiftInvertOracleTests, SingularShiftFloat) {
  expect_singular_shift<float>(Uplo::L);
  expect_singular_shift<float>(Uplo::U);
}
TEST(ShiftInvertOracleTests, SingularShiftDouble) {
  expect_singular_shift<double>(Uplo::L);
  expect_singular_shift<double>(Uplo::U);
}

TEST(ShiftInvertOracleTests, ArgumentChecks) {
  auto h = shared_device();
  const Handles hs{h};
  std::size_t bytes = 0;
  EXPECT_EQ(dense_shift_invert_bufferSize<double>(hs.solver, 4, nullptr),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(dense_shift_invert_bufferSize<double>(hs.solver, 0, &bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);

  auto ws = make_workspace<double>(h, hs, 4);
  DeviceBuffer<double> d_a(16, h);
  DeviceBuffer<double> d_v(8, h);
  DenseShiftInvert<double> op{hs.blas, hs.solver, Uplo::L, 4, d_a.data(), 4, 0.5, ws->slices};
  EXPECT_EQ(op.apply(h->stream().get(), 1, d_v.data(), d_v.data() + 4),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "apply before prepare";

  DenseShiftInvert<double> wrong_n{hs.blas, hs.solver, Uplo::L, 5, d_a.data(), 5, 0.5, ws->slices};
  EXPECT_EQ(wrong_n.prepare(h->stream().get()), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "slices carved for another n";
  DenseShiftInvert<double> bad_lda{hs.blas, hs.solver, Uplo::L, 4, d_a.data(), 3, 0.5, ws->slices};
  EXPECT_EQ(bad_lda.prepare(h->stream().get()), wwr::WWRBLAS_STATUS_INVALID_VALUE);
}

} // namespace
} // namespace calaman
