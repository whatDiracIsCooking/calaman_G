// Oracle test for calaman.posv -- the positive definite solve A X = B. posv
// factors on the device (wwr::potrf) and solves (wwr::potrs). Correctness is the
// BACKWARD error of the device X against the original matrix,
// ||A X - B|| / (||A|| ||X|| + ||B||) ~ eps, plus agreement with LAPACKE_?posv's X
// (A = M M^H + n I is well conditioned, so the forward check can be tight).
//
// A second case feeds a matrix whose leading minor of order k is not positive
// definite: the device info must equal the reference's (k) and B must come back
// untouched, since ?posv does not solve against a failed factor.
//
// Shapes span n = 1, 2, odd and larger, both UPLO, all four precisions (complex
// is Hermitian), with nrhs > 1. REQUIRES_GPU (see CMakeLists.txt). Built only
// when calaman::lapack_reference exists.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.posv;
import calaman.common;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;
import calaman.test.shared.tolerance;

namespace calaman {
namespace {

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;
using test::shared_device;
using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;
template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

template<typename T>
struct host_type {
  using type = T;
};
template<>
struct host_type<wwr::wwrFloatComplex> {
  using type = std::complex<float>;
};
template<>
struct host_type<wwr::wwrDoubleComplex> {
  using type = std::complex<double>;
};
template<typename T>
using Host = typename host_type<T>::type;

template<typename T>
struct real_of {
  using type = T;
};
template<>
struct real_of<wwr::wwrFloatComplex> {
  using type = float;
};
template<>
struct real_of<wwr::wwrDoubleComplex> {
  using type = double;
};
template<typename T>
using Real = typename real_of<T>::type;

static_assert(sizeof(std::complex<float>) == sizeof(wwr::wwrFloatComplex));
static_assert(sizeof(std::complex<double>) == sizeof(wwr::wwrDoubleComplex));

template<typename T>
constexpr bool is_complex_v = !std::is_same_v<Host<T>, T>;

template<typename T>
DeviceBuffer<T> upload(std::shared_ptr<DeviceHandle> h, const std::vector<Host<T>> &host) {
  const std::size_t n = host.empty() ? 1 : host.size();
  HostBuffer<T> staging(n);
  std::memcpy(staging.data(), host.data(), host.size() * sizeof(T));
  DeviceBuffer<T> device(n, h);
  wwr::extension::copy(device, staging, h->stream().get());
  wwr::wwrStreamSynchronize(h->stream().get());
  return device;
}

template<typename T>
std::vector<Host<T>> download(std::shared_ptr<DeviceHandle> h, const DeviceBuffer<T> &device,
                              std::size_t n) {
  HostBuffer<T> staging(n == 0 ? 1 : n);
  wwr::extension::copy(staging, device, h->stream().get());
  wwr::wwrStreamSynchronize(h->stream().get());
  std::vector<Host<T>> out(n);
  std::memcpy(out.data(), staging.data(), n * sizeof(T));
  return out;
}

template<typename T>
Host<T> conj_of(const Host<T> &v) {
  if constexpr (is_complex_v<T>) {
    return std::conj(v);
  } else {
    return v;
  }
}
template<typename T>
double abs2(const Host<T> &v) {
  if constexpr (is_complex_v<T>) {
    return std::norm(v);
  } else {
    return static_cast<double>(v) * static_cast<double>(v);
  }
}
template<typename T>
Real<T> fro(const std::vector<Host<T>> &a) {
  double s = 0;
  for (const auto &v : a) {
    s += abs2<T>(v);
  }
  return static_cast<Real<T>>(std::sqrt(s));
}
template<typename T>
Host<T> rnd(std::mt19937 &g) {
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  if constexpr (is_complex_v<T>) {
    return Host<T>(static_cast<Real<T>>(d(g)), static_cast<Real<T>>(d(g)));
  } else {
    return static_cast<Host<T>>(d(g));
  }
}

// A = M M^H + shift I, stored full column-major. shift = n gives a well-
// conditioned HPD matrix; a negative shift makes it indefinite.
template<typename T>
std::vector<Host<T>> hermitian(int n, unsigned seed, double shift) {
  std::mt19937 g(seed);
  std::vector<Host<T>> m(static_cast<std::size_t>(n) * n);
  for (auto &x : m) {
    x = rnd<T>(g);
  }
  std::vector<Host<T>> a(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      Host<T> s{};
      for (int k = 0; k < n; ++k) {
        s += m[static_cast<std::size_t>(k) * n + i] * conj_of<T>(m[static_cast<std::size_t>(k) * n + j]);
      }
      a[static_cast<std::size_t>(j) * n + i] = s;
    }
    a[static_cast<std::size_t>(j) * n + j] =
        Host<T>(static_cast<Real<T>>(std::real(a[static_cast<std::size_t>(j) * n + j]) + shift));
  }
  return a;
}

template<typename T>
std::vector<Host<T>> residual(int n, int nrhs, const std::vector<Host<T>> &a,
                              const std::vector<Host<T>> &x, const std::vector<Host<T>> &b) {
  std::vector<Host<T>> r(static_cast<std::size_t>(n) * nrhs, Host<T>{});
  for (int c = 0; c < nrhs; ++c) {
    for (int i = 0; i < n; ++i) {
      Host<T> s{};
      for (int k = 0; k < n; ++k) {
        s += a[static_cast<std::size_t>(k) * n + i] * x[static_cast<std::size_t>(c) * n + k];
      }
      r[static_cast<std::size_t>(c) * n + i] = s - b[static_cast<std::size_t>(c) * n + i];
    }
  }
  return r;
}

// ── LAPACKE posv, overloaded on the host element type ─────────────────────────
lapack_int lpk_posv(char u, int n, int nrhs, float *a, int lda, float *b, int ldb) {
  return LAPACKE_sposv(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, b, ldb);
}
lapack_int lpk_posv(char u, int n, int nrhs, double *a, int lda, double *b, int ldb) {
  return LAPACKE_dposv(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, b, ldb);
}
lapack_int lpk_posv(char u, int n, int nrhs, std::complex<float> *a, int lda,
                    std::complex<float> *b, int ldb) {
  return LAPACKE_cposv(LAPACK_COL_MAJOR, u, n, nrhs, reinterpret_cast<lapack_complex_float *>(a),
                       lda, reinterpret_cast<lapack_complex_float *>(b), ldb);
}
lapack_int lpk_posv(char u, int n, int nrhs, std::complex<double> *a, int lda,
                    std::complex<double> *b, int ldb) {
  return LAPACKE_zposv(LAPACK_COL_MAJOR, u, n, nrhs, reinterpret_cast<lapack_complex_double *>(a),
                       lda, reinterpret_cast<lapack_complex_double *>(b), ldb);
}

// Run device posv on (a, b); returns the device info and X (as Host<T>).
template<typename T>
std::pair<int, std::vector<Host<T>>> run_device(Uplo uplo, int n, int nrhs,
                                                const std::vector<Host<T>> &a,
                                                const std::vector<Host<T>> &b) {
  auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  EXPECT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  int lwork = 0;
  EXPECT_EQ(posv_bufferSize<T>(solver, uplo, n, &lwork), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = upload<T>(handle, a);
  auto d_b = upload<T>(handle, b);
  DeviceBuffer<int> d_info(1, handle);
  DeviceBuffer<T> d_work(static_cast<std::size_t>(std::max(lwork, 1)), handle);

  const auto status = posv<T>(solver, handle->stream().get(), uplo, n, nrhs, d_a.data(), n,
                              d_b.data(), n, d_work.data(), lwork, d_info.data());
  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " nrhs=" << nrhs;

  const auto info = download<int>(handle, d_info, 1);
  wwr::wwrsolverDnDestroy(solver);
  return {info[0], download<T>(handle, d_b, static_cast<std::size_t>(n) * nrhs)};
}

template<typename T>
std::vector<Host<T>> random_rhs(int n, int nrhs, unsigned seed) {
  std::vector<Host<T>> b(static_cast<std::size_t>(n) * nrhs);
  std::mt19937 g(seed);
  for (auto &x : b) {
    x = rnd<T>(g);
  }
  return b;
}

template<typename T>
void expect_solves(Uplo uplo, int n, int nrhs, unsigned seed) {
  const char cuplo = uplo == Uplo::U ? 'U' : 'L';
  using R = Real<T>;

  const auto a_orig = hermitian<T>(n, seed, static_cast<double>(n));
  const auto b = random_rhs<T>(n, nrhs, seed + 777u);

  std::vector<Host<T>> ref_a = a_orig;
  std::vector<Host<T>> x_ref = b;
  const lapack_int ri = lpk_posv(cuplo, n, nrhs, ref_a.data(), n, x_ref.data(), n);
  ASSERT_EQ(ri, 0) << "LAPACKE ?posv failed n=" << n << " uplo=" << cuplo << " seed=" << seed;

  const auto [info, x_dev] = run_device<T>(uplo, n, nrhs, a_orig, b);
  EXPECT_EQ(info, 0) << "device info n=" << n << " uplo=" << cuplo << " seed=" << seed;

  // 1. Backward error of the device solution against the original matrix.
  const auto r = residual<T>(n, nrhs, a_orig, x_dev, b);
  const R norm_a = fro<T>(a_orig);
  const R norm_x = fro<T>(x_dev);
  const R norm_b = fro<T>(b);
  const R bound = R{64} * eps<R>() * static_cast<R>(n) * (norm_a * norm_x + norm_b);
  EXPECT_LE(fro<T>(r), bound) << "backward error n=" << n << " nrhs=" << nrhs << " uplo=" << cuplo
                              << " seed=" << seed;

  // 2. Agreement with the reference solution (A is well conditioned).
  std::vector<Host<T>> diff(x_dev.size());
  for (std::size_t i = 0; i < x_dev.size(); ++i) {
    diff[i] = x_dev[i] - x_ref[i];
  }
  const R ref_bound = R{256} * eps<R>() * static_cast<R>(n) * (norm_x + R{1});
  EXPECT_LE(fro<T>(diff), ref_bound) << "vs LAPACKE n=" << n << " nrhs=" << nrhs;
}

// A not positive definite: same info as the reference, B untouched.
template<typename T>
void expect_not_pd(Uplo uplo, int n, int nrhs, unsigned seed) {
  const char cuplo = uplo == Uplo::U ? 'U' : 'L';
  const auto a_orig = hermitian<T>(n, seed, -0.5 * n);
  const auto b = random_rhs<T>(n, nrhs, seed + 777u);

  std::vector<Host<T>> ref_a = a_orig;
  std::vector<Host<T>> ref_b = b;
  const lapack_int ri = lpk_posv(cuplo, n, nrhs, ref_a.data(), n, ref_b.data(), n);
  ASSERT_GT(ri, 0) << "test matrix unexpectedly PD n=" << n << " seed=" << seed;

  const auto [info, x_dev] = run_device<T>(uplo, n, nrhs, a_orig, b);
  EXPECT_EQ(info, ri) << "info n=" << n << " uplo=" << cuplo << " seed=" << seed;
  EXPECT_EQ(std::memcmp(x_dev.data(), b.data(), b.size() * sizeof(T)), 0)
      << "B modified on a failed factor n=" << n << " uplo=" << cuplo;
}

template<typename T>
void run_all(unsigned base) {
  for (const Uplo uplo : {Uplo::L, Uplo::U}) {
    expect_solves<T>(uplo, 1, 1, base + 1);
    expect_solves<T>(uplo, 2, 3, base + 2);
    expect_solves<T>(uplo, 5, 2, base + 3);
    expect_solves<T>(uplo, 7, 4, base + 4);
    expect_solves<T>(uplo, 16, 5, base + 5);
    expect_solves<T>(uplo, 33, 2, base + 6);
    expect_solves<T>(uplo, 64, 3, base + 7);
    expect_solves<T>(uplo, 130, 2, base + 8);
    expect_not_pd<T>(uplo, 9, 2, base + 9);
    expect_not_pd<T>(uplo, 40, 3, base + 10);
  }
}

TEST(PosvOracleTests, Float) {
  run_all<float>(100);
}
TEST(PosvOracleTests, Double) {
  run_all<double>(200);
}
TEST(PosvOracleTests, ComplexFloat) {
  run_all<wwr::wwrFloatComplex>(300);
}
TEST(PosvOracleTests, ComplexDouble) {
  run_all<wwr::wwrDoubleComplex>(400);
}

TEST(PosvOracleTests, Arguments) {
  auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  const auto s = handle->stream().get();
  int lwork = -1;
  EXPECT_EQ(posv_bufferSize<double>(solver, Uplo::L, -1, &lwork), wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(posv_bufferSize<double>(solver, Uplo::L, 0, &lwork), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(lwork, 0);
  EXPECT_EQ(posv<double>(solver, s, Uplo::L, 0, 1, nullptr, 1, nullptr, 1, nullptr, 0, nullptr),
            wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(posv<double>(solver, s, Uplo::L, 4, 1, nullptr, 3, nullptr, 4, nullptr, 0, nullptr),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(posv<double>(solver, s, Uplo::L, 4, -1, nullptr, 4, nullptr, 4, nullptr, 0, nullptr),
            wwr::WWRBLAS_STATUS_INVALID_VALUE);
  wwr::wwrsolverDnDestroy(solver);
}

} // namespace
} // namespace calaman
