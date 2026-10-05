// Oracle test for calaman.sysv -- the symmetric indefinite solve A X = B. Unlike
// the sytrs suite, the factorization IS exercised here: sysv factors on the
// device (wwr::sytrf, the vendor Bunch-Kaufman) and solves with calaman.sytrs.
// The device factor differs from LAPACKE's bit-for-bit but the system's solution
// is unique, so correctness is the BACKWARD error of the device X against the
// original matrix: ||A X - B|| / (||A|| ||X|| + ||B||) ~ eps, condition-number-
// independent. The reference LAPACKE_?sysv X is compared too, loosely.
//
// Shapes force 2x2 pivots (indefinite A) and span n = 1, 2, odd and larger, both
// UPLO, both real precisions and both COMPLEX-SYMMETRIC precisions (sysv is
// symmetric, not Hermitian), with nrhs > 1 so the per-column applies see columns.
//
// REQUIRES_GPU (see CMakeLists.txt): every case factors and solves on the device,
// so `ctest -LE gpu` excludes it. Built only when calaman::lapack_reference
// exists; its CMakeLists.txt returns early otherwise.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.sysv;
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

// A = M + M^T (symmetric, generically indefinite). Stored full column-major.
template<typename T>
std::vector<Host<T>> sym_indef(int n, unsigned seed) {
  std::mt19937 g(seed);
  std::vector<Host<T>> m(static_cast<std::size_t>(n) * n);
  for (auto &x : m) {
    x = rnd<T>(g);
  }
  std::vector<Host<T>> a(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      a[static_cast<std::size_t>(j) * n + i] =
          m[static_cast<std::size_t>(j) * n + i] + m[static_cast<std::size_t>(i) * n + j];
    }
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

// ── LAPACKE sysv, overloaded on the host element type (for the loose check) ────
lapack_int lpk_sysv(char u, int n, int nrhs, float *a, int lda, lapack_int *ip, float *b, int ldb) {
  return LAPACKE_ssysv(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, ip, b, ldb);
}
lapack_int lpk_sysv(char u, int n, int nrhs, double *a, int lda, lapack_int *ip, double *b,
                    int ldb) {
  return LAPACKE_dsysv(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, ip, b, ldb);
}
lapack_int lpk_sysv(char u, int n, int nrhs, std::complex<float> *a, int lda, lapack_int *ip,
                    std::complex<float> *b, int ldb) {
  return LAPACKE_csysv(LAPACK_COL_MAJOR, u, n, nrhs, reinterpret_cast<lapack_complex_float *>(a),
                       lda, ip, reinterpret_cast<lapack_complex_float *>(b), ldb);
}
lapack_int lpk_sysv(char u, int n, int nrhs, std::complex<double> *a, int lda, lapack_int *ip,
                    std::complex<double> *b, int ldb) {
  return LAPACKE_zsysv(LAPACK_COL_MAJOR, u, n, nrhs, reinterpret_cast<lapack_complex_double *>(a),
                       lda, ip, reinterpret_cast<lapack_complex_double *>(b), ldb);
}

template<typename T>
void expect_solves(Uplo uplo, int n, int nrhs, unsigned seed) {
  const int lda = n;
  const int ldb = n;
  const char cuplo = uplo == Uplo::U ? 'U' : 'L';
  using R = Real<T>;

  const auto a_orig = sym_indef<T>(n, seed);
  std::vector<Host<T>> b(static_cast<std::size_t>(n) * nrhs);
  {
    std::mt19937 g(seed + 777u);
    for (auto &x : b) {
      x = rnd<T>(g);
    }
  }

  // Reference sysv (on copies) -- the loose forward comparison.
  std::vector<Host<T>> ref_a = a_orig;
  std::vector<Host<T>> x_ref = b;
  std::vector<lapack_int> ref_ipiv(static_cast<std::size_t>(n), 0);
  const lapack_int ri =
      lpk_sysv(cuplo, n, nrhs, ref_a.data(), lda, ref_ipiv.data(), x_ref.data(), ldb);
  ASSERT_EQ(ri, 0) << "LAPACKE ?sysv failed n=" << n << " uplo=" << cuplo << " seed=" << seed;

  // Device sysv.
  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  int lwork = 0;
  ASSERT_EQ(sysv_bufferSize<T>(solver, n, &lwork), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = upload<T>(handle, a_orig);
  auto d_b = upload<T>(handle, b);
  DeviceBuffer<int> d_ipiv(static_cast<std::size_t>(n), handle);
  DeviceBuffer<int> d_info(1, handle);
  DeviceBuffer<T> d_work(static_cast<std::size_t>(std::max(lwork, 1)), handle);

  const auto status = sysv<T>(blas, solver, handle->stream().get(), uplo, n, nrhs, d_a.data(), lda,
                              d_ipiv.data(), d_b.data(), ldb, d_work.data(), lwork, d_info.data());
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " nrhs=" << nrhs;

  const auto info = download<int>(handle, d_info, 1);
  EXPECT_EQ(info[0], 0) << "device sytrf info n=" << n << " uplo=" << cuplo << " seed=" << seed;

  wwr::wwrblasDestroy(blas);
  wwr::wwrsolverDnDestroy(solver);

  const auto x_dev = download<T>(handle, d_b, static_cast<std::size_t>(n) * nrhs);

  // 1. Backward error of the device solution against the original matrix.
  const auto r = residual<T>(n, nrhs, a_orig, x_dev, b);
  const R norm_a = fro<T>(a_orig);
  const R norm_x = fro<T>(x_dev);
  const R norm_b = fro<T>(b);
  const R bound = R{128} * eps<R>() * static_cast<R>(n) * (norm_a * norm_x + norm_b);
  EXPECT_LE(fro<T>(r), bound) << "backward error n=" << n << " nrhs=" << nrhs << " uplo=" << cuplo
                              << " seed=" << seed;

  // 2. Loose agreement with the reference solution (different factor, same system).
  std::vector<Host<T>> diff(x_dev.size());
  for (std::size_t i = 0; i < x_dev.size(); ++i) {
    diff[i] = x_dev[i] - x_ref[i];
  }
  const R ref_bound = R{4096} * eps<R>() * static_cast<R>(n) * (norm_x + R{1});
  EXPECT_LE(fro<T>(diff), ref_bound) << "vs LAPACKE n=" << n << " nrhs=" << nrhs;
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
  }
}

TEST(SysvOracleTests, Float) {
  run_all<float>(100);
}
TEST(SysvOracleTests, Double) {
  run_all<double>(200);
}
TEST(SysvOracleTests, ComplexFloat) {
  run_all<wwr::wwrFloatComplex>(300);
}
TEST(SysvOracleTests, ComplexDouble) {
  run_all<wwr::wwrDoubleComplex>(400);
}

} // namespace
} // namespace calaman
