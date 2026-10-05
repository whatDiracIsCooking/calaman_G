// Oracle test for calaman.sytrs2 -- the level-3 symmetric Bunch-Kaufman solve.
// Same contract as the sytrs suite: the factor + pivots come from LAPACKE_?sytrf,
// and the device sytrs2 must reconstruct a solution with a small BACKWARD error
// ||A_orig X - B|| / (||A|| ||X|| + ||B||) ~ eps. LAPACKE exposes no ?sytrs2
// (it is internal to ?sysv), so the forward oracle is LAPACKE_?sytrs from the same
// factor -- which computes the identical X. sytrs2 also CONVERTS the factor in
// place and REVERTS it, so the suite additionally asserts the factor is unchanged
// on return (the ?syconv round-trip moves values, no arithmetic, so it is exact).
//
// Shapes force 2x2 pivots (indefinite A) and span n = 1, 2, odd and larger, both
// UPLO, all four element types (sytrs2 is symmetric, not Hermitian), nrhs > 1.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/ipiv/B/E on the device
// and runs the convert/trsm/solve kernels, so `ctest -LE gpu` excludes it. Built
// only when calaman::lapack_reference exists.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.sytrs2;
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
constexpr bool is_complex = !std::is_same_v<Host<T>, T>;

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
  if constexpr (is_complex<T>) {
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
  if constexpr (is_complex<T>) {
    return Host<T>(static_cast<Real<T>>(d(g)), static_cast<Real<T>>(d(g)));
  } else {
    return static_cast<Host<T>>(d(g));
  }
}

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

// ── LAPACKE sytrf / sytrs (the forward oracle), overloaded on the host type ────
lapack_int lpk_sytrf(char u, int n, float *a, int lda, lapack_int *ip) {
  return LAPACKE_ssytrf(LAPACK_COL_MAJOR, u, n, a, lda, ip);
}
lapack_int lpk_sytrf(char u, int n, double *a, int lda, lapack_int *ip) {
  return LAPACKE_dsytrf(LAPACK_COL_MAJOR, u, n, a, lda, ip);
}
lapack_int lpk_sytrf(char u, int n, std::complex<float> *a, int lda, lapack_int *ip) {
  return LAPACKE_csytrf(LAPACK_COL_MAJOR, u, n, reinterpret_cast<lapack_complex_float *>(a), lda,
                        ip);
}
lapack_int lpk_sytrf(char u, int n, std::complex<double> *a, int lda, lapack_int *ip) {
  return LAPACKE_zsytrf(LAPACK_COL_MAJOR, u, n, reinterpret_cast<lapack_complex_double *>(a), lda,
                        ip);
}

lapack_int lpk_sytrs(char u, int n, int nrhs, const float *a, int lda, const lapack_int *ip,
                     float *b, int ldb) {
  return LAPACKE_ssytrs(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, ip, b, ldb);
}
lapack_int lpk_sytrs(char u, int n, int nrhs, const double *a, int lda, const lapack_int *ip,
                     double *b, int ldb) {
  return LAPACKE_dsytrs(LAPACK_COL_MAJOR, u, n, nrhs, a, lda, ip, b, ldb);
}
lapack_int lpk_sytrs(char u, int n, int nrhs, const std::complex<float> *a, int lda,
                     const lapack_int *ip, std::complex<float> *b, int ldb) {
  return LAPACKE_csytrs(LAPACK_COL_MAJOR, u, n, nrhs,
                        reinterpret_cast<const lapack_complex_float *>(a), lda, ip,
                        reinterpret_cast<lapack_complex_float *>(b), ldb);
}
lapack_int lpk_sytrs(char u, int n, int nrhs, const std::complex<double> *a, int lda,
                     const lapack_int *ip, std::complex<double> *b, int ldb) {
  return LAPACKE_zsytrs(LAPACK_COL_MAJOR, u, n, nrhs,
                        reinterpret_cast<const lapack_complex_double *>(a), lda, ip,
                        reinterpret_cast<lapack_complex_double *>(b), ldb);
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

  std::vector<Host<T>> factor = a_orig;
  std::vector<lapack_int> ipiv(static_cast<std::size_t>(n), 0);
  const lapack_int fi = lpk_sytrf(cuplo, n, factor.data(), lda, ipiv.data());
  ASSERT_EQ(fi, 0) << "LAPACKE ?sytrf failed n=" << n << " uplo=" << cuplo << " seed=" << seed;

  std::vector<Host<T>> x_ref = b;
  const lapack_int si =
      lpk_sytrs(cuplo, n, nrhs, factor.data(), lda, ipiv.data(), x_ref.data(), ldb);
  ASSERT_EQ(si, 0) << "LAPACKE ?sytrs failed";

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = upload<T>(handle, factor);
  std::vector<int> ipiv_i(ipiv.begin(), ipiv.end());
  auto d_ipiv = upload<int>(handle, ipiv_i);
  auto d_b = upload<T>(handle, b);
  DeviceBuffer<T> d_work(static_cast<std::size_t>(std::max(n, 1)), handle);

  const auto status = sytrs2<T>(blas, uplo, n, nrhs, d_a.data(), lda, d_ipiv.data(), d_b.data(),
                                ldb, d_work.data());
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " nrhs=" << nrhs;
  wwr::wwrblasDestroy(blas);

  const auto x_dev = download<T>(handle, d_b, static_cast<std::size_t>(n) * nrhs);

  // 1. Backward error of the device solution against the original matrix.
  const auto r = residual<T>(n, nrhs, a_orig, x_dev, b);
  const R norm_a = fro<T>(a_orig);
  const R norm_x = fro<T>(x_dev);
  const R norm_b = fro<T>(b);
  const R bound = R{128} * eps<R>() * static_cast<R>(n) * (norm_a * norm_x + norm_b);
  EXPECT_LE(fro<T>(r), bound) << "backward error n=" << n << " nrhs=" << nrhs << " uplo=" << cuplo
                              << " seed=" << seed;

  // 2. Loose agreement with the reference forward solution (same factor/pivots).
  std::vector<Host<T>> diff(x_dev.size());
  for (std::size_t i = 0; i < x_dev.size(); ++i) {
    diff[i] = x_dev[i] - x_ref[i];
  }
  const R ref_bound = R{1024} * eps<R>() * static_cast<R>(n) * (norm_x + R{1});
  EXPECT_LE(fro<T>(diff), ref_bound) << "vs LAPACKE n=" << n << " nrhs=" << nrhs;

  // 3. The ?syconv convert/revert round-trip leaves the factor unchanged.
  const auto a_back = download<T>(handle, d_a, static_cast<std::size_t>(lda) * n);
  std::vector<Host<T>> fdiff(a_back.size());
  for (std::size_t i = 0; i < a_back.size(); ++i) {
    fdiff[i] = a_back[i] - factor[i];
  }
  const R fac_bound = R{8} * eps<R>() * (fro<T>(factor) + R{1});
  EXPECT_LE(fro<T>(fdiff), fac_bound)
      << "factor not restored n=" << n << " uplo=" << cuplo << " seed=" << seed;
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

TEST(Sytrs2OracleTests, Float) {
  run_all<float>(100);
}
TEST(Sytrs2OracleTests, Double) {
  run_all<double>(200);
}
TEST(Sytrs2OracleTests, ComplexFloat) {
  run_all<wwr::wwrFloatComplex>(300);
}
TEST(Sytrs2OracleTests, ComplexDouble) {
  run_all<wwr::wwrDoubleComplex>(400);
}

} // namespace
} // namespace calaman
