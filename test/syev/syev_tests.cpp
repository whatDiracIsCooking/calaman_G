// Oracle test for calaman.syev -- eigenvalues and optional eigenvectors of a
// real symmetric matrix (?syev). The oracle is LAPACKE_?syev in the SAME
// precision on the host over the identical input.
//
// The vendor ?sytrd need not reproduce the reference's tridiagonal step by
// step, so per case: INFO matches; the eigenvalues match the oracle's to the
// shared factorization tolerance (and ascend); and, for Jobz::V, each column
// matches the oracle's up to sign where the eigenvalue gap makes that
// well-posed, ||Z^T Z - I|| is small, and ||A Z - Z diag(w)|| is small (formed
// on the host in long double). Both Uplo, both Jobz, lda > n, n = 0, 1, 2 up to
// 300; the opposite triangle carries garbage so a read of it would show.
// Inputs: random, clustered spectra, zero, and huge / tiny norms that take the
// lansy + lascl scaling path. REQUIRES_GPU; built only with the oracle.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>

import std;

import wwr.runtime_api;
import wwr.solver;
import wwr.extension.memory_buffer;
import calaman.syev;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;
using test::kTolFactor;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

using ld = long double;

template<typename T>
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
  const std::size_t n = host.size();
  HostBuffer<T> staging(n == 0 ? 1 : n);
  for (std::size_t i = 0; i < n; ++i) {
    staging.data()[i] = host[i];
  }
  DeviceBuffer<T> device(n == 0 ? 1 : n, handle);
  static_cast<void>(wwr::extension::copy(device, staging, handle->stream().get()));
  static_cast<void>(wwr::wwrStreamSynchronize(handle->stream().get()));
  return device;
}

template<typename T>
std::vector<T> to_host(const DeviceBuffer<T> &device, std::size_t n) {
  const auto handle = shared_device();
  HostBuffer<T> host(n == 0 ? 1 : n);
  static_cast<void>(wwr::extension::copy(host, device, handle->stream().get()));
  static_cast<void>(wwr::wwrStreamSynchronize(handle->stream().get()));
  return std::vector<T>(host.data(), host.data() + n);
}

int ref_syev(char jobz, char uplo, int n, float *a, int lda, float *w) {
  return LAPACKE_ssyev(LAPACK_COL_MAJOR, jobz, uplo, n, a, lda, w);
}
int ref_syev(char jobz, char uplo, int n, double *a, int lda, double *w) {
  return LAPACKE_dsyev(LAPACK_COL_MAJOR, jobz, uplo, n, a, lda, w);
}

enum class Kind { random, clustered, zero, huge, tiny };

const char *kind_name(Kind k) {
  switch (k) {
  case Kind::random: return "random";
  case Kind::clustered: return "clustered";
  case Kind::zero: return "zero";
  case Kind::huge: return "huge";
  case Kind::tiny: return "tiny";
  }
  return "?";
}

/// The symmetric n-by-n matrix (long double, dense, ld = n) for the given kind.
template<typename T>
std::vector<ld> make_symmetric(Kind kind, int n) {
  std::mt19937 gen(static_cast<std::uint32_t>(131 * n + 17 * static_cast<int>(kind) + 3));
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const auto un = static_cast<std::size_t>(n);
  std::vector<ld> A(un * un, 0.0L);
  if (kind == Kind::zero) {
    return A;
  }
  if (kind == Kind::clustered) {
    // Q diag(lambda) Q^T, lambda in three tight clusters; Q from n reflectors.
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<ld> Q(un * un, 0.0L);
    for (std::size_t i = 0; i < un; ++i) {
      Q[i + i * un] = 1.0L;
    }
    for (int r = 0; r < n; ++r) {
      std::vector<ld> v(un);
      ld nrm = 0.0L;
      for (auto &x : v) {
        x = g(gen);
        nrm += x * x;
      }
      for (std::size_t i = 0; i < un; ++i) {
        ld dot = 0.0L;
        for (std::size_t k = 0; k < un; ++k) {
          dot += Q[i + k * un] * v[k];
        }
        for (std::size_t k = 0; k < un; ++k) {
          Q[i + k * un] -= 2.0L * dot * v[k] / nrm;
        }
      }
    }
    std::vector<ld> lambda(un);
    for (std::size_t k = 0; k < un; ++k) {
      lambda[k] = static_cast<ld>(static_cast<int>(k % 3) - 1) + 1e-9L * u(gen);
    }
    for (std::size_t j = 0; j < un; ++j) {
      for (std::size_t k = 0; k < un; ++k) {
        const ld qjl = Q[j + k * un] * lambda[k];
        for (std::size_t i = 0; i < un; ++i) {
          A[i + j * un] += Q[i + k * un] * qjl;
        }
      }
    }
    return A;
  }
  double scale = 1.0;
  if (kind == Kind::huge) {
    scale = std::is_same_v<T, float> ? 1e20 : 1e200;
  } else if (kind == Kind::tiny) {
    scale = std::is_same_v<T, float> ? 1e-20 : 1e-200;
  }
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = j; i < un; ++i) {
      const ld v = static_cast<ld>(u(gen) * scale);
      A[i + j * un] = v;
      A[j + i * un] = v;
    }
  }
  return A;
}

template<typename T>
void run_case(Jobz jobz, Uplo uplo, Kind kind, int n) {
  const bool wantz = jobz == Jobz::V;
  const char cj = wantz ? 'V' : 'N';
  const char cu = uplo == Uplo::U ? 'U' : 'L';
  SCOPED_TRACE(std::string(1, cj) + std::string(1, cu) + " " + kind_name(kind) +
               " n=" + std::to_string(n));
  const int lda = std::max(1, n) + 3;
  const auto un = static_cast<std::size_t>(n);
  const auto ula = static_cast<std::size_t>(lda);

  // A rounded to T; the uplo triangle is the matrix, the other triangle and the
  // lda padding rows carry garbage that syev must neither read nor depend on.
  const std::vector<ld> Aex = make_symmetric<T>(kind, n);
  std::vector<T> A(un * un); // the T-rounded symmetric matrix, ld = n
  std::vector<T> a0(ula * std::max<std::size_t>(un, 1), T{-7});
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      const bool in_tri = uplo == Uplo::U ? i <= j : i >= j;
      const std::size_t s = in_tri ? i + j * un : j + i * un;
      A[i + j * un] = static_cast<T>(Aex[s]);
      a0[i + j * ula] = in_tri ? A[i + j * un] : T{1000} + static_cast<T>(i);
    }
  }

  // Oracle.
  std::vector<T> r_a = a0, r_w(std::max<std::size_t>(un, 1), T{0});
  const int r_info = ref_syev(cj, cu, n, r_a.data(), lda, r_w.data());

  // Device.
  const auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  auto d_a = to_device(a0);
  auto d_w = to_device(std::vector<T>(std::max<std::size_t>(un, 1), T{0}));
  auto d_info = to_device(std::vector<int>{-123});
  const std::size_t bytes = syev_bufferSize<T>(solver, jobz, uplo, n, lda);
  DeviceBuffer<std::byte> d_work(bytes == 0 ? 1 : bytes, handle);
  const auto status =
      syev<T>(solver, jobz, uplo, n, d_a.data(), lda, d_w.data(), d_work.data(), bytes,
              d_info.data());
  static_cast<void>(wwr::wwrStreamSynchronize(handle->stream().get()));
  wwr::wwrsolverDnDestroy(solver);
  ASSERT_EQ(status, wwr::wwrSuccess);
  const int g_info = to_host(d_info, 1)[0];
  ASSERT_EQ(g_info, r_info);
  ASSERT_EQ(g_info, 0);
  const auto g_w = to_host(d_w, un);
  const auto g_a = to_host(d_a, a0.size());

  ld anorm2 = 0.0L;
  for (const T v : A) {
    anorm2 += static_cast<ld>(v) * v;
  }
  const T anorm = static_cast<T>(std::sqrt(anorm2));
  const T tol = test::factorization_tol<T>(anorm, un, un) + std::numeric_limits<T>::min();
  for (std::size_t i = 0; i < un; ++i) {
    EXPECT_NEAR(g_w[i], r_w[i], tol) << "w(" << i << ")";
    if (i > 0) {
      EXPECT_LE(g_w[i - 1], g_w[i]) << "ascending at " << i;
    }
  }

  // The lda padding rows are never written.
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = un; i < ula; ++i) {
      ASSERT_EQ(g_a[i + j * ula], a0[i + j * ula]) << "padding A(" << i << "," << j << ")";
    }
  }
  if (!wantz) {
    return;
  }

  // Columns up to sign, where the eigenvalue gap makes them well-determined.
  const T unit_tol = test::factorization_tol<T>(T{1}, un, un);
  for (std::size_t j = 0; j < un; ++j) {
    T gap = std::numeric_limits<T>::max();
    if (j > 0) {
      gap = std::min(gap, r_w[j] - r_w[j - 1]);
    }
    if (j + 1 < un) {
      gap = std::min(gap, r_w[j + 1] - r_w[j]);
    }
    if (anorm == T{0} || gap < T{1e-2} * anorm) {
      continue;
    }
    ld dot = 0.0L;
    for (std::size_t i = 0; i < un; ++i) {
      dot += static_cast<ld>(g_a[i + j * ula]) * r_a[i + j * ula];
    }
    const T sign = dot < 0.0L ? T{-1} : T{1};
    const T vtol = unit_tol * std::max(T{1}, anorm / gap);
    for (std::size_t i = 0; i < un; ++i) {
      EXPECT_NEAR(g_a[i + j * ula], sign * r_a[i + j * ula], vtol) << "Z(" << i << "," << j << ")";
    }
  }

  // ||Z^T Z - I||_F.
  ld orth2 = 0.0L;
  for (std::size_t p = 0; p < un; ++p) {
    for (std::size_t q = 0; q < un; ++q) {
      ld s = p == q ? -1.0L : 0.0L;
      for (std::size_t i = 0; i < un; ++i) {
        s += static_cast<ld>(g_a[i + p * ula]) * g_a[i + q * ula];
      }
      orth2 += s * s;
    }
  }
  EXPECT_LE(static_cast<T>(std::sqrt(orth2)), test::factorization_tol<T>(T(n), un, un))
      << "orthogonality";

  // ||A Z - Z diag(w)||_F, in long double from the T-precision A.
  ld res2 = 0.0L;
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      ld s = -static_cast<ld>(g_a[i + j * ula]) * g_w[j];
      for (std::size_t k = 0; k < un; ++k) {
        s += static_cast<ld>(A[i + k * un]) * g_a[k + j * ula];
      }
      res2 += s * s;
    }
  }
  const T rtol = kTolFactor<T> * eps<T>() * anorm * static_cast<T>(n) *
                     std::sqrt(static_cast<T>(std::max(n, 1))) +
                 std::numeric_limits<T>::min();
  EXPECT_LE(static_cast<T>(std::sqrt(res2)), rtol) << "residual";
}

constexpr std::array kJobs = {Jobz::N, Jobz::V};
constexpr std::array kUplos = {Uplo::U, Uplo::L};
constexpr std::array kKinds = {Kind::random, Kind::clustered, Kind::zero, Kind::huge, Kind::tiny};
constexpr std::array kSizes = {0, 1, 2, 3, 7, 16, 65, 300};

template<typename T>
void run_all() {
  for (const Jobz jobz : kJobs) {
    for (const Uplo uplo : kUplos) {
      for (const Kind kind : kKinds) {
        for (const int n : kSizes) {
          run_case<T>(jobz, uplo, kind, n);
        }
      }
    }
  }
}

TEST(SyevOracleTests, Float) {
  run_all<float>();
}

TEST(SyevOracleTests, Double) {
  run_all<double>();
}

TEST(SyevOracleTests, RejectsBadArguments) {
  const auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  auto d_a = to_device(std::vector<double>(16, 1.0));
  auto d_w = to_device(std::vector<double>(4, 0.0));
  auto d_info = to_device(std::vector<int>{0});
  const std::size_t bytes = syev_bufferSize<double>(solver, Jobz::V, Uplo::L, 4, 4);
  DeviceBuffer<std::byte> d_work(bytes, handle);
  EXPECT_NE(syev<double>(solver, Jobz::V, Uplo::L, -1, d_a.data(), 4, d_w.data(), d_work.data(),
                         bytes, d_info.data()),
            wwr::wwrSuccess);
  EXPECT_NE(syev<double>(solver, Jobz::V, Uplo::L, 4, d_a.data(), 3, d_w.data(), d_work.data(),
                         bytes, d_info.data()),
            wwr::wwrSuccess);
  EXPECT_NE(syev<double>(solver, Jobz::V, Uplo::L, 4, d_a.data(), 4, d_w.data(), d_work.data(),
                         bytes - 1, d_info.data()),
            wwr::wwrSuccess);
  static_cast<void>(wwr::wwrStreamSynchronize(handle->stream().get()));
  wwr::wwrsolverDnDestroy(solver);
}

} // namespace
} // namespace calaman
