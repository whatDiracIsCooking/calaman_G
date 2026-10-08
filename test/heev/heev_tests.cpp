// Oracle test for calaman.heev -- eigenvalues and optional eigenvectors of a
// complex Hermitian matrix (?heev). The oracle is LAPACKE_?heev in the SAME
// precision on the host over the identical input.
//
// The vendor ?hetrd need not reproduce the reference's tridiagonal step by
// step, so per case: INFO matches; the eigenvalues match the oracle's to the
// shared factorization tolerance (and ascend); and, for Jobz::V, each column
// matches the oracle's up to a unit-modulus phase where the eigenvalue gap
// makes that well-posed, ||Z^H Z - I|| is small, and ||A Z - Z diag(w)|| is
// small (formed on the host in complex long double). Both Uplo, both Jobz,
// lda == n and lda > n (whose padding rows must survive), n = 0, 1, 2 up to
// 300, two seeds; the opposite triangle carries garbage so a read of it would
// show. Inputs: random, clustered spectra, zero, and huge / tiny norms that
// take the lanhe + lascl scaling path. REQUIRES_GPU; built only with the oracle.

#include <gtest/gtest.h>

#include <lapacke.h> // lapack_complex_float, lapack_complex_double

#include <cstddef>

import std;

import wwr.runtime_api;
import wwr.solver;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.heev;
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
using cld = std::complex<ld>;

// Per-element-type traits: the real component type, the builder, the widening
// to complex long double (wwrC* accessors -- .x/.y is not portable to
// hipComplex), and the oracle (the wwr complex types are layout-compatible
// with LAPACKE's).
template<typename T>
struct elem;

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static wwr::wwrFloatComplex make(cld v) {
    return wwr::make_wwrFloatComplex(static_cast<float>(v.real()), static_cast<float>(v.imag()));
  }
  static cld wide(wwr::wwrFloatComplex v) { return {wwr::wwrCrealf(v), wwr::wwrCimagf(v)}; }
  static int ref(char jobz, char uplo, int n, wwr::wwrFloatComplex *a, int lda, float *w) {
    return LAPACKE_cheev(LAPACK_COL_MAJOR, jobz, uplo, n,
                         reinterpret_cast<lapack_complex_float *>(a), lda, w);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static wwr::wwrDoubleComplex make(cld v) {
    return wwr::make_wwrDoubleComplex(static_cast<double>(v.real()), static_cast<double>(v.imag()));
  }
  static cld wide(wwr::wwrDoubleComplex v) { return {wwr::wwrCreal(v), wwr::wwrCimag(v)}; }
  static int ref(char jobz, char uplo, int n, wwr::wwrDoubleComplex *a, int lda, double *w) {
    return LAPACKE_zheev(LAPACK_COL_MAJOR, jobz, uplo, n,
                         reinterpret_cast<lapack_complex_double *>(a), lda, w);
  }
};

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

enum class Kind { random, clustered, zero, huge, tiny };

const char *kind_name(Kind k) {
  switch (k) {
  case Kind::random:
    return "random";
  case Kind::clustered:
    return "clustered";
  case Kind::zero:
    return "zero";
  case Kind::huge:
    return "huge";
  case Kind::tiny:
    return "tiny";
  }
  return "?";
}

/// The Hermitian n-by-n matrix (complex long double, dense, ld = n, real
/// diagonal) for the given kind and seed.
template<typename R>
std::vector<cld> make_hermitian(Kind kind, int n, int seed) {
  std::mt19937 gen(
      static_cast<std::uint32_t>(131 * n + 17 * static_cast<int>(kind) + 7919 * seed + 3));
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const auto un = static_cast<std::size_t>(n);
  std::vector<cld> A(un * un, 0.0L);
  if (kind == Kind::zero) {
    return A;
  }
  if (kind == Kind::clustered) {
    // Q diag(lambda) Q^H, lambda in three tight clusters; Q from n complex
    // reflectors.
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<cld> Q(un * un, 0.0L);
    for (std::size_t i = 0; i < un; ++i) {
      Q[i + i * un] = 1.0L;
    }
    for (int r = 0; r < n; ++r) {
      std::vector<cld> v(un);
      ld nrm = 0.0L;
      for (auto &x : v) {
        const ld re = g(gen);
        const ld im = g(gen);
        x = {re, im};
        nrm += std::norm(x);
      }
      for (std::size_t i = 0; i < un; ++i) {
        cld dot = 0.0L;
        for (std::size_t k = 0; k < un; ++k) {
          dot += Q[i + k * un] * v[k];
        }
        for (std::size_t k = 0; k < un; ++k) {
          Q[i + k * un] -= 2.0L * dot * std::conj(v[k]) / nrm;
        }
      }
    }
    std::vector<ld> lambda(un);
    for (std::size_t k = 0; k < un; ++k) {
      lambda[k] = static_cast<ld>(static_cast<int>(k % 3) - 1) + 1e-9L * u(gen);
    }
    for (std::size_t j = 0; j < un; ++j) {
      for (std::size_t k = 0; k < un; ++k) {
        const cld qjl = std::conj(Q[j + k * un]) * lambda[k];
        for (std::size_t i = 0; i < un; ++i) {
          A[i + j * un] += Q[i + k * un] * qjl;
        }
      }
    }
    for (std::size_t i = 0; i < un; ++i) {
      A[i + i * un] = A[i + i * un].real();
    }
    return A;
  }
  double scale = 1.0;
  if (kind == Kind::huge) {
    scale = std::is_same_v<R, float> ? 1e20 : 1e200;
  } else if (kind == Kind::tiny) {
    scale = std::is_same_v<R, float> ? 1e-20 : 1e-200;
  }
  for (std::size_t j = 0; j < un; ++j) {
    A[j + j * un] = static_cast<ld>(u(gen) * scale);
    for (std::size_t i = j + 1; i < un; ++i) {
      const cld v{static_cast<ld>(u(gen) * scale), static_cast<ld>(u(gen) * scale)};
      A[i + j * un] = v;
      A[j + i * un] = std::conj(v);
    }
  }
  return A;
}

template<typename T>
void run_case(Jobz jobz, Uplo uplo, Kind kind, int n, int pad, int seed) {
  using R = typename elem<T>::R;
  const bool wantz = jobz == Jobz::V;
  const char cj = wantz ? 'V' : 'N';
  const char cu = uplo == Uplo::U ? 'U' : 'L';
  SCOPED_TRACE(std::string(1, cj) + std::string(1, cu) + " " + kind_name(kind) + " n=" +
               std::to_string(n) + " pad=" + std::to_string(pad) + " seed=" + std::to_string(seed));
  const int lda = std::max(1, n) + pad;
  const auto un = static_cast<std::size_t>(n);
  const auto ula = static_cast<std::size_t>(lda);

  // A rounded to T; the uplo triangle is the matrix, the other triangle and the
  // lda padding rows carry garbage that heev must neither read nor depend on.
  const std::vector<cld> Aex = make_hermitian<R>(kind, n, seed);
  std::vector<cld> A(un * un); // the T-rounded Hermitian matrix, ld = n
  std::vector<T> a0(ula * std::max<std::size_t>(un, 1), elem<T>::make(cld{-7.0L, 3.0L}));
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      const bool in_tri = uplo == Uplo::U ? i <= j : i >= j;
      const T stored = elem<T>::make(in_tri ? Aex[i + j * un] : std::conj(Aex[j + i * un]));
      A[i + j * un] = elem<T>::wide(stored);
      a0[i + j * ula] = in_tri ? stored : elem<T>::make(cld{1000.0L + i, -500.0L});
    }
  }
  // Re-symmetrise the rounded A from the referenced triangle.
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      const bool in_tri = uplo == Uplo::U ? i <= j : i >= j;
      if (!in_tri) {
        A[i + j * un] = std::conj(A[j + i * un]);
      }
    }
  }

  // Oracle.
  std::vector<T> r_a = a0;
  std::vector<R> r_w(std::max<std::size_t>(un, 1), R{0});
  const int r_info = elem<T>::ref(cj, cu, n, r_a.data(), lda, r_w.data());

  // Device.
  const auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  auto d_a = to_device(a0);
  auto d_w = to_device(std::vector<R>(std::max<std::size_t>(un, 1), R{0}));
  auto d_info = to_device(std::vector<int>{-123});
  const std::size_t bytes = heev_bufferSize<T>(solver, jobz, uplo, n, lda);
  DeviceBuffer<std::byte> d_work(bytes == 0 ? 1 : bytes, handle);
  const auto status = heev<T>(solver, jobz, uplo, n, d_a.data(), lda, d_w.data(), d_work.data(),
                              bytes, d_info.data());
  static_cast<void>(wwr::wwrStreamSynchronize(handle->stream().get()));
  wwr::wwrsolverDnDestroy(solver);
  ASSERT_EQ(status, wwr::wwrSuccess);
  const int g_info = to_host(d_info, 1)[0];
  ASSERT_EQ(g_info, r_info);
  ASSERT_EQ(g_info, 0);
  const auto g_w = to_host(d_w, un);
  const auto g_at = to_host(d_a, a0.size());

  ld anorm2 = 0.0L;
  for (const cld v : A) {
    anorm2 += std::norm(v);
  }
  const R anorm = static_cast<R>(std::sqrt(anorm2));
  const R tol = test::factorization_tol<R>(anorm, un, un) + std::numeric_limits<R>::min();
  for (std::size_t i = 0; i < un; ++i) {
    EXPECT_NEAR(g_w[i], r_w[i], tol) << "w(" << i << ")";
    if (i > 0) {
      EXPECT_LE(g_w[i - 1], g_w[i]) << "ascending at " << i;
    }
  }

  std::vector<cld> g_a(g_at.size()), o_a(r_a.size());
  for (std::size_t i = 0; i < g_at.size(); ++i) {
    g_a[i] = elem<T>::wide(g_at[i]);
    o_a[i] = elem<T>::wide(r_a[i]);
  }

  // The lda padding rows are never written.
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = un; i < ula; ++i) {
      ASSERT_EQ(g_a[i + j * ula], elem<T>::wide(a0[i + j * ula]))
          << "padding A(" << i << "," << j << ")";
    }
  }
  if (!wantz) {
    return;
  }

  // Columns up to a unit-modulus phase, where the eigenvalue gap makes them
  // well-determined: g_j ~ (o_j^H g_j / |.|) o_j.
  const R unit_tol = test::factorization_tol<R>(R{1}, un, un);
  for (std::size_t j = 0; j < un; ++j) {
    R gap = std::numeric_limits<R>::max();
    if (j > 0) {
      gap = std::min(gap, r_w[j] - r_w[j - 1]);
    }
    if (j + 1 < un) {
      gap = std::min(gap, r_w[j + 1] - r_w[j]);
    }
    if (anorm == R{0} || gap < R{1e-2} * anorm) {
      continue;
    }
    cld dot = 0.0L;
    for (std::size_t i = 0; i < un; ++i) {
      dot += std::conj(o_a[i + j * ula]) * g_a[i + j * ula];
    }
    const cld phase = std::abs(dot) == 0.0L ? cld{1.0L} : dot / std::abs(dot);
    const R vtol = unit_tol * std::max(R{1}, anorm / gap);
    for (std::size_t i = 0; i < un; ++i) {
      EXPECT_LE(static_cast<R>(std::abs(g_a[i + j * ula] - phase * o_a[i + j * ula])), vtol)
          << "Z(" << i << "," << j << ")";
    }
  }

  // ||Z^H Z - I||_F.
  ld orth2 = 0.0L;
  for (std::size_t p = 0; p < un; ++p) {
    for (std::size_t q = 0; q < un; ++q) {
      cld s = p == q ? -1.0L : 0.0L;
      for (std::size_t i = 0; i < un; ++i) {
        s += std::conj(g_a[i + p * ula]) * g_a[i + q * ula];
      }
      orth2 += std::norm(s);
    }
  }
  EXPECT_LE(static_cast<R>(std::sqrt(orth2)), test::factorization_tol<R>(R(n), un, un))
      << "orthogonality";

  // ||A Z - Z diag(w)||_F, in complex long double from the T-precision A.
  ld res2 = 0.0L;
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      cld s = -g_a[i + j * ula] * static_cast<ld>(g_w[j]);
      for (std::size_t k = 0; k < un; ++k) {
        s += A[i + k * un] * g_a[k + j * ula];
      }
      res2 += std::norm(s);
    }
  }
  const R rtol = kTolFactor<R> * eps<R>() * anorm * static_cast<R>(n) *
                     std::sqrt(static_cast<R>(std::max(n, 1))) +
                 std::numeric_limits<R>::min();
  EXPECT_LE(static_cast<R>(std::sqrt(res2)), rtol) << "residual";
}

constexpr std::array kJobs = {Jobz::N, Jobz::V};
constexpr std::array kUplos = {Uplo::U, Uplo::L};
constexpr std::array kKinds = {Kind::random, Kind::clustered, Kind::zero, Kind::huge, Kind::tiny};
constexpr std::array kSizes = {0, 1, 2, 3, 7, 16, 33, 65, 128, 300};
constexpr std::array kPads = {0, 3};

template<typename T>
void run_all() {
  for (const Jobz jobz : kJobs) {
    for (const Uplo uplo : kUplos) {
      for (const Kind kind : kKinds) {
        for (const int n : kSizes) {
          for (const int pad : kPads) {
            // Two seeds below the largest order; one at it keeps the host
            // long-double checks affordable.
            const int seeds = n >= 300 ? 1 : 2;
            for (int seed = 0; seed < seeds; ++seed) {
              run_case<T>(jobz, uplo, kind, n, pad, seed);
            }
          }
        }
      }
    }
  }
}

TEST(HeevOracleTests, ComplexFloat) {
  run_all<wwr::wwrFloatComplex>();
}

TEST(HeevOracleTests, ComplexDouble) {
  run_all<wwr::wwrDoubleComplex>();
}

TEST(HeevOracleTests, RejectsBadArguments) {
  using Z = wwr::wwrDoubleComplex;
  const auto handle = shared_device();
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  auto d_a = to_device(std::vector<Z>(16, wwr::make_wwrDoubleComplex(1.0, 0.0)));
  auto d_w = to_device(std::vector<double>(4, 0.0));
  auto d_info = to_device(std::vector<int>{0});
  const std::size_t bytes = heev_bufferSize<Z>(solver, Jobz::V, Uplo::L, 4, 4);
  DeviceBuffer<std::byte> d_work(bytes, handle);
  EXPECT_NE(heev<Z>(solver, Jobz::V, Uplo::L, -1, d_a.data(), 4, d_w.data(), d_work.data(), bytes,
                    d_info.data()),
            wwr::wwrSuccess);
  EXPECT_NE(heev<Z>(solver, Jobz::V, Uplo::L, 4, d_a.data(), 3, d_w.data(), d_work.data(), bytes,
                    d_info.data()),
            wwr::wwrSuccess);
  EXPECT_NE(heev<Z>(solver, Jobz::V, Uplo::L, 4, d_a.data(), 4, d_w.data(), d_work.data(),
                    bytes - 1, d_info.data()),
            wwr::wwrSuccess);
  static_cast<void>(wwr::wwrStreamSynchronize(handle->stream().get()));
  wwr::wwrsolverDnDestroy(solver);
}

} // namespace
} // namespace calaman
