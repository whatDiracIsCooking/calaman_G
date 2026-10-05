// Oracle test for calaman.cg_unitary -- Riemannian conjugate gradient on U(n),
// exercised on the Brockett criterion J(W) = trace{W^H R W N} with
// N = diag(1..n) (paper §5.1). MAXIMIZING J drives W to the eigenvectors of the
// Hermitian R and W^H R W to a diagonal of R's eigenvalues in ASCENDING order
// (smallest paired with weight 1, largest with weight n) -- so a Hermitian
// eigendecomposition is an independent oracle: the computed W must diagonalize R,
// its W^H R W diagonal must match LAPACKE's eigenvalues, and for a true maximum
// that diagonal must come out ascending.
//
// Host arithmetic runs in std::complex (or the plain real), converting at the
// device boundary via wwr.complex's host accessors -- never .x/.y, which
// hipComplex lacks -- so the oracle shares none of the solver's device code. All
// four element types. Every case stages R/N/W/work on the device and runs the
// kernels + BLAS + expm, so the suites are REQUIRES_GPU (`ctest -LE gpu` excludes
// them). Guards on calaman::lapack_reference, since the eigenvalue oracle is the
// reference LAPACKE Hermitian solver.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.cg_unitary;
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
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// ── element traits: device type T <-> host compute type ──────────────────────

template<typename T>
struct traits;

template<>
struct traits<float> {
  using compute = float;
  using real = float;
  static float make(double re, double) { return static_cast<float>(re); }
  static compute to(float x) { return x; }
  static float from(compute c) { return c; }
};
template<>
struct traits<double> {
  using compute = double;
  using real = double;
  static double make(double re, double) { return re; }
  static compute to(double x) { return x; }
  static double from(compute c) { return c; }
};
template<>
struct traits<wwr::wwrFloatComplex> {
  using compute = std::complex<float>;
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static compute to(wwr::wwrFloatComplex x) { return {wwr::wwrCrealf(x), wwr::wwrCimagf(x)}; }
  static wwr::wwrFloatComplex from(compute c) {
    return wwr::make_wwrFloatComplex(c.real(), c.imag());
  }
};
template<>
struct traits<wwr::wwrDoubleComplex> {
  using compute = std::complex<double>;
  using real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static compute to(wwr::wwrDoubleComplex x) { return {wwr::wwrCreal(x), wwr::wwrCimag(x)}; }
  static wwr::wwrDoubleComplex from(compute c) {
    return wwr::make_wwrDoubleComplex(c.real(), c.imag());
  }
};

template<typename T>
using compute_t = typename traits<T>::compute;
template<typename T>
using real_t = typename traits<T>::real;

template<typename C>
double re_of(const C &c) {
  if constexpr (std::is_same_v<C, float> || std::is_same_v<C, double>) {
    return static_cast<double>(c);
  } else {
    return static_cast<double>(c.real());
  }
}
template<typename C>
C conj_of(const C &c) {
  if constexpr (std::is_same_v<C, float> || std::is_same_v<C, double>) {
    return c;
  } else {
    return std::conj(c);
  }
}
template<typename C>
double abs_d(const C &c) {
  if constexpr (std::is_same_v<C, float> || std::is_same_v<C, double>) {
    return std::abs(static_cast<double>(c));
  } else {
    return std::abs(std::complex<double>(c.real(), c.imag()));
  }
}

// ── device staging ───────────────────────────────────────────────────────────

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

// ── reference Hermitian/symmetric eigenvalues (ascending), via LAPACKE ────────

std::vector<float> ref_eigenvalues(int n, std::vector<float> a) {
  std::vector<float> w(n);
  const lapack_int rc = LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a.data(), n, w.data());
  EXPECT_EQ(rc, 0);
  return w;
}
std::vector<double> ref_eigenvalues(int n, std::vector<double> a) {
  std::vector<double> w(n);
  const lapack_int rc = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a.data(), n, w.data());
  EXPECT_EQ(rc, 0);
  return w;
}
std::vector<float> ref_eigenvalues(int n, std::vector<std::complex<float>> a) {
  std::vector<float> w(n);
  const lapack_int rc = LAPACKE_cheevd(LAPACK_COL_MAJOR, 'N', 'L', n,
                                       reinterpret_cast<lapack_complex_float *>(a.data()), n,
                                       w.data());
  EXPECT_EQ(rc, 0);
  return w;
}
std::vector<double> ref_eigenvalues(int n, std::vector<std::complex<double>> a) {
  std::vector<double> w(n);
  const lapack_int rc = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'N', 'L', n,
                                       reinterpret_cast<lapack_complex_double *>(a.data()), n,
                                       w.data());
  EXPECT_EQ(rc, 0);
  return w;
}

// ── a random Hermitian (symmetric) R in compute_t, column-major ──────────────

template<typename C>
std::vector<C> random_hermitian(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<C> a(static_cast<std::size_t>(n) * n, C{});
  const auto at = [&](int i, int j) -> C & { return a[static_cast<std::size_t>(j) * n + i]; };
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      if (i == j) {
        at(i, i) = static_cast<C>(dist(rng)); // real diagonal
      } else if constexpr (std::is_same_v<C, float> || std::is_same_v<C, double>) {
        const C v = static_cast<C>(dist(rng));
        at(i, j) = v;
        at(j, i) = v;
      } else {
        const C v(dist(rng), dist(rng));
        at(i, j) = v;
        at(j, i) = std::conj(v);
      }
    }
  }
  return a;
}

// ── the solve + oracle check ─────────────────────────────────────────────────

template<typename T>
void expect_brockett_maximizes(int n, unsigned seed, LineSearchMethod method) {
  using C = compute_t<T>;
  using R = real_t<T>;

  const std::vector<C> R_host = random_hermitian<C>(n, seed);
  const std::vector<R> eig = ref_eigenvalues(n, R_host); // ascending

  // Device inputs: R, N = diag(1..n), W_0 = I, all element type T.
  std::vector<T> R_dev(static_cast<std::size_t>(n) * n);
  for (std::size_t i = 0; i < R_host.size(); ++i) {
    R_dev[i] = traits<T>::from(R_host[i]);
  }
  std::vector<T> N_dev(n);
  for (int i = 0; i < n; ++i) {
    N_dev[i] = traits<T>::make(static_cast<double>(i + 1), 0.0);
  }
  std::vector<T> W_host(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  for (int i = 0; i < n; ++i) {
    W_host[static_cast<std::size_t>(i) * n + i] = traits<T>::make(1.0, 0.0);
  }

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  auto d_R = to_device(handle, R_dev);
  auto d_N = to_device(handle, N_dev);
  auto d_W = to_device(handle, W_host);

  brockett_cost<T> cost{d_R.data(), n, d_N.data(), n};

  std::size_t lwork = 0;
  ASSERT_TRUE(cg_unitary_bufferSize<T>(solver, n, cost.bufferSize(n), &lwork).ok());
  DeviceBuffer<T> d_work(lwork / sizeof(T) + 1, handle);

  CgOptions<R> opts{};
  opts.method = method;
  CgInfo<R> info{};
  const Status st = cg_unitary<T>(blas, solver, handle->stream().get(), n, d_W.data(), cost,
                                  CgDirection::Maximize, d_work.data(), lwork, opts, &info);
  ASSERT_TRUE(st.ok()) << "cg_unitary returned " << st.name();
  EXPECT_NE(info.reason, CgStopReason::NumericalFailure);

  const std::vector<T> W_out = from_device(handle, d_W, static_cast<std::size_t>(n) * n);
  wwr::wwrblasDestroy(blas);
  wwr::wwrsolverDnDestroy(solver);

  // M = W^H R W, in compute_t.
  std::vector<C> W(static_cast<std::size_t>(n) * n);
  for (std::size_t i = 0; i < W.size(); ++i) {
    W[i] = traits<T>::to(W_out[i]);
  }
  const auto R_at = [&](int i, int j) { return R_host[static_cast<std::size_t>(j) * n + i]; };
  const auto W_at = [&](int i, int j) { return W[static_cast<std::size_t>(j) * n + i]; };

  // RW[i,j] = sum_k R[i,k] W[k,j]
  std::vector<C> RW(static_cast<std::size_t>(n) * n, C{});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      C acc{};
      for (int k = 0; k < n; ++k) {
        acc += R_at(i, k) * W_at(k, j);
      }
      RW[static_cast<std::size_t>(j) * n + i] = acc;
    }
  }
  // M[a,b] = sum_i conj(W[i,a]) RW[i,b]
  std::vector<C> M(static_cast<std::size_t>(n) * n, C{});
  for (int b = 0; b < n; ++b) {
    for (int a = 0; a < n; ++a) {
      C acc{};
      for (int i = 0; i < n; ++i) {
        acc += conj_of(W_at(i, a)) * RW[static_cast<std::size_t>(b) * n + i];
      }
      M[static_cast<std::size_t>(b) * n + a] = acc;
    }
  }

  double scale = 1.0;
  for (const R e : eig) {
    scale = std::max(scale, std::abs(static_cast<double>(e)));
  }
  const double tol = (std::is_same_v<R, float> ? 2e-2 : 1e-5) * scale;

  // W is unitary: W^H W = I (a necessary condition the exponential maintains).
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      C acc{};
      for (int i = 0; i < n; ++i) {
        acc += conj_of(W_at(i, a)) * W_at(i, b);
      }
      const double expected = (a == b) ? 1.0 : 0.0;
      EXPECT_NEAR(re_of(acc), expected, (std::is_same_v<R, float> ? 1e-3 : 1e-9))
          << "W not unitary at (" << a << "," << b << ") n=" << n;
    }
  }

  // Off-diagonals of M vanish: W columns are eigenvectors of R.
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      if (a != b) {
        EXPECT_NEAR(abs_d(M[static_cast<std::size_t>(b) * n + a]), 0.0, tol)
            << "M off-diagonal (" << a << "," << b << ") n=" << n;
      }
    }
  }

  // Diagonal of M equals the eigenvalues, and comes out ASCENDING -- the global
  // maximum of the Brockett criterion.
  std::vector<double> diag(n);
  for (int i = 0; i < n; ++i) {
    diag[i] = re_of(M[static_cast<std::size_t>(i) * n + i]);
  }
  for (int i = 0; i + 1 < n; ++i) {
    EXPECT_GE(diag[i + 1], diag[i] - tol) << "diagonal not ascending at " << i << " n=" << n;
  }
  std::vector<double> diag_sorted = diag;
  std::sort(diag_sorted.begin(), diag_sorted.end());
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(diag_sorted[i], static_cast<double>(eig[i]), tol)
        << "eigenvalue " << i << " n=" << n;
  }
}

} // namespace

// ── Polynomial line search (Table 1), all four element types ─────────────────

TEST(CgUnitaryBrockettTests, DoubleReal) {
  expect_brockett_maximizes<double>(4, 1, LineSearchMethod::Polynomial);
  expect_brockett_maximizes<double>(6, 7, LineSearchMethod::Polynomial);
}

TEST(CgUnitaryBrockettTests, DoubleComplex) {
  expect_brockett_maximizes<wwr::wwrDoubleComplex>(4, 2, LineSearchMethod::Polynomial);
  expect_brockett_maximizes<wwr::wwrDoubleComplex>(6, 11, LineSearchMethod::Polynomial);
}

TEST(CgUnitaryBrockettTests, FloatReal) {
  expect_brockett_maximizes<float>(4, 3, LineSearchMethod::Polynomial);
}

TEST(CgUnitaryBrockettTests, FloatComplex) {
  expect_brockett_maximizes<wwr::wwrFloatComplex>(4, 5, LineSearchMethod::Polynomial);
}

// ── DFT line search (Table 2), the heavier method with its own root solver ────

TEST(CgUnitaryBrockettTests, DoubleComplexDft) {
  expect_brockett_maximizes<wwr::wwrDoubleComplex>(5, 13, LineSearchMethod::Dft);
}

} // namespace calaman
