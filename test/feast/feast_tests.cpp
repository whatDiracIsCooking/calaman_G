// Oracle test for calaman.feast -- the FEAST eigensolver for real symmetric
// matrices. FEAST returns the eigenpairs with eigenvalues in [Emin, Emax], so
// the oracles are:
//
//   * the rational filter rho -- a pure host function with known values: exactly
//     1 at the interval centre, symmetric about it, and decaying outside at the
//     rate the paper quotes (2.1e-3 two radii out with 4 nodes, 1.6e-5 with 8);
//   * a diagonal matrix, whose eigenpairs (d_i, e_i) are known exactly, so the
//     count, the values and the vectors in a chosen interval are all asserted
//     against closed forms;
//   * the reference LAPACK -- LAPACKE_?syevd on a dense random symmetric matrix
//     gives every eigenpair; the ones in [Emin, Emax] are the oracle feast is
//     checked against (count, values, and per-pair backward error);
//   * an empty interval, where feast must report m = 0 and converge.
//
// The host arithmetic is done in double regardless of T, so the oracle shares
// none of feast's device code. float and double (FEAST is real-symmetric only).
// The numerical suites stage the matrices on the device and run the kernels +
// batched BLAS + eigensolver, so they are REQUIRES_GPU (labeled `gpu`, excluded
// by `ctest -LE gpu`); the filter and argument-checking cases are host-only
// (feast_rational_filter and feast_solver's rejection paths return before any
// device work).
//
// Guarded on calaman::lapack_reference (see CMakeLists.txt): the reference suite
// is the eigensolver's natural oracle, and bundling the rest in the same binary
// keeps one target.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.feast;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

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

struct Handles {
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
};

Handles make_handles(std::shared_ptr<DeviceHandle> handle) {
  Handles h;
  EXPECT_EQ(wwr::wwrblasCreate(&h.blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(h.blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnCreate(&h.solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnSetStream(h.solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  return h;
}

void destroy_handles(Handles &h) {
  wwr::wwrblasDestroy(h.blas);
  wwr::wwrsolverDnDestroy(h.solver);
}

// ── host helpers (all arithmetic in double) ──────────────────────────────────

// Full (both triangles) random symmetric matrix, column-major.
template<typename T>
std::vector<T> random_symmetric(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      const T v = static_cast<T>(dist(rng));
      a[static_cast<std::size_t>(j) * n + i] = v;
      a[static_cast<std::size_t>(i) * n + j] = v;
    }
  }
  return a;
}

// A diagonal matrix with the given diagonal, column-major n x n.
template<typename T>
std::vector<T> diagonal(const std::vector<T> &diag) {
  const int n = static_cast<int>(diag.size());
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    a[static_cast<std::size_t>(i) * n + i] = diag[i];
  }
  return a;
}

// A random n x m starting subspace (full column rank with probability 1).
template<typename T>
std::vector<T> random_matrix(int n, int m, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<T> q(static_cast<std::size_t>(n) * m);
  for (auto &x : q) {
    x = static_cast<T>(dist(rng));
  }
  return q;
}

// Induced 1-norm (max absolute column sum) of a full n x n matrix.
template<typename T>
double host_norm1(int n, const std::vector<T> &a) {
  double best = 0.0;
  for (int j = 0; j < n; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
      s += std::abs(static_cast<double>(a[static_cast<std::size_t>(j) * n + i]));
    }
    best = std::max(best, s);
  }
  return best;
}

// Relative backward error ||A q - lambda q||_2 / (||A||_1 ||q||_2) of one pair,
// with the full symmetric A on the host.
template<typename T>
double pair_backward_error(int n, const std::vector<T> &a, double norm_a, T lambda, const T *q) {
  std::vector<double> aq(n, 0.0);
  for (int j = 0; j < n; ++j) {
    const double qj = static_cast<double>(q[j]);
    for (int i = 0; i < n; ++i) {
      aq[i] += static_cast<double>(a[static_cast<std::size_t>(j) * n + i]) * qj;
    }
  }
  double nr = 0.0;
  double nq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double qi = static_cast<double>(q[i]);
    const double ri = aq[i] - static_cast<double>(lambda) * qi;
    nr += ri * ri;
    nq += qi * qi;
  }
  return std::sqrt(nr) / (norm_a * std::sqrt(nq));
}

// Reference eigenvalues (ascending) of the symmetric A, via the reference LAPACK.
// Takes a copy -- LAPACKE_?syevd overwrites its input with the eigenvectors.
lapack_int call_syevd(int n, float *a, float *w) {
  return LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}
lapack_int call_syevd(int n, double *a, double *w) {
  return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}

template<typename T>
std::vector<T> reference_eigenvalues(int n, std::vector<T> a) {
  std::vector<T> w(n);
  EXPECT_EQ(call_syevd(n, a.data(), w.data()), 0);
  return w;
}

// ── device runner ─────────────────────────────────────────────────────────────

template<typename T>
struct FeastResult {
  Status status{wwr::WWRBLAS_STATUS_SUCCESS};
  FeastInfo<T> info{};
  std::vector<T> lambda; // m0
  std::vector<T> q;      // n * m0
};

template<typename T, std::size_t Ne>
FeastResult<T> run_feast(std::shared_ptr<DeviceHandle> handle, Handles &h, int n,
                         const std::vector<T> &a_full, T Emin, T Emax, int m0,
                         const std::vector<T> &q0, FeastOptions<T> opts = {}) {
  auto d_a = to_device(handle, a_full);
  auto d_q = to_device(handle, q0);

  std::size_t bytes = 0;
  EXPECT_EQ((feast_bufferSize<T, Ne>(h.solver, n, m0, &bytes)), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<T> d_work(bytes / sizeof(T) + 1, handle);
  DeviceBuffer<T> d_lambda(m0, handle);

  FeastResult<T> r;
  r.status = feast_solver<T, Ne>(h.blas, h.solver, handle->stream().get(),
                                 wwr::WWRBLAS_FILL_MODE_LOWER, n, d_a.data(), n, Emin, Emax, m0,
                                 d_lambda.data(), d_q.data(), d_work.data(), bytes, opts, &r.info);
  wwr::wwrStreamSynchronize(handle->stream().get());
  r.lambda = from_device(handle, d_lambda, m0);
  r.q = from_device(handle, d_q, static_cast<std::size_t>(n) * m0);
  return r;
}

// ========================================================================
// Host-only: the rational filter
// ========================================================================

template<typename T, std::size_t N>
void check_filter_shape() {
  // Interval [1, 3]: centre 2, radius 1. Two radii out is lambda = 4.
  const T emin = T(1);
  const T emax = T(3);
  const T center = T(2);
  const T two_radii_out = T(4);
  const T tol = std::is_same_v<T, float> ? T(1e-4) : T(1e-9);

  // Exactly 1 at the centre (every quadrature term is omega_e/2, and the GL
  // weights sum to 2). Locals keep the template-argument comma out of the macro.
  const T rho_center = feast_rational_filter<T, N>(emin, emax, center);
  EXPECT_NEAR(rho_center, T(1), tol);

  // Symmetric about the centre.
  const T rho_left = feast_rational_filter<T, N>(emin, emax, T(1.5));
  const T rho_right = feast_rational_filter<T, N>(emin, emax, T(2.5));
  EXPECT_NEAR(rho_left, rho_right, tol);

  // Falls away outside: strictly smaller than at the centre, and tiny two radii
  // out -- the paper's 2.1e-3 (N=4) / 1.6e-5 (N=8), asserted with a margin.
  const T rho_out = feast_rational_filter<T, N>(emin, emax, two_radii_out);
  const T out_bound = std::is_same_v<T, float> ? T(5e-2) : (N == 8 ? T(1e-4) : T(5e-3));
  EXPECT_LT(std::abs(rho_out), out_bound);
  EXPECT_LT(std::abs(rho_out), T(1));
}

TEST(FeastQuadratureTests, FilterShapeN4Double) { check_filter_shape<double, 4>(); }
TEST(FeastQuadratureTests, FilterShapeN8Double) { check_filter_shape<double, 8>(); }
TEST(FeastQuadratureTests, FilterShapeN8Float) { check_filter_shape<float, 8>(); }

TEST(FeastQuadratureTests, MoreNodesDecayFaster) {
  // Two radii out, 8 nodes damp the filter far harder than 4.
  const double r4 = std::abs(feast_rational_filter<double, 4>(1.0, 3.0, 4.0));
  const double r8 = std::abs(feast_rational_filter<double, 8>(1.0, 3.0, 4.0));
  EXPECT_LT(r8, r4);
}

// ========================================================================
// Host-only: argument validation (rejection paths return before device work)
// ========================================================================

TEST(FeastArgCheckTests, RejectsBadArgumentsBeforeTouchingTheDevice) {
  const int n = 4;
  const int m0 = 2;
  std::vector<double> a(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<double> lambda(m0, 0.0);
  std::vector<double> q(static_cast<std::size_t>(n) * m0, 1.0);
  std::vector<double> work(1024, 0.0);
  wwr::wwrblasHandle_t null_blas{};
  wwr::wwrsolverDnHandle_t null_solver{};

  auto call = [&](int nn, int mm0, int lda, double lo, double hi,
                  wwr::wwrblasFillMode_t uplo) {
    return feast_solver<double, 8>(null_blas, null_solver, wwr::wwrStream_t{}, uplo, nn, a.data(),
                                   lda, lo, hi, mm0, lambda.data(), q.data(), work.data(),
                                   work.size() * sizeof(double), {}, nullptr);
  };
  const auto lower = wwr::WWRBLAS_FILL_MODE_LOWER;
  EXPECT_EQ(call(0, m0, n, 1.0, 2.0, lower), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "n < 1";
  EXPECT_EQ(call(n, 0, n, 1.0, 2.0, lower), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "m0 < 1";
  EXPECT_EQ(call(n, n + 1, n, 1.0, 2.0, lower), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "m0 > n";
  EXPECT_EQ(call(n, m0, n - 1, 1.0, 2.0, lower), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "lda < n";
  EXPECT_EQ(call(n, m0, n, 2.0, 1.0, lower), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "Emin >= Emax";
  EXPECT_EQ(call(n, m0, n, std::nan(""), 2.0, lower), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "non-finite Emin";
  EXPECT_EQ(call(n, m0, n, 1.0, 2.0, static_cast<wwr::wwrblasFillMode_t>(999)),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "bad uplo";

  // The workspace query rejects a null out-pointer and a bad dimension before it
  // ever touches the (null) handle.
  std::size_t bytes = 0;
  EXPECT_EQ((feast_bufferSize<double, 8>(null_solver, n, m0, nullptr)),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "null lwork";
  EXPECT_EQ((feast_bufferSize<double, 8>(null_solver, 0, m0, &bytes)),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "n < 1";
  EXPECT_EQ((feast_bufferSize<double, 8>(null_solver, n, n + 1, &bytes)),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "m0 > n";
}

// ========================================================================
// Device: a diagonal matrix, whose eigenpairs are known exactly
// ========================================================================

template<typename T, std::size_t Ne>
void check_diagonal() {
  // A = diag(0, 1, 2, ..., n-1). The interval [2.5, 6.5] holds exactly
  // {3, 4, 5, 6}, with its endpoints well clear of any eigenvalue.
  const int n = 12;
  std::vector<T> diag(n);
  for (int i = 0; i < n; ++i) {
    diag[i] = static_cast<T>(i);
  }
  const auto a = diagonal(diag);
  const T emin = T(2.5);
  const T emax = T(6.5);
  const std::vector<T> expected = {T(3), T(4), T(5), T(6)};
  const int count = static_cast<int>(expected.size());
  const int m0 = 8;

  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto q0 = random_matrix<T>(n, m0, 1234);
  const auto r = run_feast<T, Ne>(handle, h, n, a, emin, emax, m0, q0);

  ASSERT_EQ(r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(static_cast<int>(r.info.reason), static_cast<int>(FeastStopReason::Converged));
  ASSERT_EQ(r.info.m, count);

  const double norm_a = host_norm1(n, a);
  const T evtol = std::is_same_v<T, float> ? T(1e-3) : T(1e-9);
  const double restol = std::is_same_v<T, float> ? 1e-3 : 1e-9;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(r.lambda[i], expected[i], evtol) << "eigenvalue " << i;
    const double be = pair_backward_error(n, a, norm_a, r.lambda[i], &r.q[static_cast<std::size_t>(i) * n]);
    EXPECT_LE(be, restol) << "backward error of pair " << i;
  }
  destroy_handles(h);
}

TEST(FeastDiagonalTests, N8Double) { check_diagonal<double, 8>(); }
TEST(FeastDiagonalTests, N4Double) { check_diagonal<double, 4>(); }
TEST(FeastDiagonalTests, N8Float) { check_diagonal<float, 8>(); }

// ========================================================================
// Device: a dense random symmetric matrix against the reference LAPACK
// ========================================================================

template<typename T, std::size_t Ne>
void check_reference(int n, unsigned seed) {
  const auto a = random_symmetric<T>(n, seed);
  const auto w = reference_eigenvalues<T>(n, a); // ascending

  // Bracket an interior run of eigenvalues, with the endpoints at the midpoints
  // between neighbours so no eigenvalue sits on the boundary and the count is
  // unambiguous.
  const int lo = n / 4;
  const int hi = n / 2;
  const int count = hi - lo;
  ASSERT_GT(count, 0);
  const T emin = static_cast<T>((static_cast<double>(w[lo - 1]) + static_cast<double>(w[lo])) / 2.0);
  const T emax =
      static_cast<T>((static_cast<double>(w[hi - 1]) + static_cast<double>(w[hi])) / 2.0);
  const int m0 = std::min(n, count + 4);

  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto q0 = random_matrix<T>(n, m0, seed + 1);
  const auto r = run_feast<T, Ne>(handle, h, n, a, emin, emax, m0, q0);

  ASSERT_EQ(r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(static_cast<int>(r.info.reason), static_cast<int>(FeastStopReason::Converged));
  ASSERT_EQ(r.info.m, count) << "count in [Emin, Emax]";

  const double norm_a = host_norm1(n, a);
  const T evtol = static_cast<T>((std::is_same_v<T, float> ? 1e-3 : 1e-8) * std::max(1.0, norm_a));
  const double restol = std::is_same_v<T, float> ? 1e-3 : 1e-9;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(r.lambda[i], w[lo + i], evtol) << "eigenvalue " << i;
    const double be =
        pair_backward_error(n, a, norm_a, r.lambda[i], &r.q[static_cast<std::size_t>(i) * n]);
    EXPECT_LE(be, restol) << "backward error of pair " << i;
  }
  destroy_handles(h);
}

TEST(FeastReferenceTests, DenseSymmetricDouble) { check_reference<double, 8>(24, 7); }
TEST(FeastReferenceTests, DenseSymmetricDoubleOtherSeed) { check_reference<double, 8>(20, 99); }
TEST(FeastReferenceTests, DenseSymmetricFloat) { check_reference<float, 8>(16, 7); }

// ========================================================================
// Device: an interval holding no eigenvalues
// ========================================================================

template<typename T, std::size_t Ne>
void check_empty_interval() {
  // A = diag(0 .. n-1); an interval above the whole spectrum holds nothing. feast
  // must report m = 0 and converge (the "m repeats" rule tolerates a stray Ritz
  // value crowding in on the first iteration).
  const int n = 10;
  std::vector<T> diag(n);
  for (int i = 0; i < n; ++i) {
    diag[i] = static_cast<T>(i);
  }
  const auto a = diagonal(diag);
  const T emin = static_cast<T>(n + 5);
  const T emax = static_cast<T>(n + 10);
  const int m0 = 4;

  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto q0 = random_matrix<T>(n, m0, 555);
  const auto r = run_feast<T, Ne>(handle, h, n, a, emin, emax, m0, q0);

  ASSERT_EQ(r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(r.info.m, 0);
  EXPECT_EQ(static_cast<int>(r.info.reason), static_cast<int>(FeastStopReason::Converged));
  destroy_handles(h);
}

TEST(FeastEmptyIntervalTests, N8Double) { check_empty_interval<double, 8>(); }
TEST(FeastEmptyIntervalTests, N8Float) { check_empty_interval<float, 8>(); }

} // namespace
} // namespace calaman
