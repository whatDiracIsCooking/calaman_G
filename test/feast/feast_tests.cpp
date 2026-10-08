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
//   * an empty interval, where feast must report m = 0 and converge;
//   * the dense model behind a wrapper hiding its norm1_estimate hook, so the
//     residuals' ||A||_1 is lacn2's estimate: same eigenpairs, and an estimate
//     never above the exact norm (and equal to it on a diagonal matrix).
//   * KrylovResolvent, the matrix-free model: its filter against the dense
//     model's on the blocks FEAST filters, within the inner tolerance's bound;
//     a whole solve through it against the reference eigenvalues; and an inner
//     solve cut short failing the solve rather than passing silently.
//   * its residual form (IFEAST): filter_residual against the dense filter on
//     the Ritz pairs FEAST produces, within the bound that shrinks with r.
//   * matrix-free end to end: a 2-D Dirichlet Laplacian applied in Kronecker
//     form (never assembled), through the model feast entry point, against its
//     closed-form eigenvalues -- with a loose inner tolerance at which the
//     plain form stalls and the residual form reaches the outer tol.
//
// The host arithmetic is done in double regardless of T, so the oracle shares
// none of feast's device code. float and double (FEAST is real-symmetric only).
// The numerical suites stage the matrices on the device and run the kernels +
// batched BLAS + eigensolver, so they are REQUIRES_GPU (labeled `gpu`, excluded
// by `ctest -LE gpu`); the filter and argument-checking cases are host-only
// (feast_rational_filter and feast's rejection paths return before any
// device work).
//
// Guarded on calaman::lapack_reference (see CMakeLists.txt): the reference suite
// is the eigensolver's natural oracle, and bundling the rest in the same binary
// keeps one target.

#include <gtest/gtest.h>

#include <lapacke.h>

#include "shared/expect_converged.h"

// Returns a failing Status from the enclosing function, as CLM_TRY does in src/.
#define FEAST_TEST_TRY(expr)                                                                       \
  do {                                                                                             \
    if (const ::calaman::Status feast_test_status_ = (expr); !feast_test_status_.ok()) {           \
      return feast_test_status_;                                                                   \
    }                                                                                              \
  } while (0)

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.wrappers.common;
import wwr.wrappers.blas;
import wwr.extension.memory_buffer;
import calaman.feast;
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

template<typename T, std::size_t Ne, class Wrap = std::identity>
FeastResult<T> run_feast(std::shared_ptr<DeviceHandle> handle, Handles &h, int n,
                         const std::vector<T> &a_full, T Emin, T Emax, int m0,
                         const std::vector<T> &q0, FeastOptions<T> opts = {}, Wrap wrap = {}) {
  auto d_a = to_device(handle, a_full);
  auto d_q = to_device(handle, q0);

  std::size_t bytes = 0;
  EXPECT_EQ((feast_bufferSize<T, Ne>(h.solver, n, m0, &bytes)), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<T> d_work(bytes / sizeof(T) + 1, handle);
  DeviceBuffer<T> d_lambda(m0, handle);

  FeastResult<T> r;
  r.status = feast<T, Ne>(h.blas, h.solver, handle->stream().get(), wwr::WWRBLAS_FILL_MODE_LOWER, n,
                          d_a.data(), n, Emin, Emax, m0, d_lambda.data(), d_q.data(), d_work.data(),
                          bytes, opts, &r.info, wrap);
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

  auto call = [&](int nn, int mm0, int lda, double lo, double hi, wwr::wwrblasFillMode_t uplo) {
    return feast<double, 8>(null_blas, null_solver, wwr::wwrStream_t{}, uplo, nn, a.data(), lda, lo,
                            hi, mm0, lambda.data(), q.data(), work.data(),
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
  // Neither LOWER nor UPPER, but inside both enums' value ranges (cuBLAS 0..3,
  // hipBLAS 0..127): a value outside the range, like 999, is UB to load.
  constexpr auto bad_uplo = static_cast<wwr::wwrblasFillMode_t>(2);
  static_assert(bad_uplo != wwr::WWRBLAS_FILL_MODE_LOWER &&
                bad_uplo != wwr::WWRBLAS_FILL_MODE_UPPER);
  EXPECT_EQ(call(n, m0, n, 1.0, 2.0, bad_uplo),
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
  EXPECT_CONVERGED(r.info);
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
  EXPECT_CONVERGED(r.info);
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
  EXPECT_CONVERGED(r.info);
  destroy_handles(h);
}

TEST(FeastEmptyIntervalTests, N8Double) { check_empty_interval<double, 8>(); }
TEST(FeastEmptyIntervalTests, N8Float) { check_empty_interval<float, 8>(); }

// ========================================================================
// Device: a model with no norm1_estimate hook, so ||A||_1 comes from lacn2
// ========================================================================

// Forwards the resolvent model's apply/prepare/filter and nothing else, so the
// driver falls back to lacn2 (driven by apply) for the residuals' scale.
template<class Inner>
struct HideNorm1 {
  Inner &inner;

  template<class T>
  Status apply(wwr::wwrStream_t stream, int k, const T *X, T *Y) {
    return inner.apply(stream, k, X, Y);
  }
  template<class Contour>
  Status prepare(wwr::wwrStream_t stream, const Contour &contour) {
    return inner.prepare(stream, contour);
  }
  template<class Contour, class T>
  Status filter(wwr::wwrStream_t stream, const Contour &contour, int k, const T *Y, T *out) {
    return inner.filter(stream, contour, k, Y, out);
  }
};

template<typename T>
struct HideNorm1Wrap {
  template<class Inner>
  HideNorm1<Inner> operator()(Inner &inner) const {
    static_assert(feast_norm1_hook<Inner, T>, "the dense model has the hook");
    static_assert(feast_resolvent<HideNorm1<Inner>, T>);
    static_assert(!feast_norm1_hook<HideNorm1<Inner>, T>, "the wrapper hides it");
    return HideNorm1<Inner>{inner};
  }
};

// Runs feast on A twice, from the same start -- the dense model as is, and
// behind HideNorm1 -- and checks the hook-less run against the reference
// eigenvalues ref[lo .. lo + count) and against the hooked run. Its ||A||_1 is
// lacn2's: never above the exact one, and equal to it when @p exact_estimate.
template<typename T, std::size_t Ne>
void check_hidden_norm(int n, const std::vector<T> &a, const std::vector<T> &ref, int lo,
                       int count, int m0, bool exact_estimate, unsigned seed) {
  const T emin = static_cast<T>((static_cast<double>(ref[lo - 1]) + ref[lo]) / 2.0);
  const T emax =
      static_cast<T>((static_cast<double>(ref[lo + count - 1]) + ref[lo + count]) / 2.0);

  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto q0 = random_matrix<T>(n, m0, seed);
  const auto dense = run_feast<T, Ne>(handle, h, n, a, emin, emax, m0, q0);
  const auto hidden =
      run_feast<T, Ne>(handle, h, n, a, emin, emax, m0, q0, {}, HideNorm1Wrap<T>{});

  ASSERT_EQ(dense.status, wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hidden.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_CONVERGED(dense.info);
  EXPECT_CONVERGED(hidden.info);
  ASSERT_EQ(dense.info.m, count);
  ASSERT_EQ(hidden.info.m, count);

  // The hook is exact; lacn2's estimate is a lower bound on it.
  const double norm_a = host_norm1(n, a);
  const double ntol = (std::is_same_v<T, float> ? 1e-5 : 1e-12) * norm_a;
  EXPECT_NEAR(dense.info.norm_a, norm_a, ntol) << "hooked ||A||_1";
  EXPECT_GT(hidden.info.norm_a, T(0));
  EXPECT_LE(hidden.info.norm_a, norm_a + ntol) << "lacn2 estimate above ||A||_1";
  if (exact_estimate) {
    EXPECT_NEAR(hidden.info.norm_a, norm_a, ntol) << "lacn2 exact on this class";
  }

  const T evtol = static_cast<T>((std::is_same_v<T, float> ? 1e-3 : 1e-8) * std::max(1.0, norm_a));
  const double restol = std::is_same_v<T, float> ? 1e-3 : 1e-9;
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(hidden.lambda[i], ref[lo + i], evtol) << "eigenvalue " << i;
    EXPECT_NEAR(hidden.lambda[i], dense.lambda[i], evtol) << "vs the hooked run, " << i;
    const double be = pair_backward_error(n, a, norm_a, hidden.lambda[i],
                                          &hidden.q[static_cast<std::size_t>(i) * n]);
    EXPECT_LE(be, restol) << "backward error of pair " << i;
  }
  destroy_handles(h);
}

TEST(FeastHiddenNormTests, DiagonalDouble) {
  // diag(0 .. 11): lacn2 is exact on a diagonal matrix; [2.5, 6.5] holds 3..6.
  const int n = 12;
  std::vector<double> d(n);
  for (int i = 0; i < n; ++i) {
    d[i] = static_cast<double>(i);
  }
  check_hidden_norm<double, 8>(n, diagonal(d), d, 3, 4, 8, true, 1234);
}

TEST(FeastHiddenNormTests, DenseSymmetricDouble) {
  const int n = 24;
  const auto a = random_symmetric<double>(n, 7);
  check_hidden_norm<double, 8>(n, a, reference_eigenvalues<double>(n, a), n / 4, n / 4, n / 4 + 4,
                               false, 8);
}

TEST(FeastHiddenNormTests, DenseSymmetricFloat) {
  const int n = 16;
  const auto a = random_symmetric<float>(n, 7);
  check_hidden_norm<float, 8>(n, a, reference_eigenvalues<float>(n, a), n / 4, n / 4, n / 4 + 4,
                              false, 8);
}

// ========================================================================
// KrylovResolvent: the matrix-free model, shifted_cocg over a linear_operator
// ========================================================================

/// Y = A X for a dense symmetric n x n device matrix (lower triangle): the test
/// matrix as a plain linear_operator, sharing no code with DenseResolvent.
template<typename T>
struct SymmOperator {
  wwr::wwrblasHandle_t blas{};
  int n = 0;
  const T *d_a = nullptr;

  Status apply(wwr::wwrStream_t, const int k, const T *X, T *Y) {
    const T one{1};
    const T zero{0};
    return wwr::symm<T, int>(blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, n, k, &one,
                             d_a, n, X, n, &zero, Y, n);
  }
};

template<typename T>
using Krylov = KrylovResolvent<SymmOperator<T>, T>;

static_assert(feast_resolvent<Krylov<double>, double>);
static_assert(feast_resolvent<Krylov<float>, float>);
static_assert(!feast_resolvent<Krylov<double>, float>);
static_assert(!feast_norm1_hook<Krylov<double>, double>, "no hook: lacn2 scales the residuals");

/// The gap allowed between the inner solves' residual estimate and the filter
/// difference it bounds: rounding in the recurrences and the dense LU (as in
/// shifted_cocg_tests).
constexpr double kInnerSlack = 10.0;

TEST(FeastArgCheckTests, KrylovResolventBufferSize) {
  std::size_t bytes = 0;
  EXPECT_EQ(krylov_resolvent_bufferSize<double>(8, 2, 8, nullptr),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "null lwork";
  EXPECT_EQ(krylov_resolvent_bufferSize<double>(0, 2, 8, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "n < 1";
  EXPECT_EQ(krylov_resolvent_bufferSize<double>(8, 0, 8, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "k_max < 1";
  EXPECT_EQ(krylov_resolvent_bufferSize<double>(8, 2, 0, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "ne < 1";
  EXPECT_EQ(krylov_resolvent_bufferSize<double>(8, 2, 9, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "ne > kFeastMaxNodes";

  // At least shifted_cocg's workspace plus the 2 Ne split solution blocks.
  std::size_t cocg = 0;
  ASSERT_EQ(shifted_cocg_bufferSize<double>(16, 3, 8, &cocg), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(krylov_resolvent_bufferSize<double>(16, 3, 8, &bytes), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_GE(bytes, cocg + 2 * 16 * 3 * 8 * sizeof(double));
}

/// A KrylovResolvent over SymmOperator, with its own copy of A and workspace.
template<typename T>
struct KrylovFixture {
  DeviceBuffer<T> d_a;
  DeviceBuffer<T> d_work;
  SymmOperator<T> op;
  KrylovResolventSlices<T> slices;

  KrylovFixture(std::shared_ptr<DeviceHandle> handle, wwr::wwrblasHandle_t blas, int n,
                const std::vector<T> &a, int k_max)
      : d_a(to_device(handle, a)), d_work(work_elems(n, k_max), handle), op{blas, n, d_a.data()} {
    EXPECT_EQ(make_krylov_resolvent_slices<T>(n, k_max, 8, d_work.data(), &slices, nullptr),
              wwr::WWRBLAS_STATUS_SUCCESS);
  }

  static std::size_t work_elems(int n, int k_max) {
    std::size_t bytes = 0;
    EXPECT_EQ(krylov_resolvent_bufferSize<T>(n, k_max, 8, &bytes), wwr::WWRBLAS_STATUS_SUCCESS);
    return bytes / sizeof(T) + 1;
  }
};

/// Passes the dense model through to feast, and on every filter also runs the
/// Krylov model on the same block and contour, recording per column
///
///   ||out_krylov - out_dense||_2 / (tol * sum_e |w_e| / Im Z_e * ||y||_2),
///
/// the difference in units of what the inner tolerance allows: X_e's error is
/// at most ||(Z_e I - A)^{-1}||_2 tol ||y|| <= tol ||y|| / Im Z_e.
template<class Dense, typename T>
struct CompareFilters {
  Dense &dense;
  Krylov<T> &krylov;
  T *d_kout; // n x k_max
  int n;
  double tol;
  std::vector<double> &ratios; // one per filter call: the worst column
  std::vector<int> &inner_iterations;

  Status apply(wwr::wwrStream_t stream, int k, const T *X, T *Y) {
    return dense.apply(stream, k, X, Y);
  }
  template<class Contour>
  Status prepare(wwr::wwrStream_t stream, const Contour &contour) {
    FEAST_TEST_TRY(krylov.prepare(stream, contour));
    return dense.prepare(stream, contour);
  }
  template<class Contour>
  Status filter(wwr::wwrStream_t stream, const Contour &contour, int k, const T *Y, T *out) {
    FEAST_TEST_TRY(dense.filter(stream, contour, k, Y, out));
    FEAST_TEST_TRY(krylov.filter(stream, contour, k, Y, d_kout));
    inner_iterations.push_back(krylov.last_solve().iterations);

    const std::size_t len = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    std::vector<T> y(len);
    std::vector<T> od(len);
    std::vector<T> ok(len);
    FEAST_TEST_TRY(wwr::wwrMemcpyAsync(y.data(), Y, len * sizeof(T), wwr::wwrMemcpyDeviceToHost,
                                     stream));
    FEAST_TEST_TRY(wwr::wwrMemcpyAsync(od.data(), out, len * sizeof(T), wwr::wwrMemcpyDeviceToHost,
                                     stream));
    FEAST_TEST_TRY(wwr::wwrMemcpyAsync(ok.data(), d_kout, len * sizeof(T),
                                     wwr::wwrMemcpyDeviceToHost, stream));
    FEAST_TEST_TRY(wwr::wwrStreamSynchronize(stream));

    double gain = 0.0; // sum_e |w_e| / Im Z_e
    for (int e = 0; e < contour.count; ++e) {
      gain += std::hypot(static_cast<double>(contour.wr[e]), static_cast<double>(contour.wi[e])) /
              static_cast<double>(contour.zi[e]);
    }
    double worst = 0.0;
    for (int j = 0; j < k; ++j) {
      double diff = 0.0;
      double ynorm = 0.0;
      for (int i = 0; i < n; ++i) {
        const std::size_t at = static_cast<std::size_t>(j) * n + i;
        const double d = static_cast<double>(ok[at]) - static_cast<double>(od[at]);
        diff += d * d;
        ynorm += static_cast<double>(y[at]) * static_cast<double>(y[at]);
      }
      worst = std::max(worst, std::sqrt(diff) / (tol * gain * std::sqrt(ynorm)));
    }
    ratios.push_back(worst);
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

// Runs feast's dense model for a few iterations with CompareFilters in the
// seam, so the Krylov filter is checked on the blocks FEAST actually filters:
// the random start, then partly converged subspaces.
template<typename T>
void check_krylov_filter(int n, unsigned seed, T inner_tol) {
  const auto a = random_symmetric<T>(n, seed);
  const auto w = reference_eigenvalues<T>(n, a);
  const int lo = n / 4;
  const int hi = n / 2;
  const T emin = static_cast<T>((static_cast<double>(w[lo - 1]) + w[lo]) / 2.0);
  const T emax = static_cast<T>((static_cast<double>(w[hi - 1]) + w[hi]) / 2.0);
  const int m0 = hi - lo + 4;

  auto handle = shared_device();
  Handles h = make_handles(handle);
  KrylovFixture<T> fx(handle, h.blas, n, a, m0);
  ShiftedCocgOptions<T> inner;
  inner.tolerance = inner_tol;
  inner.max_iterations = 20 * n;
  Krylov<T> krylov{fx.op, n, fx.slices, inner};
  DeviceBuffer<T> d_kout(static_cast<std::size_t>(n) * m0, handle);

  std::vector<double> ratios;
  std::vector<int> inner_iterations;
  const auto wrap = [&](auto &dense) {
    return CompareFilters<std::remove_reference_t<decltype(dense)>, T>{
        dense, krylov, d_kout.data(), n, static_cast<double>(inner_tol), ratios, inner_iterations};
  };
  FeastOptions<T> opts;
  opts.max_iterations = 3;
  const auto q0 = random_matrix<T>(n, m0, seed + 1);
  const auto r = run_feast<T, 8>(handle, h, n, a, emin, emax, m0, q0, opts, wrap);

  ASSERT_EQ(r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_FALSE(ratios.empty());
  for (std::size_t c = 0; c < ratios.size(); ++c) {
    EXPECT_LE(ratios[c], kInnerSlack) << "filter call " << c;
    EXPECT_GE(inner_iterations[c], 1) << "filter call " << c;
  }
  destroy_handles(h);
}

TEST(FeastKrylovResolventTests, FilterMatchesDenseDouble) {
  check_krylov_filter<double>(24, 7, 1e-10);
}
TEST(FeastKrylovResolventTests, FilterMatchesDenseFloat) {
  check_krylov_filter<float>(16, 7, 1e-4F);
}

// The whole solve through the Krylov model -- inner solves tight enough that
// the plain (non-residual) form reaches the outer tolerance -- against the
// reference eigenvalues.
TEST(FeastKrylovResolventTests, SolveThroughKrylovDouble) {
  const int n = 24;
  const auto a = random_symmetric<double>(n, 7);
  const auto w = reference_eigenvalues<double>(n, a);
  const int lo = n / 4;
  const int hi = n / 2;
  const int count = hi - lo;
  const double emin = (w[lo - 1] + w[lo]) / 2.0;
  const double emax = (w[hi - 1] + w[hi]) / 2.0;
  const int m0 = count + 4;

  auto handle = shared_device();
  Handles h = make_handles(handle);
  KrylovFixture<double> fx(handle, h.blas, n, a, m0);
  ShiftedCocgOptions<double> inner;
  inner.tolerance = 1e-14;
  inner.max_iterations = 20 * n;
  Krylov<double> krylov{fx.op, n, fx.slices, inner};

  const auto wrap = [&](auto &) -> Krylov<double> & { return krylov; };
  const auto q0 = random_matrix<double>(n, m0, 8);
  const auto r = run_feast<double, 8>(handle, h, n, a, emin, emax, m0, q0, {}, wrap);

  ASSERT_EQ(r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_CONVERGED(r.info);
  EXPECT_TRUE(converged(krylov.last_solve()));
  ASSERT_EQ(r.info.m, count);
  const double norm_a = host_norm1(n, a);
  for (int i = 0; i < count; ++i) {
    EXPECT_NEAR(r.lambda[i], w[lo + i], 1e-8 * std::max(1.0, norm_a)) << "eigenvalue " << i;
    const double be =
        pair_backward_error(n, a, norm_a, r.lambda[i], &r.q[static_cast<std::size_t>(i) * n]);
    EXPECT_LE(be, 1e-9) << "backward error of pair " << i;
  }
  destroy_handles(h);
}

// An inner solve cut off before its tolerance fails the filter -- feast stops
// with NumericalFailure -- and last_solve() says why: never a silent pass.
TEST(FeastKrylovResolventTests, InnerNonConvergenceIsReported) {
  const int n = 24;
  const auto a = random_symmetric<double>(n, 7);
  const int m0 = 6;

  auto handle = shared_device();
  Handles h = make_handles(handle);
  KrylovFixture<double> fx(handle, h.blas, n, a, m0);
  ShiftedCocgOptions<double> inner;
  inner.tolerance = 1e-12;
  inner.max_iterations = 2;
  Krylov<double> krylov{fx.op, n, fx.slices, inner};

  const auto wrap = [&](auto &) -> Krylov<double> & { return krylov; };
  const auto q0 = random_matrix<double>(n, m0, 8);
  const auto r = run_feast<double, 8>(handle, h, n, a, -0.5, 0.5, m0, q0, {}, wrap);

  EXPECT_EQ(r.status, wwr::WWRBLAS_STATUS_EXECUTION_FAILED);
  EXPECT_EQ(r.info.reason, FeastStopReason::NumericalFailure);
  EXPECT_EQ(krylov.last_solve().reason, ShiftedCocgStopReason::MaxIterations);
  EXPECT_EQ(krylov.last_solve().iterations, 2);
  destroy_handles(h);
}

// ========================================================================
// The residual form (IFEAST): filter_residual against the dense filter
// ========================================================================

/// Passes the dense model through to feast; on every residual-form call also
/// runs the Krylov model's filter_residual on the same Ritz pairs, recording per
/// column the difference from the dense filter of X in units of its bound
///
///   tol ||r_j|| sum_e |w_e| / (Im Z_e)^2   (+ the dense solves' rounding),
///
/// since X_e's error is at most tol ||r_j|| / Im Z_e and |Z_e - lambda| >= Im Z_e.
template<class Dense, typename T>
struct CompareResidualFilters {
  Dense &dense;
  Krylov<T> &krylov;
  T *d_kout; // n x k_max
  int n;
  double norm_a;
  double tol;
  std::vector<double> &ratios; // one per filter_residual call: the worst column

  Status apply(wwr::wwrStream_t stream, int k, const T *X, T *Y) {
    return dense.apply(stream, k, X, Y);
  }
  template<class Contour>
  Status prepare(wwr::wwrStream_t stream, const Contour &contour) {
    FEAST_TEST_TRY(krylov.prepare(stream, contour));
    return dense.prepare(stream, contour);
  }
  template<class Contour>
  Status filter(wwr::wwrStream_t stream, const Contour &contour, int k, const T *Y, T *out) {
    return dense.filter(stream, contour, k, Y, out);
  }
  template<class Contour>
  Status filter_residual(wwr::wwrStream_t stream, const Contour &contour, int k, const T *X,
                         const T *lambda, const T *R, T *out) {
    FEAST_TEST_TRY(dense.filter(stream, contour, k, X, out));
    FEAST_TEST_TRY(krylov.filter_residual(stream, contour, k, X, lambda, R, d_kout));

    const std::size_t len = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    std::vector<T> x(len);
    std::vector<T> r(len);
    std::vector<T> od(len);
    std::vector<T> ok(len);
    const auto copy = [&](std::vector<T> &to, const T *from) {
      return wwr::wwrMemcpyAsync(to.data(), from, len * sizeof(T), wwr::wwrMemcpyDeviceToHost,
                                 stream);
    };
    FEAST_TEST_TRY(copy(x, X));
    FEAST_TEST_TRY(copy(r, R));
    FEAST_TEST_TRY(copy(od, out));
    FEAST_TEST_TRY(copy(ok, d_kout));
    FEAST_TEST_TRY(wwr::wwrStreamSynchronize(stream));

    double gain2 = 0.0; // sum_e |w_e| / (Im Z_e)^2
    double zmax = 0.0;
    for (int e = 0; e < contour.count; ++e) {
      const double zi = static_cast<double>(contour.zi[e]);
      gain2 += std::hypot(static_cast<double>(contour.wr[e]), static_cast<double>(contour.wi[e])) /
               (zi * zi);
      zmax = std::max(zmax, std::hypot(static_cast<double>(contour.zr[e]), zi));
    }
    // The dense LU's own error on (Z_e I - A)^{-1} x: eps (||A|| + |Z_e|) / Im Z_e^2.
    const double rounding = 64.0 * static_cast<double>(test::eps<T>()) * (norm_a + zmax) * gain2;
    double worst = 0.0;
    for (int j = 0; j < k; ++j) {
      double diff = 0.0;
      double rnorm = 0.0;
      double xnorm = 0.0;
      for (int i = 0; i < n; ++i) {
        const std::size_t at = static_cast<std::size_t>(j) * n + i;
        const double d = static_cast<double>(ok[at]) - static_cast<double>(od[at]);
        diff += d * d;
        rnorm += static_cast<double>(r[at]) * static_cast<double>(r[at]);
        xnorm += static_cast<double>(x[at]) * static_cast<double>(x[at]);
      }
      const double bound = tol * gain2 * std::sqrt(rnorm) + rounding * std::sqrt(xnorm);
      worst = std::max(worst, std::sqrt(diff) / bound);
    }
    ratios.push_back(worst);
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

TEST(FeastKrylovResolventTests, ResidualFilterMatchesDenseDouble) {
  const int n = 24;
  const unsigned seed = 7;
  const double inner_tol = 1e-6;
  const auto a = random_symmetric<double>(n, seed);
  const auto w = reference_eigenvalues<double>(n, a);
  const int lo = n / 4;
  const int hi = n / 2;
  const double emin = (w[lo - 1] + w[lo]) / 2.0;
  const double emax = (w[hi - 1] + w[hi]) / 2.0;
  const int m0 = hi - lo + 4;

  auto handle = shared_device();
  Handles h = make_handles(handle);
  KrylovFixture<double> fx(handle, h.blas, n, a, m0);
  ShiftedCocgOptions<double> inner;
  inner.tolerance = inner_tol;
  inner.max_iterations = 20 * n;
  Krylov<double> krylov{fx.op, n, fx.slices, inner};
  DeviceBuffer<double> d_kout(static_cast<std::size_t>(n) * m0, handle);

  std::vector<double> ratios;
  const auto wrap = [&](auto &dense) {
    return CompareResidualFilters<std::remove_reference_t<decltype(dense)>, double>{
        dense, krylov, d_kout.data(), n, host_norm1(n, a), inner_tol, ratios};
  };
  FeastOptions<double> opts;
  opts.max_iterations = 3;
  const auto q0 = random_matrix<double>(n, m0, seed + 1);
  const auto r = run_feast<double, 8>(handle, h, n, a, emin, emax, m0, q0, opts, wrap);

  ASSERT_EQ(r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_FALSE(ratios.empty()) << "the driver never took the residual form";  for (std::size_t c = 0; c < ratios.size(); ++c) {
    EXPECT_LE(ratios[c], kInnerSlack) << "filter_residual call " << c;
  }
  destroy_handles(h);
}

// ========================================================================
// Matrix-free end to end: the 2-D Dirichlet Laplacian, never assembled
// ========================================================================

/// The 5-point Laplacian on an N x N grid, n = N^2, as the Kronecker sum
/// L = I (x) T + T (x) I with T = tridiag(-1, 2, -1): each column of X, read as
/// an N x N grid U, maps to T U + U T. Only the N x N T is stored, never L.
template<typename T>
struct LaplacianOperator {
  wwr::wwrblasHandle_t blas{};
  int grid = 0;           // N
  const T *d_t = nullptr; // N x N

  Status apply(wwr::wwrStream_t, const int k, const T *X, T *Y) {
    const T one{1};
    const T zero{0};
    // T U for every column at once: X is N x (N k).
    FEAST_TEST_TRY((wwr::gemm<T, int>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, grid, grid * k,
                                      grid, &one, d_t, grid, X, grid, &zero, Y, grid)));
    // + U T per column, T broadcast with stride 0.
    const long long sq = static_cast<long long>(grid) * grid;
    return wwr::gemmStridedBatched<T, int>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, grid, grid,
                                           grid, &one, X, grid, sq, d_t, grid, 0, &one, Y, grid,
                                           sq, k);
  }
};

/// The Laplacian's eigenvalues in closed form, ascending:
/// (2 - 2 cos(p pi / (N+1))) + (2 - 2 cos(q pi / (N+1))), 1 <= p, q <= N.
std::vector<double> laplacian_eigenvalues(const int grid) {
  std::vector<double> mu(grid);
  for (int p = 1; p <= grid; ++p) {
    mu[p - 1] = 2.0 - 2.0 * std::cos(p * std::numbers::pi / (grid + 1));
  }
  std::vector<double> w;
  for (const double a : mu) {
    for (const double b : mu) {
      w.push_back(a + b);
    }
  }
  std::ranges::sort(w);
  return w;
}

/// Forwards apply/prepare/filter only, hiding filter_residual: the plain form.
template<class Inner>
struct PlainForm {
  Inner &inner;

  template<class T>
  Status apply(wwr::wwrStream_t stream, int k, const T *X, T *Y) {
    return inner.apply(stream, k, X, Y);
  }
  template<class Contour>
  Status prepare(wwr::wwrStream_t stream, const Contour &contour) {
    return inner.prepare(stream, contour);
  }
  template<class Contour, class T>
  Status filter(wwr::wwrStream_t stream, const Contour &contour, int k, const T *Y, T *out) {
    return inner.filter(stream, contour, k, Y, out);
  }
};

using Laplacian = KrylovResolvent<LaplacianOperator<double>, double>;
static_assert(feast_residual_hook<Laplacian, double>);
static_assert(feast_resolvent<PlainForm<Laplacian>, double>);
static_assert(!feast_residual_hook<PlainForm<Laplacian>, double>);

/// The Laplacian problem: grid, closed-form spectrum, and an interval whose ends
/// sit in gaps of the spectrum (it has double eigenvalues, p != q).
struct LaplacianProblem {
  int grid = 16;
  int n = 256;
  std::vector<double> w;
  int lo = 0;
  int count = 0;
  double emin = 0.0;
  double emax = 0.0;
  int m0 = 0;

  LaplacianProblem() : w(laplacian_eigenvalues(grid)) {
    int first = 6;
    int last = 16;
    while (w[first] - w[first - 1] < 1e-2) {
      ++first;
    }
    while (w[last] - w[last - 1] < 1e-2) {
      ++last;
    }
    lo = first;
    count = last - first;
    emin = (w[first - 1] + w[first]) / 2.0;
    emax = (w[last - 1] + w[last]) / 2.0;
    m0 = count + 8;
  }
};

struct MatrixFreeRun {
  FeastResult<double> r;
  ShiftedCocgInfo<double> last_solve;
};

/// feast through the model entry point over KrylovResolvent<LaplacianOperator>,
/// inner tolerance @p inner_tol, in residual form or (@p plain) not.
MatrixFreeRun run_laplacian(const LaplacianProblem &p, const double inner_tol, const bool plain,
                            const FeastOptions<double> &opts) {
  auto handle = shared_device();
  Handles h = make_handles(handle);

  std::vector<double> t(static_cast<std::size_t>(p.grid) * p.grid, 0.0);
  for (int i = 0; i < p.grid; ++i) {
    t[static_cast<std::size_t>(i) * p.grid + i] = 2.0;
    if (i + 1 < p.grid) {
      t[static_cast<std::size_t>(i) * p.grid + i + 1] = -1.0;
      t[static_cast<std::size_t>(i + 1) * p.grid + i] = -1.0;
    }
  }
  auto d_t = to_device(handle, t);
  LaplacianOperator<double> op{h.blas, p.grid, d_t.data()};

  std::size_t model_bytes = 0;
  EXPECT_EQ(krylov_resolvent_bufferSize<double>(p.n, p.m0, 8, &model_bytes),
            wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<double> d_model(model_bytes / sizeof(double) + 1, handle);
  KrylovResolventSlices<double> slices;
  EXPECT_EQ(make_krylov_resolvent_slices<double>(p.n, p.m0, 8, d_model.data(), &slices, nullptr),
            wwr::WWRBLAS_STATUS_SUCCESS);
  ShiftedCocgOptions<double> inner;
  inner.tolerance = inner_tol;
  inner.max_iterations = 8 * p.n;
  Laplacian krylov{op, p.n, slices, inner};

  std::size_t bytes = 0;
  EXPECT_EQ(feast_driver_bufferSize<double>(h.solver, p.n, p.m0, &bytes),
            wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<double> d_work(bytes / sizeof(double) + 1, handle);
  DeviceBuffer<double> d_lambda(p.m0, handle);
  auto d_q = to_device(handle, random_matrix<double>(p.n, p.m0, 2024));

  MatrixFreeRun run;
  const auto solve = [&](auto &model) {
    return feast<double, 8>(h.blas, h.solver, handle->stream().get(), model, p.n, p.emin, p.emax,
                            p.m0, d_lambda.data(), d_q.data(), d_work.data(), bytes, opts,
                            &run.r.info);
  };
  if (plain) {
    PlainForm<Laplacian> model{krylov};
    run.r.status = solve(model);
  } else {
    run.r.status = solve(krylov);
  }
  wwr::wwrStreamSynchronize(handle->stream().get());
  run.r.lambda = from_device(handle, d_lambda, p.m0);
  run.last_solve = krylov.last_solve();
  destroy_handles(h);
  return run;
}

/// Loose next to the outer tol (1e-12): the plain form's floor sits far above it.
constexpr double kLooseInnerTol = 1e-4;

TEST(FeastMatrixFreeTests, LaplacianResidualFormMatchesClosedForm) {
  const LaplacianProblem p;
  ASSERT_GT(p.count, 0);
  FeastOptions<double> opts;
  opts.max_iterations = 30;
  const auto run = run_laplacian(p, kLooseInnerTol, false, opts);

  ASSERT_EQ(run.r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_CONVERGED(run.r.info);
  EXPECT_LT(run.r.info.max_residual, opts.tol);
  EXPECT_TRUE(converged(run.last_solve));
  ASSERT_EQ(run.r.info.m, p.count) << "count in [Emin, Emax]";
  // ||L||_1 = 8: an interior grid point's column is 4 + 4 * |-1|.
  const double tol = test::factorization_tol<double>(8.0, p.n, p.n);
  for (int i = 0; i < p.count; ++i) {
    EXPECT_NEAR(run.r.lambda[i], p.w[p.lo + i], tol) << "eigenvalue " << i;
  }
}

TEST(FeastMatrixFreeTests, PlainFormStallsWhereResidualFormConverges) {
  const LaplacianProblem p;
  FeastOptions<double> opts;
  opts.max_iterations = 30;
  const auto plain = run_laplacian(p, kLooseInnerTol, true, opts);
  const auto residual = run_laplacian(p, kLooseInnerTol, false, opts);

  // Same inner tolerance, same start: the plain form runs out of iterations
  // with its residual stuck orders of magnitude above tol ...
  ASSERT_EQ(plain.r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(plain.r.info.reason, FeastStopReason::MaxIterations);
  EXPECT_EQ(plain.r.info.iterations, opts.max_iterations);
  EXPECT_GT(plain.r.info.max_residual, 1e3 * opts.tol);
  // ... while the residual form converges, well inside the budget.
  ASSERT_EQ(residual.r.status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_CONVERGED(residual.r.info);
  EXPECT_LT(residual.r.info.iterations, opts.max_iterations);
}

// The model entry point's argument checks return before any device work.
struct NullModel {
  Status apply(wwr::wwrStream_t, int, const double *, double *) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  template<class Contour>
  Status prepare(wwr::wwrStream_t, const Contour &) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
  template<class Contour>
  Status filter(wwr::wwrStream_t, const Contour &, int, const double *, double *) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

TEST(FeastArgCheckTests, ModelEntryPointRejectsBadArguments) {
  static_assert(feast_resolvent<NullModel, double>);
  const int n = 4;
  const int m0 = 2;
  std::vector<double> lambda(m0, 0.0);
  std::vector<double> q(static_cast<std::size_t>(n) * m0, 1.0);
  std::vector<double> work(1024, 0.0);
  wwr::wwrblasHandle_t null_blas{};
  wwr::wwrsolverDnHandle_t null_solver{};
  NullModel model;

  auto call = [&](int nn, int mm0, double lo, double hi, double *d_lambda) {
    return feast<double, 8>(null_blas, null_solver, wwr::wwrStream_t{}, model, nn, lo, hi, mm0,
                            d_lambda, q.data(), work.data(), work.size() * sizeof(double));
  };
  EXPECT_EQ(call(0, m0, 1.0, 2.0, lambda.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "n < 1";
  EXPECT_EQ(call(n, 0, 1.0, 2.0, lambda.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "m0 < 1";
  EXPECT_EQ(call(n, n + 1, 1.0, 2.0, lambda.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "m0 > n";
  EXPECT_EQ(call(n, m0, 2.0, 1.0, lambda.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "Emin >= Emax";
  EXPECT_EQ(call(n, m0, 1.0, std::numeric_limits<double>::infinity(), lambda.data()),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "non-finite Emax";
  EXPECT_EQ(call(n, m0, 1.0, 2.0, nullptr), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "null lambda";

  std::size_t bytes = 0;
  EXPECT_EQ(feast_driver_bufferSize<double>(null_solver, n, m0, nullptr),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "null lwork";
  EXPECT_EQ(feast_driver_bufferSize<double>(null_solver, n, n + 1, &bytes),
            wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "m0 > n";
}

} // namespace
} // namespace calaman
