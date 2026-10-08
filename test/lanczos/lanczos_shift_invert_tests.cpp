// Suite for calaman.lanczos's :shift_invert partition.
//
//   * LanczosShiftInvertTransformTests (host-only): shift_invert_back_transform
//     maps theta to lambda = sigma + 1/theta and sorts ascending, reporting the
//     source of each output; the shifted_operator concept takes DenseShiftInvert
//     and refuses a linear_operator with no sigma().
//   * LanczosShiftInvertReferenceTests (REQUIRES_GPU): lanczos_shift_invert_solve
//     on calaman.shift_invert's DenseShiftInvert of a dense A = Q diag(lambda)
//     Q^T, against LAPACKE_?syevd on the same A, float and double. The nev
//     eigenvalues nearest sigma (straddling it, so both theta signs and the
//     eigenvector reordering are exercised) match the reference, ascending, to
//     the shared tolerance; each Ritz vector's ||A x - lambda x|| meets it, the
//     block is orthonormal, and each vector is within the Davis-Kahan angle of
//     the reference's. Plus: verify_residuals, and an unprepared operator
//     surfacing as a failing Status.
//   * LanczosShiftInvertInteriorTests (REQUIRES_GPU): the same checks against
//     LAPACKE_?syevr over just the wanted index window (plus one neighbour each
//     side), float and double: windows across a graded spectrum, and the edge
//     cases -- sigma between two close eigenvalues, sigma very near one, sigma
//     outside the spectrum, and sigma exactly on one (prepare fails, and so
//     does the solve, leaving its output untouched).

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

#include "shared/expect_converged.h"

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lanczos;
import calaman.shift_invert;
import calaman.common;
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

// ── host side ────────────────────────────────────────────────────────────────

/// A linear_operator with no sigma(): not a shifted_operator.
struct NoShift {
  Status apply(wwr::wwrStream_t, int, const double *, double *) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};
static_assert(linear_operator<NoShift, double>);
static_assert(!shifted_operator<NoShift, double>);
static_assert(shifted_operator<DenseShiftInvert<float>, float>);
static_assert(shifted_operator<DenseShiftInvert<double>, double>);
static_assert(!shifted_operator<DenseShiftInvert<double>, float>);

template<typename T>
void check_back_transform() {
  const T sigma = T(0.5);
  // Ascending theta, as lanczos_solve returns them: negative theta map below
  // sigma, positive above, each group reversed.
  const std::vector<T> theta = {T(-4), T(-0.5), T(2), T(10)};
  std::vector<T> values = theta;
  const std::vector<int> order = shift_invert_back_transform<T>(sigma, values);
  const std::vector<int> expected_order = {1, 0, 3, 2};
  EXPECT_EQ(order, expected_order);
  for (std::size_t j = 0; j < values.size(); ++j) {
    EXPECT_EQ(values[j], sigma + T(1) / theta[static_cast<std::size_t>(order[j])]) << j;
    if (j > 0) {
      EXPECT_LT(values[j - 1], values[j]) << j;
    }
  }
  EXPECT_EQ(values[0], T(-1.5));
  EXPECT_EQ(values[1], T(0.25));
  EXPECT_EQ(values[3], T(1));

  std::vector<T> none;
  EXPECT_TRUE(shift_invert_back_transform<T>(sigma, none).empty());
}

TEST(LanczosShiftInvertTransformTests, BackTransformAndOrderDouble) {
  check_back_transform<double>();
}
TEST(LanczosShiftInvertTransformTests, BackTransformAndOrderFloat) {
  check_back_transform<float>();
}

// One sign only (sigma below the spectrum): a pure reversal.
TEST(LanczosShiftInvertTransformTests, OneSignReverses) {
  std::vector<double> values = {1.0, 2.0, 4.0};
  const std::vector<int> order = shift_invert_back_transform<double>(-1.0, values);
  EXPECT_EQ(order, (std::vector<int>{2, 1, 0}));
  EXPECT_EQ(values, (std::vector<double>{-0.75, -0.5, 0.0}));
}

lapack_int ref_syevd(lapack_int n, float *a, float *w) {
  return LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}
lapack_int ref_syevd(lapack_int n, double *a, double *w) {
  return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
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

/// The reference: every eigenpair of @p a, ascending (vectors column-major).
template<typename T>
struct Reference {
  std::vector<T> w;
  std::vector<T> v;
};

template<typename T>
Reference<T> reference(const std::vector<T> &a, int n) {
  Reference<T> ref{std::vector<T>(static_cast<std::size_t>(n)), a};
  EXPECT_EQ(ref_syevd(n, ref.v.data(), ref.w.data()), 0);
  return ref;
}

lapack_int ref_syevr(lapack_int n, float *a, lapack_int il, lapack_int iu, lapack_int *m, float *w,
                     float *z, lapack_int *isuppz) {
  return LAPACKE_ssyevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', n, a, n, 0.0f, 0.0f, il, iu, 0.0f, m, w, z,
                        n, isuppz);
}
lapack_int ref_syevr(lapack_int n, double *a, lapack_int il, lapack_int iu, lapack_int *m,
                     double *w, double *z, lapack_int *isuppz) {
  return LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'L', n, a, n, 0.0, 0.0, il, iu, 0.0, m, w, z, n,
                        isuppz);
}

/// The reference over positions lo..hi (0-based, inclusive) only, ascending.
template<typename T>
Reference<T> reference_window(std::vector<T> a, int n, int lo, int hi) {
  const auto count = static_cast<std::size_t>(hi - lo + 1);
  Reference<T> ref{std::vector<T>(count), std::vector<T>(static_cast<std::size_t>(n) * count)};
  std::vector<lapack_int> isuppz(2 * count);
  lapack_int m = 0;
  EXPECT_EQ(ref_syevr(n, a.data(), lo + 1, hi + 1, &m, ref.w.data(), ref.v.data(), isuppz.data()),
            0);
  ref.w.resize(static_cast<std::size_t>(m));
  return ref;
}

/// The @p nev positions of @p w nearest @p sigma, ascending.
template<typename T>
std::vector<int> nearest(const std::vector<T> &w, T sigma, int nev) {
  std::vector<int> idx(w.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::ranges::stable_sort(idx, [&](int a, int b) {
    return std::abs(w[static_cast<std::size_t>(a)] - sigma) <
           std::abs(w[static_cast<std::size_t>(b)] - sigma);
  });
  idx.resize(static_cast<std::size_t>(nev));
  std::ranges::sort(idx);
  return idx;
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

/// A on the device, the operator's and the driver's workspaces, the outputs.
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
  DeviceBuffer<T> d_w;
  std::unique_ptr<DeviceBuffer<std::byte>> op_work;
  std::unique_ptr<DeviceBuffer<std::byte>> work;
  DenseShiftInvertSlices<T> op_slices;
  LanczosSlices<T> s;

  Rig(const std::vector<T> &a, int n_, int nev_, int ncv_)
      : n(n_), nev(nev_), ncv(ncv_), d_a(a.size(), handle),
        d_x(static_cast<std::size_t>(n_) * static_cast<std::size_t>(nev_), handle),
        d_w(static_cast<std::size_t>(nev_), handle) {
    EXPECT_EQ(wwr::wwrMemcpyAsync(d_a.data(), a.data(), sizeof(T) * a.size(),
                                  wwr::wwrMemcpyHostToDevice, stream),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
    std::size_t bytes = 0;
    EXPECT_TRUE(dense_shift_invert_bufferSize<T>(h.solver, n, &bytes).ok());
    op_work = std::make_unique<DeviceBuffer<std::byte>>(bytes, handle);
    EXPECT_TRUE(
        make_dense_shift_invert_slices<T>(h.solver, n, op_work->data(), &op_slices, nullptr).ok());
    EXPECT_TRUE(lanczos_bufferSize<T>(h.solver, n, nev, ncv, &bytes).ok());
    work = std::make_unique<DeviceBuffer<std::byte>>(bytes, handle);
    EXPECT_TRUE(make_lanczos_slices<T>(h.solver, n, nev, ncv, work->data(), &s, nullptr).ok());
  }

  DenseShiftInvert<T> op(T sigma) {
    return DenseShiftInvert<T>{h.blas, h.solver, Uplo::L, n, d_a.data(), n, sigma, op_slices};
  }
  Status solve(DenseShiftInvert<T> &op, const LanczosOptions<T> &options, LanczosInfo *info) {
    return lanczos_shift_invert_solve<T>(h.blas, h.solver, stream, n, nev, ncv, s, op, d_w.data(),
                                         d_x.data(), info, options);
  }
  std::vector<T> download(const T *src, std::size_t count) {
    std::vector<T> out(count);
    EXPECT_EQ(
        wwr::wwrMemcpyAsync(out.data(), src, sizeof(T) * count, wwr::wwrMemcpyDeviceToHost, stream),
        wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
    return out;
  }
};

/// An evenly spaced spectrum in [-1, 1]: interior gaps h = 2 / (n - 1).
std::vector<double> even_spectrum(int n) {
  std::vector<double> lambda;
  for (int i = 0; i < n; ++i) {
    lambda.push_back(-1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n - 1));
  }
  return lambda;
}

/// Solve for the @p nev pairs nearest @p sigma and check them against @p ref:
/// its pairs lo..lo+nev-1 are the wanted ones, any others their neighbours.
template<typename T>
void check_against(const std::vector<T> &a, int n, int nev, int ncv, T sigma,
                   const LanczosOptions<T> &options, const Reference<T> &ref, int lo) {
  SCOPED_TRACE(::testing::Message()
               << "n=" << n << " nev=" << nev << " ncv=" << ncv << " sigma=" << sigma
               << " sizeof(T)=" << sizeof(T) << " verify=" << options.verify_residuals);
  Rig<T> rig(a, n, nev, ncv);
  auto op = rig.op(sigma);
  ASSERT_TRUE(op.prepare(rig.stream).ok());
  LanczosInfo info;
  const Status st = rig.solve(op, options, &info);
  ASSERT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(info);

  const auto nz = static_cast<std::size_t>(n);
  const auto w = rig.download(rig.d_w.data(), static_cast<std::size_t>(nev));
  const auto x = rig.download(rig.d_x.data(), nz * static_cast<std::size_t>(nev));
  const T tol = factorization_tol<T>(frobenius_norm(a), nz, nz);
  for (std::size_t c = 0; c < w.size(); ++c) {
    SCOPED_TRACE(::testing::Message() << "pair " << c);
    const auto r = static_cast<std::size_t>(lo) + c;
    EXPECT_NEAR(w[c], ref.w[r], tol);
    if (c > 0) {
      EXPECT_LT(w[c - 1], w[c]);
    }
    // ||A x - lambda x||, ||x||, and x^T v_ref (for its sign): in double.
    double res = 0.0;
    double norm = 0.0;
    double dot = 0.0;
    for (std::size_t i = 0; i < nz; ++i) {
      double ax = 0.0;
      for (std::size_t l = 0; l < nz; ++l) {
        ax += static_cast<double>(a[i + l * nz]) * static_cast<double>(x[l + c * nz]);
      }
      const double xi = static_cast<double>(x[i + c * nz]);
      const double e = ax - static_cast<double>(w[c]) * xi;
      res += e * e;
      norm += xi * xi;
      dot += xi * static_cast<double>(ref.v[i + r * nz]);
    }
    EXPECT_LE(std::sqrt(res), static_cast<double>(tol));
    EXPECT_NEAR(std::sqrt(norm), 1.0, static_cast<double>(factorization_tol<T>(T(1), nz, nz)));
    // Davis-Kahan: sin(angle to v_ref) <= ||r|| / gap, gap to the next eigenvalue.
    double gap = std::numeric_limits<double>::infinity();
    if (r > 0) {
      gap = std::min(gap, static_cast<double>(ref.w[r] - ref.w[r - 1]));
    }
    if (r + 1 < ref.w.size()) {
      gap = std::min(gap, static_cast<double>(ref.w[r + 1] - ref.w[r]));
    }
    // ||x - sign(dot) v_ref|| = 2 sin(angle / 2), free of the cancellation
    // sqrt(1 - dot^2) suffers at dot ~ 1, and never below sin(angle).
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    double chord = 0.0;
    for (std::size_t i = 0; i < nz; ++i) {
      const double d =
          static_cast<double>(x[i + c * nz]) - sign * static_cast<double>(ref.v[i + r * nz]);
      chord += d * d;
    }
    EXPECT_LE(std::sqrt(chord), 2.0 * static_cast<double>(tol) / gap);
  }
  // Orthonormal block.
  for (std::size_t c = 0; c < w.size(); ++c) {
    for (std::size_t d = 0; d < c; ++d) {
      double dot = 0.0;
      for (std::size_t i = 0; i < nz; ++i) {
        dot += static_cast<double>(x[i + c * nz]) * static_cast<double>(x[i + d * nz]);
      }
      EXPECT_NEAR(dot, 0.0, static_cast<double>(factorization_tol<T>(T(1), nz, nz)))
          << c << "," << d;
    }
  }
}

/// The whole spectrum by syevd; the wanted pairs are the nev nearest sigma.
template<typename T>
void check_interior(const std::vector<T> &a, int n, int nev, int ncv, T sigma,
                    const LanczosOptions<T> &options) {
  const Reference<T> ref = reference(a, n);
  const std::vector<int> want = nearest(ref.w, sigma, nev);
  ASSERT_EQ(want.back() - want.front(), nev - 1) << "the nearest pairs are contiguous";
  check_against<T>(a, n, nev, ncv, sigma, options, ref, want.front());
}

/// Only an index window by syevr: the positions of the nev entries of
/// @p spectrum (A's, by construction) nearest sigma, widened by one each side
/// for the neighbours that bound the Davis-Kahan gaps.
template<typename T>
void check_window(const std::vector<T> &a, const std::vector<double> &spectrum, int nev, int ncv,
                  T sigma, const LanczosOptions<T> &options) {
  const int n = static_cast<int>(spectrum.size());
  const int first = nearest(spectrum, static_cast<double>(sigma), nev).front();
  const int lo = std::max(first - 1, 0);
  const int hi = std::min(first + nev, n - 1);
  const Reference<T> ref = reference_window(a, n, lo, hi);
  ASSERT_EQ(ref.w.size(), static_cast<std::size_t>(hi - lo + 1));
  // The window is the nearest in A's own spectrum: no neighbour is nearer.
  const auto at = [&](int i) { return std::abs(ref.w[static_cast<std::size_t>(i - lo)] - sigma); };
  const T farthest = std::max(at(first), at(first + nev - 1));
  if (lo < first) {
    EXPECT_GE(at(lo), farthest) << "below the window";
  }
  if (first + nev <= hi) {
    EXPECT_GE(at(hi), farthest) << "above the window";
  }
  check_against<T>(a, n, nev, ncv, sigma, options, ref, first - lo);
}

template<typename T>
LanczosOptions<T> tight_options() {
  LanczosOptions<T> options;
  options.tolerance = T(1000) * test::eps<T>();
  return options;
}

template<typename T>
void check_interior_cases() {
  constexpr int n = 120;
  const double h = 2.0 / static_cast<double>(n - 1);
  const auto a = from_spectrum<T>(even_spectrum(n), 20261008u);
  auto options = tight_options<T>();
  // sigma 0.3 h above an eigenvalue near 0.1: the nev nearest straddle it.
  const T sigma = static_cast<T>(-1.0 + 66.0 * h + 0.3 * h);
  check_interior<T>(a, n, 4, 20, sigma, options);
  check_interior<T>(a, n, 5, 20, sigma, options);
  options.verify_residuals = true;
  check_interior<T>(a, n, 4, 20, sigma, options);
}

TEST(LanczosShiftInvertReferenceTests, NearestSigmaMatchesSyevdDouble) {
  check_interior_cases<double>();
}
TEST(LanczosShiftInvertReferenceTests, NearestSigmaMatchesSyevdFloat) {
  check_interior_cases<float>();
}

// An operator never prepared fails its first apply: a failing Status and
// reason NumericalFailure, not garbage.
TEST(LanczosShiftInvertReferenceTests, UnpreparedOperatorFails) {
  constexpr int n = 24;
  const auto a = from_spectrum<double>(even_spectrum(n), 5u);
  Rig<double> rig(a, n, 2, 8);
  auto op = rig.op(0.01);
  LanczosInfo info;
  const Status st = rig.solve(op, tight_options<double>(), &info);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(static_cast<int>(info.reason), static_cast<int>(LanczosStopReason::NumericalFailure));
}

// ── interior windows and edge cases ──────────────────────────────────────────

/// lambda_i = -1 + 2 (i / (n - 1))^2: crowded at the bottom, sparse at the top.
std::vector<double> graded_spectrum(int n) {
  std::vector<double> lambda;
  for (int i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(n - 1);
    lambda.push_back(-1.0 + 2.0 * t * t);
  }
  return lambda;
}

// The pairs around a position in the crowded bottom, the middle and the sparse
// top, sigma 0.3 of a gap above it, against syevr over just that window.
template<typename T>
void check_index_windows() {
  constexpr int n = 150;
  const auto spectrum = graded_spectrum(n);
  const auto a = from_spectrum<T>(spectrum, 318u);
  const auto options = tight_options<T>();
  for (const int p : {12, n / 2, 9 * n / 10}) {
    const auto pz = static_cast<std::size_t>(p);
    const T sigma = static_cast<T>(spectrum[pz] + 0.3 * (spectrum[pz + 1] - spectrum[pz]));
    check_window<T>(a, spectrum, 3, 16, sigma, options);
    check_window<T>(a, spectrum, 5, 24, sigma, options);
  }
}

TEST(LanczosShiftInvertInteriorTests, IndexWindowsMatchSyevrDouble) {
  check_index_windows<double>();
}
TEST(LanczosShiftInvertInteriorTests, IndexWindowsMatchSyevrFloat) {
  check_index_windows<float>();
}

// sigma midway between two eigenvalues 1e-3 h apart: theta = +-2 / gap, equal
// magnitudes of opposite sign, both wanted. nev = 2 only: a farther pair would
// be judged against that theta, so resolved no better than tol * theta_max / theta.
template<typename T>
void check_close_pair() {
  constexpr int n = 120;
  const double h = 2.0 / static_cast<double>(n - 1);
  auto spectrum = even_spectrum(n);
  spectrum[67] = spectrum[66] + 1e-3 * h;
  const auto a = from_spectrum<T>(spectrum, 11u);
  const T sigma = static_cast<T>(0.5 * (spectrum[66] + spectrum[67]));
  check_window<T>(a, spectrum, 2, 20, sigma, tight_options<T>());
}

TEST(LanczosShiftInvertInteriorTests, SigmaBetweenClosePairDouble) {
  check_close_pair<double>();
}
TEST(LanczosShiftInvertInteriorTests, SigmaBetweenClosePairFloat) {
  check_close_pair<float>();
}

// sigma sqrt(eps) h from an eigenvalue, either side: A - sigma I is
// ill-conditioned (theta ~ 1 / (sqrt(eps) h)) but nonsingular, and the nearest
// pair still converges. nev = 1, for the reason check_close_pair gives.
template<typename T>
void check_near_eigenvalue() {
  constexpr int n = 120;
  const double h = 2.0 / static_cast<double>(n - 1);
  const auto spectrum = even_spectrum(n);
  const auto a = from_spectrum<T>(spectrum, 23u);
  const double delta = std::sqrt(static_cast<double>(test::eps<T>())) * h;
  check_window<T>(a, spectrum, 1, 12, static_cast<T>(spectrum[40] + delta), tight_options<T>());
  check_window<T>(a, spectrum, 1, 12, static_cast<T>(spectrum[40] - delta), tight_options<T>());
}

TEST(LanczosShiftInvertInteriorTests, SigmaNearEigenvalueDouble) {
  check_near_eigenvalue<double>();
}
TEST(LanczosShiftInvertInteriorTests, SigmaNearEigenvalueFloat) {
  check_near_eigenvalue<float>();
}

// sigma below or above the spectrum: every theta has one sign and the solve is
// an extreme-end one, the nev smallest or largest pairs.
template<typename T>
void check_outside_spectrum() {
  constexpr int n = 120;
  const auto spectrum = even_spectrum(n);
  const auto a = from_spectrum<T>(spectrum, 29u);
  check_window<T>(a, spectrum, 4, 20, T(-1.5), tight_options<T>());
  check_window<T>(a, spectrum, 4, 20, T(1.5), tight_options<T>());
}

TEST(LanczosShiftInvertInteriorTests, SigmaOutsideSpectrumDouble) {
  check_outside_spectrum<double>();
}
TEST(LanczosShiftInvertInteriorTests, SigmaOutsideSpectrumFloat) {
  check_outside_spectrum<float>();
}

/// 2 x 2 blocks [[3i, 1], [1, 3i]] (eigenvalues 3i -+ 1, i = 1..n/2) under a
/// random symmetric permutation: integer entries, so the elimination of
/// A - (3i + 1) I cancels to an exact zero pivot in either precision.
template<typename T>
std::vector<T> integer_blocks(int n, std::uint32_t seed) {
  const auto nz = static_cast<std::size_t>(n);
  std::vector<T> b(nz * nz, T(0));
  for (std::size_t k = 0; k + 1 < nz; k += 2) {
    const auto d = static_cast<T>(3 * (k / 2 + 1));
    b[k + k * nz] = d;
    b[(k + 1) + (k + 1) * nz] = d;
    b[(k + 1) + k * nz] = T(1);
    b[k + (k + 1) * nz] = T(1);
  }
  std::vector<std::size_t> p(nz);
  std::iota(p.begin(), p.end(), std::size_t{0});
  std::mt19937 gen(seed);
  std::ranges::shuffle(p, gen);
  std::vector<T> a(nz * nz);
  for (std::size_t j = 0; j < nz; ++j) {
    for (std::size_t i = 0; i < nz; ++i) {
      a[i + j * nz] = b[p[i] + p[j] * nz];
    }
  }
  return a;
}

std::vector<double> integer_blocks_spectrum(int n) {
  std::vector<double> lambda;
  for (int i = 1; i <= n / 2; ++i) {
    lambda.push_back(3.0 * i - 1.0);
    lambda.push_back(3.0 * i + 1.0);
  }
  return lambda;
}

// sigma exactly on an eigenvalue: prepare fails, and the solve on the unfactored
// operator returns a failing Status with reason NumericalFailure, leaving the
// output untouched. A quarter off it, the same A solves.
template<typename T>
void check_singular_shift() {
  constexpr int n = 24;
  constexpr int nev = 2;
  const auto a = integer_blocks<T>(n, 31u);
  const T sigma = T(16); // 3 * 5 + 1
  Rig<T> rig(a, n, nev, 10);
  auto op = rig.op(sigma);
  EXPECT_FALSE(op.prepare(rig.stream).ok());

  const std::vector<T> sentinel(nev, T(-7));
  ASSERT_EQ(wwr::wwrMemcpyAsync(rig.d_w.data(), sentinel.data(), sizeof(T) * sentinel.size(),
                                wwr::wwrMemcpyHostToDevice, rig.stream),
            wwr::wwrSuccess);
  LanczosInfo info;
  const Status st = rig.solve(op, tight_options<T>(), &info);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(static_cast<int>(info.reason), static_cast<int>(LanczosStopReason::NumericalFailure));
  EXPECT_EQ(rig.download(rig.d_w.data(), sentinel.size()), sentinel);

  check_window<T>(a, integer_blocks_spectrum(n), nev, 10, sigma + T(0.25), tight_options<T>());
}

TEST(LanczosShiftInvertInteriorTests, SingularShiftFailsDouble) {
  check_singular_shift<double>();
}
TEST(LanczosShiftInvertInteriorTests, SingularShiftFailsFloat) {
  check_singular_shift<float>();
}

} // namespace
} // namespace calaman
