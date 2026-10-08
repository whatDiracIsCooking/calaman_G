// Suite for calaman.shifted_cocg. The operator is a dense random symmetric A on
// the device, applied through a symm-backed linear_operator; the oracle is
// LAPACKE_zgesv on z_e I - A, in double, for each shift. Checked:
//
//   * FEAST-like shifts (a half-circle contour inside the spectrum) plus nodes
//     1e-3 above the real axis: every X_e has true relative residual within
//     kSlack * tol, and forward error within kSlack * kappa_e * tol of zgesv's,
//     kappa_e = max |z_e - lambda| / min |z_e - lambda| (the 2-norm condition
//     number of the normal matrix z_e I - A, eigenvalues from LAPACKE_dsyevd);
//   * per-shift iteration counts: within [1, iterations], the slowest rounded
//     up to the check interval equal to iterations, a near-axis node slower
//     than a contour node, one apply a step;
//   * the check interval: one host sync per check_interval steps, and the same
//     X, per-shift counts and residuals at every interval;
//   * max_iterations exhaustion (success, reason MaxIterations), a zero column
//     of B (X's column zero), an invariant subspace (exact after one step);
//   * the argument checks and the workspace sizing, which need no card.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

#include "shared/expect_converged.h"

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.wrappers.blas;
import wwr.extension.memory_buffer;
import calaman.shifted_cocg;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

constexpr int kInvalidValue = static_cast<int>(wwr::WWRBLAS_STATUS_INVALID_VALUE);

/// The gap allowed between the residual estimate the solver stops on and what
/// the host measures, for rounding in the recurrences and in the measurement.
constexpr double kSlack = 10.0;

// ── operators ────────────────────────────────────────────────────────────────

/// Y = A X for a dense symmetric n x n device matrix (lower triangle read).
template<typename T>
struct DenseOperator {
  wwr::wwrblasHandle_t blas{};
  int n = 0;
  const T *d_a = nullptr;
  int applies = 0;

  Status apply(wwr::wwrStream_t, const int k, const T *X, T *Y) {
    ++applies;
    const T one{1};
    const T zero{0};
    return wwr::symm<T, int>(blas, wwr::WWRBLAS_SIDE_LEFT, wwr::WWRBLAS_FILL_MODE_LOWER, n, k, &one,
                             d_a, n, X, n, &zero, Y, n);
  }
};
static_assert(linear_operator<DenseOperator<double>, double>);
static_assert(linear_operator<DenseOperator<float>, float>);

/// An operator the argument checks must reject before ever applying.
struct NeverApplied {
  Status apply(wwr::wwrStream_t, int, const double *, double *) {
    ADD_FAILURE() << "operator applied";
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

// ── host side ────────────────────────────────────────────────────────────────

/// Symmetric, entries uniform in [-1, 1] / sqrt(n): spectrum roughly [-1.15, 1.15].
std::vector<double> random_symmetric(const int n, const std::uint32_t seed) {
  const auto nz = static_cast<std::size_t>(n);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(n));
  std::vector<double> a(nz * nz);
  for (std::size_t j = 0; j < nz; ++j) {
    for (std::size_t i = j; i < nz; ++i) {
      const double v = dist(gen) * scale;
      a[i + j * nz] = v;
      a[j + i * nz] = v;
    }
  }
  return a;
}

std::vector<double> random_block(const std::size_t count, const std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> b(count);
  for (double &x : b) {
    x = dist(gen);
  }
  return b;
}

std::vector<double> eigenvalues(const std::vector<double> &a, const int n) {
  std::vector<double> copy = a;
  std::vector<double> w(static_cast<std::size_t>(n));
  EXPECT_EQ(LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', n, copy.data(), n, w.data()), 0);
  return w;
}

/// Ne nodes on the upper half of the circle around [emin, emax], at angles
/// pi (e + 1/2) / Ne: the shape of a FEAST contour.
std::vector<std::complex<double>> contour(const double emin, const double emax, const int ne) {
  const double c = 0.5 * (emin + emax);
  const double r = 0.5 * (emax - emin);
  std::vector<std::complex<double>> z;
  for (int e = 0; e < ne; ++e) {
    const double theta = std::numbers::pi * (e + 0.5) / ne;
    z.emplace_back(c + r * std::cos(theta), r * std::sin(theta));
  }
  return z;
}

/// (z I - A)^{-1} B by LAPACKE_zgesv; n x k, column-major.
std::vector<std::complex<double>> zgesv_solve(const std::vector<double> &a,
                                              const std::vector<double> &b, const int n,
                                              const int k, const std::complex<double> z) {
  const auto nz = static_cast<std::size_t>(n);
  std::vector<std::complex<double>> m(nz * nz);
  for (std::size_t idx = 0; idx < nz * nz; ++idx) {
    m[idx] = -a[idx];
  }
  for (std::size_t i = 0; i < nz; ++i) {
    m[i + i * nz] += z;
  }
  std::vector<std::complex<double>> x(b.begin(), b.end());
  std::vector<lapack_int> ipiv(nz);
  EXPECT_EQ(LAPACKE_zgesv(LAPACK_COL_MAJOR, n, k,
                          reinterpret_cast<lapack_complex_double *>(m.data()), n, ipiv.data(),
                          reinterpret_cast<lapack_complex_double *>(x.data()), n),
            0);
  return x;
}

/// Largest over the columns of ||b_j - (z I - A) x_j||_2 / ||b_j||_2 (zero columns skipped).
double true_residual(const std::vector<double> &a, const std::vector<double> &b,
                     const std::vector<std::complex<double>> &x, const int n, const int k,
                     const std::complex<double> z) {
  const auto nz = static_cast<std::size_t>(n);
  double worst = 0.0;
  for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < nz; ++i) {
      std::complex<double> ax = 0.0;
      for (std::size_t l = 0; l < nz; ++l) {
        ax += a[i + l * nz] * x[l + j * nz];
      }
      const std::complex<double> r = b[i + j * nz] - (z * x[i + j * nz] - ax);
      rr += std::norm(r);
      bb += b[i + j * nz] * b[i + j * nz];
    }
    if (bb > 0.0) {
      worst = std::max(worst, std::sqrt(rr / bb));
    }
  }
  return worst;
}

double relative_error(const std::vector<std::complex<double>> &x,
                      const std::vector<std::complex<double>> &ref) {
  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    num += std::norm(x[i] - ref[i]);
    den += std::norm(ref[i]);
  }
  return std::sqrt(num / den);
}

double kappa(const std::vector<double> &lambda, const std::complex<double> z) {
  double lo = std::numeric_limits<double>::infinity();
  double hi = 0.0;
  for (const double l : lambda) {
    const double d = std::abs(z - l);
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  return hi / lo;
}

template<typename T>
std::vector<T> cast(const std::vector<double> &v) {
  return {v.begin(), v.end()};
}

template<typename T>
std::vector<double> widen(const std::vector<T> &v) {
  return {v.begin(), v.end()};
}

// ── device rig ───────────────────────────────────────────────────────────────

struct BlasHandle {
  wwr::wwrblasHandle_t h{};
  explicit BlasHandle(wwr::wwrStream_t stream) {
    EXPECT_EQ(wwr::wwrblasCreate(&h), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrblasSetStream(h, stream), wwr::WWRBLAS_STATUS_SUCCESS);
  }
  ~BlasHandle() { wwr::wwrblasDestroy(h); }
  BlasHandle(const BlasHandle &) = delete;
  BlasHandle &operator=(const BlasHandle &) = delete;
};

/// A, B and the outputs on the device, A and B given in double and rounded to T.
template<typename T>
struct Rig {
  std::shared_ptr<DeviceHandle> handle = shared_device();
  wwr::wwrStream_t stream = handle->stream().get();
  BlasHandle blas{stream};
  int n;
  int k;
  int ne;
  std::vector<double> a; ///< A as the device holds it, widened back to double
  std::vector<double> b; ///< likewise B
  DeviceBuffer<T> d_a;
  DeviceBuffer<T> d_b;
  DeviceBuffer<T> d_xr;
  DeviceBuffer<T> d_xi;
  std::unique_ptr<DeviceBuffer<std::byte>> work;
  ShiftedCocgSlices<T> s;
  DenseOperator<T> op;

  Rig(const std::vector<double> &a_in, const std::vector<double> &b_in, int n_, int k_, int ne_)
      : n(n_), k(k_), ne(ne_), a(widen(cast<T>(a_in))), b(widen(cast<T>(b_in))),
        d_a(a_in.size(), handle), d_b(b_in.size(), handle), d_xr(b_in.size() * ne_, handle),
        d_xi(b_in.size() * ne_, handle) {
    upload(d_a.data(), cast<T>(a_in));
    upload(d_b.data(), cast<T>(b_in));
    std::size_t lwork = 0;
    EXPECT_TRUE(shifted_cocg_bufferSize<T>(n, k, ne, &lwork).ok());
    work = std::make_unique<DeviceBuffer<std::byte>>(lwork, handle);
    EXPECT_TRUE(make_shifted_cocg_slices<T>(n, k, ne, work->data(), &s, nullptr).ok());
    op = DenseOperator<T>{blas.h, n, d_a.data()};
  }

  void upload(T *dst, const std::vector<T> &src) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(dst, src.data(), sizeof(T) * src.size(),
                                  wwr::wwrMemcpyHostToDevice, stream),
              wwr::wwrSuccess);
    ASSERT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  }
  std::vector<T> download(const T *src, std::size_t count) {
    std::vector<T> out(count);
    EXPECT_EQ(
        wwr::wwrMemcpyAsync(out.data(), src, sizeof(T) * count, wwr::wwrMemcpyDeviceToHost, stream),
        wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
    return out;
  }

  Status solve(const std::vector<std::complex<double>> &z, const ShiftedCocgOptions<T> &options,
               ShiftedCocgInfo<T> *info) {
    std::vector<T> zr;
    std::vector<T> zi;
    for (const auto &c : z) {
      zr.push_back(static_cast<T>(c.real()));
      zi.push_back(static_cast<T>(c.imag()));
    }
    return shifted_cocg<T>(stream, op, n, k, zr, zi, d_b.data(), d_xr.data(), d_xi.data(), s, info,
                           options);
  }

  /// X_e as complex double, n x k.
  std::vector<std::complex<double>> x(const int e) {
    const std::size_t block = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
    const std::vector<T> re = download(d_xr.data() + static_cast<std::size_t>(e) * block, block);
    const std::vector<T> im = download(d_xi.data() + static_cast<std::size_t>(e) * block, block);
    std::vector<std::complex<double>> out(block);
    for (std::size_t i = 0; i < block; ++i) {
      out[i] = {static_cast<double>(re[i]), static_cast<double>(im[i])};
    }
    return out;
  }
};

/// The steps a converged solve runs: @p slowest rounded up to a check.
int rounded_to_check(const int slowest, const int interval) {
  return (slowest + interval - 1) / interval * interval;
}

/// Solve on a random symmetric A for @p z, then check every X_e against zgesv.
/// Returns the info for the caller's iteration-count assertions.
template<typename T>
ShiftedCocgInfo<T>
solve_and_check(const int n, const int k, const std::vector<std::complex<double>> &z, const T tol,
                const int check_interval = ShiftedCocgOptions<T>{}.check_interval) {
  const int ne = static_cast<int>(z.size());
  Rig<T> rig(random_symmetric(n, 17U),
             random_block(static_cast<std::size_t>(n) * static_cast<std::size_t>(k), 29U), n, k,
             ne);
  const std::vector<double> lambda = eigenvalues(rig.a, n);

  ShiftedCocgOptions<T> options;
  options.tolerance = tol;
  options.max_iterations = 20 * n;
  options.check_interval = check_interval;
  ShiftedCocgInfo<T> info;
  EXPECT_TRUE(rig.solve(z, options, &info).ok());
  EXPECT_CONVERGED(info);
  EXPECT_EQ(rig.op.applies, info.iterations);
  EXPECT_EQ(info.shift_iterations.size(), z.size());
  EXPECT_EQ(info.shift_residual.size(), z.size());

  int slowest = 0;
  for (int e = 0; e < ne; ++e) {
    const auto ez = static_cast<std::size_t>(e);
    SCOPED_TRACE(::testing::Message() << "shift " << e << " = " << z[ez]);
    EXPECT_GE(info.shift_iterations[ez], 1);
    EXPECT_LE(info.shift_iterations[ez], info.iterations);
    EXPECT_LE(info.shift_residual[ez], tol);
    slowest = std::max(slowest, info.shift_iterations[ez]);

    const std::vector<std::complex<double>> x = rig.x(e);
    const std::vector<std::complex<double>> ref = zgesv_solve(rig.a, rig.b, n, k, z[ez]);
    const double bound = kSlack * static_cast<double>(tol);
    EXPECT_LE(true_residual(rig.a, rig.b, x, n, k, z[ez]), bound);
    EXPECT_LE(relative_error(x, ref), kappa(lambda, z[ez]) * bound);
  }
  EXPECT_EQ(rounded_to_check(slowest, check_interval), info.iterations);
  EXPECT_EQ(info.convergence_checks, 1 + info.iterations / check_interval);
  return info;
}

// ── argument checks and sizing: no card ─────────────────────────────────────

TEST(ShiftedCocgArgCheckTests, RejectsBadArguments) {
  ShiftedCocgSlices<double> s;
  s.v = reinterpret_cast<double *>(std::uintptr_t{256}); // never dereferenced
  s.n = 4;
  s.k_max = 2;
  s.shifts_max = 2;
  NeverApplied op;
  ShiftedCocgInfo<double> info;
  double dummy = 0.0;
  const std::vector<double> zr = {0.0, 1.0};
  const std::vector<double> zi = {0.5, 0.5};
  const auto call = [&](int n, int k, std::span<const double> r, std::span<const double> i,
                        ShiftedCocgInfo<double> *inf, const ShiftedCocgOptions<double> &o) {
    return shifted_cocg<double>(nullptr, op, n, k, r, i, &dummy, &dummy, &dummy, s, inf, o).code;
  };
  const ShiftedCocgOptions<double> ok_options;
  EXPECT_EQ(call(4, 2, zr, zi, nullptr, ok_options), kInvalidValue);
  EXPECT_EQ(call(4, 0, zr, zi, &info, ok_options), kInvalidValue);
  EXPECT_EQ(call(4, 3, zr, zi, &info, ok_options), kInvalidValue); // k > k_max
  EXPECT_EQ(call(5, 2, zr, zi, &info, ok_options), kInvalidValue); // n differs from the carve
  EXPECT_EQ(call(4, 2, {}, {}, &info, ok_options), kInvalidValue);
  EXPECT_EQ(call(4, 2, zr, std::span<const double>(zi).first(1), &info, ok_options), kInvalidValue);
  const std::vector<double> three = {0.0, 1.0, 2.0};
  EXPECT_EQ(call(4, 2, three, three, &info, ok_options), kInvalidValue); // ne > shifts_max
  const std::vector<double> real_axis = {0.5, 0.0};
  EXPECT_EQ(call(4, 2, zr, real_axis, &info, ok_options), kInvalidValue);
  const std::vector<double> lower = {0.5, -0.5};
  EXPECT_EQ(call(4, 2, zr, lower, &info, ok_options), kInvalidValue);
  ShiftedCocgOptions<double> bad = ok_options;
  bad.tolerance = -1.0;
  EXPECT_EQ(call(4, 2, zr, zi, &info, bad), kInvalidValue);
  bad = ok_options;
  bad.max_iterations = -1;
  EXPECT_EQ(call(4, 2, zr, zi, &info, bad), kInvalidValue);
  bad = ok_options;
  bad.check_interval = 0;
  EXPECT_EQ(call(4, 2, zr, zi, &info, bad), kInvalidValue);
  const ShiftedCocgSlices<double> uncarved;
  EXPECT_EQ(shifted_cocg<double>(nullptr, op, 4, 2, zr, zi, &dummy, &dummy, &dummy, uncarved, &info,
                                 ok_options)
                .code,
            kInvalidValue);
}

TEST(ShiftedCocgArgCheckTests, BufferSize) {
  std::size_t small = 0;
  std::size_t wide = 0;
  EXPECT_TRUE(shifted_cocg_bufferSize<double>(100, 2, 4, &small).ok());
  EXPECT_TRUE(shifted_cocg_bufferSize<double>(100, 2, 8, &wide).ok());
  // P alone is 2 x ne blocks of n x k.
  EXPECT_GE(small, 2U * 4U * 100U * 2U * sizeof(double));
  EXPECT_GT(wide, small);
  std::size_t bytes = 0;
  EXPECT_EQ(shifted_cocg_bufferSize<double>(0, 2, 4, &bytes).code, kInvalidValue);
  EXPECT_EQ(shifted_cocg_bufferSize<double>(100, 0, 4, &bytes).code, kInvalidValue);
  EXPECT_EQ(shifted_cocg_bufferSize<double>(100, 2, 0, &bytes).code, kInvalidValue);
  EXPECT_EQ(shifted_cocg_bufferSize<double>(100, 2, 4, nullptr).code, kInvalidValue);
}

// ── against zgesv ────────────────────────────────────────────────────────────

TEST(ShiftedCocgReferenceTests, FeastContourAndNearAxisDouble) {
  // Eight contour nodes around [-0.3, 0.3], then two 1e-3 above the axis, one
  // in that interval and one in the bulk of the spectrum.
  std::vector<std::complex<double>> z = contour(-0.3, 0.3, 8);
  z.emplace_back(0.05, 1e-3);
  z.emplace_back(-0.7, 1e-3);
  const ShiftedCocgInfo<double> info = solve_and_check<double>(200, 3, z, 1e-10);
  // A node near the axis is worse conditioned, so slower, than the top of the circle.
  EXPECT_GT(info.shift_iterations[8], info.shift_iterations[4]);
  EXPECT_GT(info.shift_iterations[9], info.shift_iterations[4]);
}

TEST(ShiftedCocgReferenceTests, FeastContourFloat) {
  solve_and_check<float>(128, 2, contour(-0.4, 0.2, 8), 1e-4F);
}

TEST(ShiftedCocgReferenceTests, SingleColumnDouble) {
  solve_and_check<double>(64, 1, contour(0.1, 0.6, 4), 1e-11);
}

TEST(ShiftedCocgReferenceTests, CheckEveryStepDouble) {
  const ShiftedCocgInfo<double> info =
      solve_and_check<double>(120, 2, contour(-0.3, 0.3, 6), 1e-10, 1);
  EXPECT_EQ(info.convergence_checks, 1 + info.iterations);
}

TEST(ShiftedCocgReferenceTests, CheckIntervalChangesOnlyTheStepCount) {
  // A converged pair freezes, so X, the per-shift counts and the residuals are
  // the same at every interval; only the steps (and the syncs) differ.
  const int n = 150;
  const int k = 2;
  const std::vector<std::complex<double>> z = contour(-0.25, 0.25, 6);
  Rig<double> rig(random_symmetric(n, 41U), random_block(static_cast<std::size_t>(n) * k, 43U), n,
                  k, static_cast<int>(z.size()));
  ShiftedCocgOptions<double> options;
  options.tolerance = 1e-10;
  options.max_iterations = 20 * n;
  options.check_interval = 1;
  ShiftedCocgInfo<double> every;
  ASSERT_TRUE(rig.solve(z, options, &every).ok());
  EXPECT_CONVERGED(every);
  std::vector<std::vector<std::complex<double>>> x_every;
  for (int e = 0; e < static_cast<int>(z.size()); ++e) {
    x_every.push_back(rig.x(e));
  }
  for (const int interval : {3, 8, 64}) {
    SCOPED_TRACE(::testing::Message() << "check_interval " << interval);
    options.check_interval = interval;
    ShiftedCocgInfo<double> info;
    ASSERT_TRUE(rig.solve(z, options, &info).ok());
    EXPECT_CONVERGED(info);
    EXPECT_EQ(info.iterations, rounded_to_check(every.iterations, interval));
    EXPECT_EQ(info.convergence_checks, 1 + info.iterations / interval);
    EXPECT_EQ(info.shift_iterations, every.shift_iterations);
    EXPECT_EQ(info.shift_residual, every.shift_residual);
    for (int e = 0; e < static_cast<int>(z.size()); ++e) {
      EXPECT_EQ(rig.x(e), x_every[static_cast<std::size_t>(e)]) << "shift " << e;
    }
  }
}

TEST(ShiftedCocgReferenceTests, MaxIterationsIsAnOutcome) {
  const int n = 100;
  const int k = 2;
  Rig<double> rig(random_symmetric(n, 5U), random_block(static_cast<std::size_t>(n) * k, 7U), n, k,
                  3);
  ShiftedCocgOptions<double> options;
  options.tolerance = 1e-12;
  options.max_iterations = 5;
  options.check_interval = 2; // checks at steps 2, 4 and the last, 5
  ShiftedCocgInfo<double> info;
  ASSERT_TRUE(rig.solve(contour(-0.2, 0.2, 3), options, &info).ok());
  EXPECT_EQ(info.reason, ShiftedCocgStopReason::MaxIterations);
  EXPECT_FALSE(converged(info));
  EXPECT_EQ(info.iterations, 5);
  EXPECT_EQ(rig.op.applies, 5);
  EXPECT_EQ(info.convergence_checks, 4);
  for (std::size_t e = 0; e < 3; ++e) {
    EXPECT_EQ(info.shift_iterations[e], 5);
    EXPECT_GT(info.shift_residual[e], 1e-12);
  }
}

TEST(ShiftedCocgReferenceTests, ZeroColumnStaysZero) {
  const int n = 80;
  const int k = 3;
  std::vector<double> b = random_block(static_cast<std::size_t>(n) * k, 11U);
  std::fill_n(b.begin() + n, n, 0.0); // column 1
  Rig<double> rig(random_symmetric(n, 13U), b, n, k, 2);
  const std::vector<std::complex<double>> z = {{0.1, 0.2}, {-0.3, 0.05}};
  ShiftedCocgOptions<double> options;
  options.tolerance = 1e-10;
  ShiftedCocgInfo<double> info;
  ASSERT_TRUE(rig.solve(z, options, &info).ok());
  EXPECT_CONVERGED(info);
  for (int e = 0; e < 2; ++e) {
    const std::vector<std::complex<double>> x = rig.x(e);
    for (int i = 0; i < n; ++i) {
      EXPECT_EQ(x[static_cast<std::size_t>(n + i)], std::complex<double>(0.0, 0.0));
    }
    EXPECT_LE(true_residual(rig.a, rig.b, x, n, k, z[static_cast<std::size_t>(e)]),
              kSlack * options.tolerance);
  }
}

TEST(ShiftedCocgReferenceTests, AllZeroRightHandSideTakesNoStep) {
  const int n = 16;
  Rig<double> rig(random_symmetric(n, 3U), std::vector<double>(static_cast<std::size_t>(n), 0.0), n,
                  1, 1);
  ShiftedCocgInfo<double> info;
  ASSERT_TRUE(rig.solve(std::vector<std::complex<double>>{{0.0, 1.0}}, {}, &info).ok());
  EXPECT_CONVERGED(info);
  EXPECT_EQ(info.iterations, 0);
  EXPECT_EQ(rig.op.applies, 0);
  EXPECT_EQ(info.shift_iterations[0], 0);
  EXPECT_EQ(info.convergence_checks, 1);
}

TEST(ShiftedCocgReferenceTests, InvariantSubspaceIsExactInOneStep) {
  // A diagonal and b = e_0: K(A, b) = span{e_0}, so beta_2 = 0 and step 1 is
  // exact; the steps after it, to the first check, leave X alone.
  const int n = 32;
  const auto nz = static_cast<std::size_t>(n);
  std::vector<double> a(nz * nz, 0.0);
  for (std::size_t i = 0; i < nz; ++i) {
    a[i + i * nz] = 0.1 * static_cast<double>(i) - 1.0;
  }
  std::vector<double> b(nz, 0.0);
  b[0] = 2.0;
  Rig<double> rig(a, b, n, 1, 2);
  const std::vector<std::complex<double>> z = {{-0.5, 0.25}, {0.5, 1e-3}};
  ShiftedCocgOptions<double> options;
  options.tolerance = 0.0;
  ShiftedCocgInfo<double> info;
  ASSERT_TRUE(rig.solve(z, options, &info).ok());
  EXPECT_CONVERGED(info);
  EXPECT_EQ(info.iterations, options.check_interval);
  for (int e = 0; e < 2; ++e) {
    const auto ez = static_cast<std::size_t>(e);
    EXPECT_EQ(info.shift_iterations[ez], 1);
    const std::vector<std::complex<double>> x = rig.x(e);
    const std::complex<double> expected = 2.0 / (z[ez] - a[0]);
    EXPECT_LE(std::abs(x[0] - expected),
              8.0 * std::numeric_limits<double>::epsilon() * std::abs(expected));
    for (std::size_t i = 1; i < nz; ++i) {
      EXPECT_EQ(x[i], std::complex<double>(0.0, 0.0));
    }
  }
}

} // namespace
} // namespace calaman
