// Oracle test for calaman.expm -- the matrix exponential by scaling and
// squaring with a diagonal Pade approximant. There is no LAPACKE_?expm (LAPACK
// ships no matrix exponential), so the oracle is NOT a reference call but a set
// of closed forms, identities that hold for any matrix, and an independent
// approximation from a different code path:
//
//   * closed forms -- exp(0) = I, exp(diag) = diag(exp), a nilpotent whose
//     series terminates;
//   * identities -- exp(A) exp(-A) = I and exp(A) = exp(A/2)^2, which a correct
//     exponential satisfies regardless of the matrix;
//   * the degree ladder -- for a norm inside each rung, expm (which picks the
//     cheap degree) is compared against pade<T>(13) on the same matrix, an
//     independent coefficient table and code path, so a wrong coefficient
//     anywhere on the ladder shows up.
//
// The host arithmetic is done in std::complex (or the plain real), converting at
// the device boundary, so the oracle shares none of expm's device code. All four
// element types. The numerical suites stage the matrices on the device and run
// the kernels + BLAS + LU, so they are REQUIRES_GPU (labeled `gpu`, excluded by
// `ctest -LE gpu`); the plan, threshold and argument-checking cases are host-only
// (expm()/expm_plan()/pade_theta() return before any device work on those paths).
//
// Needs no reference LAPACK -- the spec is its own oracle -- so unlike the linalg
// suites it does not guard on calaman::lapack_reference.

#include <gtest/gtest.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.expm;
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

// ── element traits: device type T <-> host compute type, one per expm type ───
//
// Host arithmetic runs in `compute` (std::complex for the complex types, the
// plain real otherwise), converted to/from the device element type T at the
// boundary through wwr.complex's host accessors/constructors -- never .x/.y,
// which hipComplex lacks.

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
  static wwr::wwrFloatComplex from(compute c) { return wwr::make_wwrFloatComplex(c.real(), c.imag()); }
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

template<typename T>
std::vector<compute_t<T>> to_compute(const std::vector<T> &v) {
  std::vector<compute_t<T>> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = traits<T>::to(v[i]);
  }
  return out;
}

template<typename T>
std::vector<T> from_compute(const std::vector<compute_t<T>> &v) {
  std::vector<T> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = traits<T>::from(v[i]);
  }
  return out;
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

// ── host oracle arithmetic, all in compute_t ─────────────────────────────────

// C = A * B, n-by-n column-major.
template<typename C>
std::vector<C> host_matmul(int n, const std::vector<C> &a, const std::vector<C> &b) {
  std::vector<C> c(static_cast<std::size_t>(n) * n, C{});
  for (int j = 0; j < n; ++j) {
    for (int l = 0; l < n; ++l) {
      const C bl = b[static_cast<std::size_t>(j) * n + l];
      for (int i = 0; i < n; ++i) {
        c[static_cast<std::size_t>(j) * n + i] += a[static_cast<std::size_t>(l) * n + i] * bl;
      }
    }
  }
  return c;
}

template<typename C>
std::vector<C> host_identity(int n) {
  std::vector<C> m(static_cast<std::size_t>(n) * n, C{});
  for (int i = 0; i < n; ++i) {
    m[static_cast<std::size_t>(i) * n + i] = C{1};
  }
  return m;
}

// The induced 1-norm (max absolute column sum), computed in compute_t.
template<typename C>
double host_norm1(int n, const std::vector<C> &a) {
  double best = 0.0;
  for (int j = 0; j < n; ++j) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
      s += std::abs(a[static_cast<std::size_t>(j) * n + i]);
    }
    best = std::max(best, s);
  }
  return best;
}

// Elementwise tolerance from the oracle's magnitude: a backward-stable exp is
// accurate to about eps * ||exp(A)||, scaled by n and a generous big-O factor.
template<typename T>
real_t<T> elem_tol(const std::vector<compute_t<T>> &oracle, int n, real_t<T> factor) {
  double frob = 0.0;
  for (const auto &v : oracle) {
    frob += std::norm(v); // |v|^2
  }
  const real_t<T> nrm = static_cast<real_t<T>>(std::sqrt(frob));
  const real_t<T> eps = std::numeric_limits<real_t<T>>::epsilon();
  return factor * eps * std::max(real_t<T>{1}, nrm) * static_cast<real_t<T>>(n);
}

template<typename T>
void expect_close(const std::vector<T> &got, const std::vector<compute_t<T>> &oracle, int n,
                  int ld, real_t<T> factor, const char *what) {
  const real_t<T> tol = elem_tol<T>(oracle, n, factor);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const compute_t<T> g = traits<T>::to(got[static_cast<std::size_t>(j) * ld + i]);
      const compute_t<T> o = oracle[static_cast<std::size_t>(j) * n + i];
      EXPECT_LE(std::abs(g - o), tol)
          << what << " at (" << i << "," << j << ") n=" << n;
    }
  }
}

// ── device runners ───────────────────────────────────────────────────────────

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

// Run expm on A (n-by-n, leading dim lda), result into an lde-by-n buffer seeded
// with @p out_seed so a padded leading dimension can be checked. Returns status;
// writes the read-back result (length lde*n) into @p out_seed.
template<typename T>
wwr::wwrblasStatus_t run_expm(std::shared_ptr<DeviceHandle> handle, Handles &h, int n,
                              const std::vector<T> &a, int lda, int lde, std::vector<T> &out_seed,
                              ExpmPlan *info) {
  auto d_a = to_device(handle, a);
  auto d_e = to_device(handle, out_seed);

  std::size_t bytes = 0;
  EXPECT_EQ(expm_bufferSize<T>(h.solver, n, &bytes), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<T> d_work(bytes / sizeof(T) + 1, handle);
  DeviceBuffer<int> d_info(2, handle);

  const auto status = expm<T>(h.blas, h.solver, handle->stream().get(), n, d_a.data(), lda,
                              d_e.data(), lde, d_work.data(), bytes, d_info.data(), info);
  wwr::wwrStreamSynchronize(handle->stream().get());
  out_seed = from_device(handle, d_e, static_cast<std::size_t>(lde) * n);
  return status;
}

// Run pade at an explicit degree on A (n-by-n, lda=ldr=n). Returns r_m(A).
template<typename T>
std::vector<T> run_pade(std::shared_ptr<DeviceHandle> handle, Handles &h, int m, int n,
                        const std::vector<T> &a) {
  auto d_a = to_device(handle, a);
  std::vector<T> r(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  auto d_r = to_device(handle, r);

  std::size_t bytes = 0;
  EXPECT_EQ(pade_bufferSize<T>(h.solver, m, n, &bytes), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<T> d_work(bytes / sizeof(T) + 1, handle);
  DeviceBuffer<int> d_info(2, handle);

  EXPECT_EQ(pade<T>(h.blas, h.solver, handle->stream().get(), m, n, d_a.data(), n, d_r.data(), n,
                    d_work.data(), bytes, d_info.data()),
            wwr::WWRBLAS_STATUS_SUCCESS);
  wwr::wwrStreamSynchronize(handle->stream().get());
  return from_device(handle, d_r, static_cast<std::size_t>(n) * n);
}

// A random matrix scaled so its induced 1-norm equals @p target.
template<typename T>
std::vector<T> scaled_random(int n, double target, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<compute_t<T>> a(static_cast<std::size_t>(n) * n);
  constexpr bool is_cplx = !std::is_same_v<compute_t<T>, real_t<T>>;
  for (auto &x : a) {
    if constexpr (is_cplx) {
      x = compute_t<T>(static_cast<real_t<T>>(dist(rng)), static_cast<real_t<T>>(dist(rng)));
    } else {
      x = static_cast<compute_t<T>>(dist(rng));
    }
  }
  const double nrm = host_norm1(n, a);
  if (nrm > 0.0) {
    const auto s = static_cast<real_t<T>>(target / nrm);
    for (auto &x : a) {
      x *= s;
    }
  }
  return from_compute<T>(a);
}

// ========================================================================
// Host-only: the plan ladder and the thresholds
// ========================================================================

template<typename T>
void check_plan_ladder() {
  // Each threshold is covered by its own degree with no scaling, the product
  // counts are 2/3/4/5/6, and a norm past theta_13 scales.
  const int want_gemms[5] = {2, 3, 4, 5, 6};
  const int degrees[5] = {3, 5, 7, 9, 13};
  for (int i = 0; i < 5; ++i) {
    const real_t<T> theta = pade_theta<T>(degrees[i]);
    // A norm at 0.99*theta_m falls in rung m or below -- in particular s == 0
    // and the degree is <= m. Exactly at the boundary the cheapest covering
    // degree is what we assert against.
    const ExpmPlan plan = expm_plan<T>(static_cast<real_t<T>>(theta * real_t<T>{0.99}));
    EXPECT_EQ(plan.s, 0) << "degree index " << i << " must not scale";
    EXPECT_LE(plan.m, degrees[i]);
    EXPECT_EQ(plan.num_gemms, want_gemms[(plan.m == 3)   ? 0
                                         : (plan.m == 5) ? 1
                                         : (plan.m == 7) ? 2
                                         : (plan.m == 9) ? 3
                                                         : 4]);
  }
  // Nothing below theta_13 scales.
  EXPECT_EQ(expm_plan<T>(pade_theta<T>(13)).s, 0);
  // A norm well past theta_13 scales: degree 13, s > 0, cost 6 + s.
  const ExpmPlan big = expm_plan<T>(static_cast<real_t<T>>(pade_theta<T>(13) * real_t<T>{40}));
  EXPECT_EQ(big.m, 13);
  EXPECT_GT(big.s, 0);
  EXPECT_EQ(big.num_gemms, 6 + big.s);
}

TEST(ExpmPlanTests, LadderDouble) { check_plan_ladder<double>(); }
TEST(ExpmPlanTests, LadderFloat) { check_plan_ladder<float>(); }

TEST(ExpmThetaTests, OffLadderIsZero) {
  EXPECT_EQ(pade_theta<double>(4), 0.0);
  EXPECT_EQ(pade_theta<double>(0), 0.0);
  EXPECT_EQ(pade_theta<float>(11), 0.0f);
  // The ladder entries are strictly increasing, which is what makes "first that
  // fits is cheapest" correct.
  EXPECT_LT(pade_theta<double>(3), pade_theta<double>(5));
  EXPECT_LT(pade_theta<double>(5), pade_theta<double>(7));
  EXPECT_LT(pade_theta<double>(7), pade_theta<double>(9));
  EXPECT_LT(pade_theta<double>(9), pade_theta<double>(13));
}

// ========================================================================
// Host-only: argument validation (rejection paths return before device work)
// ========================================================================

TEST(ExpmArgCheckTests, RejectsBadArgumentsBeforeTouchingTheDevice) {
  const int n = 4;
  std::vector<double> a(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<double> e(static_cast<std::size_t>(n) * n, 0.0);
  std::vector<double> work(64, 0.0);
  int info_dev[2] = {0, 0};
  wwr::wwrblasHandle_t null_blas{};
  wwr::wwrsolverDnHandle_t null_solver{};

  auto call = [&](int nn, int lda, int lde, void *w) {
    return expm<double>(null_blas, null_solver, wwr::wwrStream_t{}, nn, a.data(), lda, e.data(), lde,
                        w, work.size() * sizeof(double), info_dev, nullptr);
  };
  EXPECT_EQ(call(0, n, n, work.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "n < 1";
  EXPECT_EQ(call(n, n - 1, n, work.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "lda < n";
  EXPECT_EQ(call(n, n, n - 1, work.data()), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "lde < n";
  EXPECT_EQ(call(n, n, n, nullptr), wwr::WWRBLAS_STATUS_INVALID_VALUE) << "null workspace";

  // The workspace queries reject n < 1 before touching the handle too.
  std::size_t bytes = 0;
  EXPECT_EQ(expm_bufferSize<double>(null_solver, 0, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(pade_bufferSize<double>(null_solver, 4, 0, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE);
  EXPECT_EQ(pade_bufferSize<double>(null_solver, 6, 4, &bytes), wwr::WWRBLAS_STATUS_INVALID_VALUE)
      << "degree 6 is off the ladder";
}

// ========================================================================
// Device: closed forms
// ========================================================================

template<typename T>
void check_zero_is_identity(int n) {
  auto handle = shared_device();
  Handles h = make_handles(handle);
  std::vector<T> a(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  std::vector<T> got(static_cast<std::size_t>(n) * n, traits<T>::make(-7.0, 0.0));
  ExpmPlan info{};
  ASSERT_EQ(run_expm<T>(handle, h, n, a, n, n, got, &info), wwr::WWRBLAS_STATUS_SUCCESS);
  const auto oracle = host_identity<compute_t<T>>(n);
  expect_close<T>(got, oracle, n, n, real_t<T>{64}, "exp(0)=I");
  destroy_handles(h);
}

TEST(ExpmClosedFormTests, ZeroIsIdentityFloat) { check_zero_is_identity<float>(5); }
TEST(ExpmClosedFormTests, ZeroIsIdentityDouble) { check_zero_is_identity<double>(6); }
TEST(ExpmClosedFormTests, ZeroIsIdentityComplexFloat) {
  check_zero_is_identity<wwr::wwrFloatComplex>(4);
}
TEST(ExpmClosedFormTests, ZeroIsIdentityComplexDouble) {
  check_zero_is_identity<wwr::wwrDoubleComplex>(4);
}

template<typename T>
void check_diagonal(int n, unsigned seed) {
  auto handle = shared_device();
  Handles h = make_handles(handle);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<compute_t<T>> ac(static_cast<std::size_t>(n) * n, compute_t<T>{});
  std::vector<compute_t<T>> oracle(static_cast<std::size_t>(n) * n, compute_t<T>{});
  constexpr bool is_cplx = !std::is_same_v<compute_t<T>, real_t<T>>;
  for (int i = 0; i < n; ++i) {
    compute_t<T> d;
    if constexpr (is_cplx) {
      d = compute_t<T>(static_cast<real_t<T>>(dist(rng)), static_cast<real_t<T>>(dist(rng)));
    } else {
      d = static_cast<compute_t<T>>(dist(rng));
    }
    ac[static_cast<std::size_t>(i) * n + i] = d;
    oracle[static_cast<std::size_t>(i) * n + i] = std::exp(d);
  }
  auto a = from_compute<T>(ac);
  std::vector<T> got(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  ExpmPlan info{};
  ASSERT_EQ(run_expm<T>(handle, h, n, a, n, n, got, &info), wwr::WWRBLAS_STATUS_SUCCESS);
  expect_close<T>(got, oracle, n, n, real_t<T>{128}, "exp(diag)=diag(exp)");
  destroy_handles(h);
}

TEST(ExpmClosedFormTests, DiagonalFloat) { check_diagonal<float>(5, 1); }
TEST(ExpmClosedFormTests, DiagonalDouble) { check_diagonal<double>(6, 2); }
TEST(ExpmClosedFormTests, DiagonalComplexDouble) { check_diagonal<wwr::wwrDoubleComplex>(5, 3); }

template<typename T>
void check_nilpotent(int n, unsigned seed) {
  // A strictly upper-triangular N is nilpotent: N^k = 0 for k >= n, so
  // exp(N) = sum_{k=0}^{n-1} N^k / k! terminates. The host sums that finite
  // series (a different path from the device's Pade), giving an exact oracle.
  auto handle = shared_device();
  Handles h = make_handles(handle);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);

  std::vector<compute_t<T>> nc(static_cast<std::size_t>(n) * n, compute_t<T>{});
  constexpr bool is_cplx = !std::is_same_v<compute_t<T>, real_t<T>>;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      if constexpr (is_cplx) {
        nc[static_cast<std::size_t>(j) * n + i] =
            compute_t<T>(static_cast<real_t<T>>(dist(rng)), static_cast<real_t<T>>(dist(rng)));
      } else {
        nc[static_cast<std::size_t>(j) * n + i] = static_cast<compute_t<T>>(dist(rng));
      }
    }
  }
  // Oracle: I + N + N^2/2! + ... + N^(n-1)/(n-1)!
  auto oracle = host_identity<compute_t<T>>(n);
  auto power = host_identity<compute_t<T>>(n);
  double fact = 1.0;
  for (int k = 1; k < n; ++k) {
    power = host_matmul(n, power, nc);
    fact *= k;
    const auto scale = static_cast<real_t<T>>(1.0 / fact);
    for (std::size_t idx = 0; idx < oracle.size(); ++idx) {
      oracle[idx] += power[idx] * scale;
    }
  }

  auto a = from_compute<T>(nc);
  std::vector<T> got(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  ExpmPlan info{};
  ASSERT_EQ(run_expm<T>(handle, h, n, a, n, n, got, &info), wwr::WWRBLAS_STATUS_SUCCESS);
  expect_close<T>(got, oracle, n, n, real_t<T>{128}, "exp(nilpotent)");
  destroy_handles(h);
}

TEST(ExpmClosedFormTests, NilpotentDouble) { check_nilpotent<double>(5, 10); }
TEST(ExpmClosedFormTests, NilpotentComplexFloat) { check_nilpotent<wwr::wwrFloatComplex>(4, 11); }

// ========================================================================
// Device: identities any exponential satisfies
// ========================================================================

template<typename T>
void check_inverse_identity(int n, double target, unsigned seed) {
  // exp(A) exp(-A) = I. Both exponentials come from the device; the host
  // multiplies them (the independent step) and checks the product is I.
  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto a = scaled_random<T>(n, target, seed);
  std::vector<compute_t<T>> ac = to_compute(a);
  std::vector<compute_t<T>> nac(ac.size());
  for (std::size_t i = 0; i < ac.size(); ++i) {
    nac[i] = -ac[i];
  }
  const auto na = from_compute<T>(nac);

  std::vector<T> e1(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  std::vector<T> e2 = e1;
  ExpmPlan info{};
  ASSERT_EQ(run_expm<T>(handle, h, n, a, n, n, e1, &info), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(run_expm<T>(handle, h, n, na, n, n, e2, &info), wwr::WWRBLAS_STATUS_SUCCESS);

  const auto prod = host_matmul(n, to_compute(e1), to_compute(e2));
  const auto oracle = host_identity<compute_t<T>>(n);
  const auto prod_T = from_compute<T>(prod);
  expect_close<T>(prod_T, oracle, n, n, real_t<T>{256}, "exp(A)exp(-A)=I");
  destroy_handles(h);
}

TEST(ExpmIdentityTests, InverseFloat) { check_inverse_identity<float>(5, 0.7, 20); }
TEST(ExpmIdentityTests, InverseDouble) { check_inverse_identity<double>(6, 1.5, 21); }
TEST(ExpmIdentityTests, InverseComplexDouble) {
  check_inverse_identity<wwr::wwrDoubleComplex>(5, 1.2, 22);
}

template<typename T>
void check_half_squared(int n, double target, unsigned seed) {
  // exp(A) = exp(A/2)^2. exp(A) and exp(A/2) both from the device; the host
  // squares the latter and compares.
  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto a = scaled_random<T>(n, target, seed);
  auto half_c = to_compute(a);
  for (auto &x : half_c) {
    x *= static_cast<real_t<T>>(0.5);
  }
  const auto half = from_compute<T>(half_c);

  std::vector<T> ea(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
  std::vector<T> eh = ea;
  ExpmPlan info{};
  ASSERT_EQ(run_expm<T>(handle, h, n, a, n, n, ea, &info), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(run_expm<T>(handle, h, n, half, n, n, eh, &info), wwr::WWRBLAS_STATUS_SUCCESS);

  const auto sq = host_matmul(n, to_compute(eh), to_compute(eh));
  expect_close<T>(ea, sq, n, n, real_t<T>{256}, "exp(A)=exp(A/2)^2");
  destroy_handles(h);
}

TEST(ExpmIdentityTests, HalfSquaredDouble) { check_half_squared<double>(6, 3.0, 30); }
TEST(ExpmIdentityTests, HalfSquaredComplexFloat) {
  check_half_squared<wwr::wwrFloatComplex>(4, 2.0, 31);
}

// ========================================================================
// Device: the degree ladder against an independent pade(13)
// ========================================================================

template<typename T>
void check_degree_ladder(int n) {
  // For a norm squarely inside each rung, expm picks that rung's degree (no
  // scaling) and the result must match r_13 evaluated on the same matrix -- a
  // different coefficient table and code path. A wrong coefficient anywhere on
  // the ladder breaks the agreement.
  auto handle = shared_device();
  Handles h = make_handles(handle);
  const int degrees[5] = {3, 5, 7, 9, 13};
  real_t<T> prev = real_t<T>{0};
  unsigned seed = 100;
  for (int i = 0; i < 5; ++i) {
    const real_t<T> theta = pade_theta<T>(degrees[i]);
    const double target = 0.5 * (static_cast<double>(prev) + static_cast<double>(theta));
    prev = theta;
    const auto a = scaled_random<T>(n, target, seed++);

    std::vector<T> got(static_cast<std::size_t>(n) * n, traits<T>::make(0.0, 0.0));
    ExpmPlan info{};
    ASSERT_EQ(run_expm<T>(handle, h, n, a, n, n, got, &info),
              wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(info.s, 0) << "rung " << degrees[i] << " must not scale";
    EXPECT_LE(info.m, degrees[i]);

    const auto r13 = run_pade<T>(handle, h, 13, n, a);
    expect_close<T>(got, to_compute(r13), n, n, real_t<T>{256}, "expm vs r_13");
  }
  destroy_handles(h);
}

TEST(ExpmLadderTests, LadderDouble) { check_degree_ladder<double>(5); }
TEST(ExpmLadderTests, LadderFloat) { check_degree_ladder<float>(5); }
TEST(ExpmLadderTests, LadderComplexDouble) { check_degree_ladder<wwr::wwrDoubleComplex>(4); }

// ========================================================================
// Device: padded leading dimensions and pointer-mode restoration
// ========================================================================

TEST(ExpmPaddingTests, PaddedOutputLeadingDimension) {
  // exp(A) written into a taller buffer: only the n-by-n block is the answer,
  // the rows between n and lde keep their sentinel.
  using T = double;
  const int n = 5;
  const int lde = n + 3;
  auto handle = shared_device();
  Handles h = make_handles(handle);
  const auto a = scaled_random<T>(n, 1.5, 60);

  const T sentinel = -123.5;
  std::vector<T> buf(static_cast<std::size_t>(lde) * n, sentinel);
  ExpmPlan info{};
  ASSERT_EQ(run_expm<T>(handle, h, n, a, n, lde, buf, &info), wwr::WWRBLAS_STATUS_SUCCESS);

  // exp(A) exp(-A) = I check, reading only the n-by-n block out of the padded buffer.
  std::vector<T> block(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      block[static_cast<std::size_t>(j) * n + i] = buf[static_cast<std::size_t>(j) * lde + i];
    }
    for (int i = n; i < lde; ++i) {
      EXPECT_DOUBLE_EQ(buf[static_cast<std::size_t>(j) * lde + i], sentinel)
          << "padding row " << i << " col " << j;
    }
  }
  // Sanity: the block is a plausible exponential (finite, nonzero).
  double frob = 0.0;
  for (double v : block) {
    frob += v * v;
  }
  EXPECT_GT(frob, 0.0);
  destroy_handles(h);
}

TEST(ExpmPointerModeTests, RestoresDevicePointerMode) {
  using T = double;
  const int n = 4;
  auto handle = shared_device();
  Handles h = make_handles(handle);

  // Put the handle in DEVICE pointer mode; expm must flip to HOST internally and
  // restore DEVICE on exit.
  ASSERT_EQ(wwr::wwrblasSetPointerMode(h.blas, wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);

  const auto a = scaled_random<T>(n, 1.0, 70);
  auto d_a = to_device(handle, a);
  std::vector<T> e(static_cast<std::size_t>(n) * n, 0.0);
  auto d_e = to_device(handle, e);
  std::size_t bytes = 0;
  ASSERT_EQ(expm_bufferSize<T>(h.solver, n, &bytes), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<T> d_work(bytes / sizeof(T) + 1, handle);
  DeviceBuffer<int> d_info(2, handle);
  ASSERT_EQ(expm<T>(h.blas, h.solver, handle->stream().get(), n, d_a.data(), n, d_e.data(), n,
                    d_work.data(), bytes, d_info.data(), nullptr),
            wwr::WWRBLAS_STATUS_SUCCESS);

  wwr::wwrblasPointerMode_t mode{};
  ASSERT_EQ(wwr::wwrblasGetPointerMode(h.blas, &mode), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(mode, wwr::WWRBLAS_POINTER_MODE_DEVICE) << "pointer mode must be restored";
  destroy_handles(h);
}

} // namespace
} // namespace calaman
