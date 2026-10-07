// Oracle test for calaman.rscl -- scale a strided vector by 1/a without
// over/underflow. The oracle is the reference ?rscl in the SAME precision on
// the host, over the same buffers (the wwr complex types are layout-compatible
// with the Fortran COMPLEX types).
//
// NEITHER LAPACKE NOR lapack.h WRAPS ?rscl, so the four Fortran symbols are
// declared here -- test/lartg/'s precedent. They are real LAPACK routines
// (srscl_, drscl_, csrscl_, zdrscl_ are all in liblapack), not a reimplementation.
//
// THE COMPARISON IS BIT-FOR-BIT. Both paths run the same guarded multiplier
// loop and so apply the identical chain of factors in the identical order, each
// as one multiply per element -- so there is nothing for the two to round
// differently, including in the cases chosen to make the chain take more than
// one step.
//
// All suites stage data on the device, so they are REQUIRES_GPU (labeled
// `gpu`). Built only when calaman::lapack_reference exists; see this
// directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h> // lapack_int, lapack_complex_*

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.rscl;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

// The reference ?rscl family, by Fortran symbol: everything by reference, and
// no hidden string-length argument (no CHARACTER parameter).
extern "C" {
void srscl_(const lapack_int *n, const float *sa, float *sx, const lapack_int *incx);
void drscl_(const lapack_int *n, const double *sa, double *sx, const lapack_int *incx);
void csrscl_(const lapack_int *n, const float *sa, lapack_complex_float *sx,
             const lapack_int *incx);
void zdrscl_(const lapack_int *n, const double *sa, lapack_complex_double *sx,
             const lapack_int *incx);
}

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

// ========================================================================
// Per-element-type traits: the builder, the component accessors, the oracle,
// and the two scalars that make the multiplier loop take more than one step
// ========================================================================

template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static float make(double re, double) { return static_cast<float>(re); }
  static float re(float v) { return v; }
  static float im(float) { return 0.0F; }
  static void ref(std::size_t n, float a, float *x, int incx) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    srscl_(&ni, &a, x, &ii);
  }
};

template<>
struct elem<double> {
  using R = double;
  static double make(double re, double) { return re; }
  static double re(double v) { return v; }
  static double im(double) { return 0.0; }
  static void ref(std::size_t n, double a, double *x, int incx) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    drscl_(&ni, &a, x, &ii);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  // Through the host wwrC* accessors in wwr.complex -- the member .x/.y is not
  // portable to hipComplex.
  static float re(wwr::wwrFloatComplex v) { return wwr::wwrCrealf(v); }
  static float im(wwr::wwrFloatComplex v) { return wwr::wwrCimagf(v); }
  static void ref(std::size_t n, float a, wwr::wwrFloatComplex *x, int incx) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    csrscl_(&ni, &a, reinterpret_cast<lapack_complex_float *>(x), &ii);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double re(wwr::wwrDoubleComplex v) { return wwr::wwrCreal(v); }
  static double im(wwr::wwrDoubleComplex v) { return wwr::wwrCimag(v); }
  static void ref(std::size_t n, double a, wwr::wwrDoubleComplex *x, int incx) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    zdrscl_(&ni, &a, reinterpret_cast<lapack_complex_double *>(x), &ii);
  }
};

// ========================================================================
// Staging
// ========================================================================

template<typename T>
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
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
std::vector<T> from_device(const DeviceBuffer<T> &device, std::size_t n) {
  const auto handle = shared_device();
  HostBuffer<T> host(n == 0 ? 1 : n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

template<typename T>
void expect_same(const std::vector<T> &got, const std::vector<T> &want, const std::string &what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(elem<T>::re(got[i]), elem<T>::re(want[i])) << what << " re at " << i;
    EXPECT_EQ(elem<T>::im(got[i]), elem<T>::im(want[i])) << what << " im at " << i;
  }
}

// Mixed-sign values over a few binades, every component shifted by 2^shift so
// the quotient x/a can be placed in range for any a the loop is tested with.
template<typename T>
std::vector<T> random_vector(std::size_t len, std::uint32_t seed, int shift = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  std::vector<T> v(len);
  for (auto &e : v) {
    e = elem<T>::make(std::ldexp(dist(gen), shift), std::ldexp(dist(gen), shift));
  }
  return v;
}

// ========================================================================
// The one comparison every case runs
// ========================================================================

template<typename T>
void expect_matches_reference(const std::vector<T> &x, std::size_t n, typename elem<T>::R a,
                              int incx, const std::string &what) {
  std::vector<T> want = x;
  elem<T>::ref(n, a, want.data(), incx);

  auto d_x = to_device(x);
  const Status s =
      rscl<T>(shared_device()->stream().get(), static_cast<int>(n), a, d_x.data(), incx);
  ASSERT_TRUE(s.ok()) << what << ": rscl returned " << s.name() << ": " << s.message();
  wwr::wwrStreamSynchronize(shared_device()->stream().get());

  expect_same(from_device(d_x, x.size()), want, what);
}

constexpr std::size_t kSizes[] = {1, 2, 3, 7, 64, 255, 256, 257, 1000, 4099};

// ========================================================================
// Ordinary scalars, every size
// ========================================================================

template<typename T>
void run_sizes() {
  using R = typename elem<T>::R;
  for (const std::size_t n : kSizes) {
    const auto x = random_vector<T>(n, static_cast<std::uint32_t>(19 * n + 3));
    for (const R a : {R{2}, R{-3.5}, R{0.125}, R{1}}) {
      expect_matches_reference<T>(x, n, a, 1,
                                  "n=" + std::to_string(n) + " a=" + std::to_string(a));
    }
  }
}

TEST(RsclOracleTests, SizesFloat) {
  run_sizes<float>();
}
TEST(RsclOracleTests, SizesDouble) {
  run_sizes<double>();
}
TEST(RsclOracleTests, SizesComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(RsclOracleTests, SizesComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// ========================================================================
// The multi-step chains, which are the reason ?rscl is not x * (1/a)
// ========================================================================

// a above bignum: 1/a is subnormal or zero, so the loop shrinks x by smlnum
// first and finishes with a second, in-range factor. The vector is shifted up
// so the quotient lands mid-range rather than in the subnormals, where a
// comparison would only be testing the rounding of a denormal.
template<typename T>
void run_huge_scalar() {
  using R = typename elem<T>::R;
  const R a = std::numeric_limits<R>::max();
  const int shift = std::numeric_limits<R>::max_exponent / 2;
  for (const std::size_t n : {std::size_t{1}, std::size_t{300}, std::size_t{4099}}) {
    const auto x = random_vector<T>(n, static_cast<std::uint32_t>(23 * n + 11), shift);
    expect_matches_reference<T>(x, n, a, 1, "huge a, n=" + std::to_string(n));
    expect_matches_reference<T>(x, n, -a, 1, "huge -a, n=" + std::to_string(n));
  }
}

// a below smlnum (a subnormal): 1/a overflows, so the loop grows x by bignum
// first. The vector is shifted down to keep the quotient in range.
template<typename T>
void run_tiny_scalar() {
  using R = typename elem<T>::R;
  const R a = std::numeric_limits<R>::denorm_min() * R{4};
  const int shift = (std::numeric_limits<R>::min_exponent - std::numeric_limits<R>::digits) / 2;
  for (const std::size_t n : {std::size_t{1}, std::size_t{300}, std::size_t{4099}}) {
    const auto x = random_vector<T>(n, static_cast<std::uint32_t>(29 * n + 13), shift);
    expect_matches_reference<T>(x, n, a, 1, "tiny a, n=" + std::to_string(n));
    expect_matches_reference<T>(x, n, -a, 1, "tiny -a, n=" + std::to_string(n));
  }
}

TEST(RsclOracleTests, HugeScalarFloat) {
  run_huge_scalar<float>();
}
TEST(RsclOracleTests, HugeScalarDouble) {
  run_huge_scalar<double>();
}
TEST(RsclOracleTests, HugeScalarComplexDouble) {
  run_huge_scalar<wwr::wwrDoubleComplex>();
}
TEST(RsclOracleTests, TinyScalarFloat) {
  run_tiny_scalar<float>();
}
TEST(RsclOracleTests, TinyScalarDouble) {
  run_tiny_scalar<double>();
}
TEST(RsclOracleTests, TinyScalarComplexDouble) {
  run_tiny_scalar<wwr::wwrDoubleComplex>();
}

// The factor chain must be more than one step for the two tests above to be
// testing anything, so assert the conditions that drive it rather than trusting
// the chosen constants: netlib ?rscl shifts when |a*smlnum| > 1 or smlnum > |a|.
template<typename R>
void expect_chain_is_multi_step() {
  const R smlnum = std::numeric_limits<R>::min();
  EXPECT_GT(std::abs(std::numeric_limits<R>::max() * smlnum), R{1});
  EXPECT_GT(smlnum, std::numeric_limits<R>::denorm_min() * R{4});
}

TEST(RsclOracleTests, ChosenScalarsDoExerciseTheChainFloat) {
  expect_chain_is_multi_step<float>();
}
TEST(RsclOracleTests, ChosenScalarsDoExerciseTheChainDouble) {
  expect_chain_is_multi_step<double>();
}

// ========================================================================
// Strides and the no-op edges
// ========================================================================

// incx > 1: the slots between the scaled elements must come back untouched,
// which the reference's ?scal guarantees too -- so comparing the whole buffer
// checks the stride in both directions at once.
template<typename T>
void run_strides() {
  const std::size_t n = 257;
  for (const int incx : {2, 3}) {
    const auto stride = static_cast<std::size_t>(incx);
    const auto x = random_vector<T>((n - 1) * stride + 1, 83);
    expect_matches_reference<T>(x, n, typename elem<T>::R{-7}, incx,
                                "incx=" + std::to_string(incx));
  }
}

TEST(RsclOracleTests, StridesFloat) {
  run_strides<float>();
}
TEST(RsclOracleTests, StridesDouble) {
  run_strides<double>();
}
TEST(RsclOracleTests, StridesComplexDouble) {
  run_strides<wwr::wwrDoubleComplex>();
}

// n == 0 and incx <= 0 are both no-ops -- the second because the reference's
// ?scal returns early for a non-positive stride, whatever the factor chain did.
TEST(RsclOracleTests, EmptyAndNonPositiveStrideAreNoOps) {
  const auto x = random_vector<double>(64, 97);
  for (const std::size_t n : {std::size_t{0}, std::size_t{64}}) {
    for (const int incx : {1, 0, -1, -2}) {
      if (n > 0 && incx > 0) {
        continue; // the one combination that does scale
      }
      expect_matches_reference<double>(x, n, 2.0, incx,
                                       "n=" + std::to_string(n) + " incx=" + std::to_string(incx));
    }
  }
}

// A negative n is n == 0, as in the reference's `IF (N.LE.0) RETURN`. It cannot
// go through expect_matches_reference, whose n is unsigned.
TEST(RsclOracleTests, NegativeOrderIsANoOp) {
  const auto x = random_vector<double>(64, 99);
  for (const int n : {-1, -64}) {
    auto d_x = to_device(x);
    const Status s = rscl<double>(shared_device()->stream().get(), n, 2.0, d_x.data(), 1);
    ASSERT_TRUE(s.ok()) << "n=" << n << ": " << s.name();
    wwr::wwrStreamSynchronize(shared_device()->stream().get());
    expect_same(from_device(d_x, x.size()), x, "n=" + std::to_string(n));
  }
}

// ========================================================================
// The rejected scalars -- calaman's one divergence from netlib ?rscl
// ========================================================================

// a == 0 and a non-finite a are InvalidValue, and x is left alone. The reference
// reports no INFO here and spins forever on an infinite a, so there is nothing
// to compare against: see src/lapack/rscl/README.md.
TEST(RsclOracleTests, RejectsZeroAndNonFiniteScalar) {
  const auto x = random_vector<double>(100, 101);
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double a : {0.0, -0.0, inf, -inf, nan}) {
    auto d_x = to_device(x);
    const Status s = rscl<double>(shared_device()->stream().get(), 100, a, d_x.data(), 1);
    EXPECT_EQ(s, wwr::wwrErrorInvalidValue) << "a=" << a << " status=" << s.name();
    wwr::wwrStreamSynchronize(shared_device()->stream().get());
    expect_same(from_device(d_x, x.size()), x, "rejected a=" + std::to_string(a));
  }
}

} // namespace
} // namespace calaman
