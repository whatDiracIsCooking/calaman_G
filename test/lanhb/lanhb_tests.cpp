// Oracle test for calaman.lanhb -- the ?lanhb norm of a complex Hermitian band
// matrix in LAPACK band storage. The oracle is the reference ?lanhb in the
// SAME precision on the host, called through lapack.h's LAPACK_?lanhb (LAPACKE
// has no ?lanhb wrapper). The wwr complex types are layout-compatible with
// lapack_complex_*.
//
// Both UPLO values run. Each diagonal entry carries the largest finite
// imaginary part -- ?lanhb reads only the real part, so a read of it overflows
// the result. Every slot of AB that holds no in-matrix band entry of the stored
// triangle (the unused corner and the ldab > k+1 padding rows) carries garbage
// (NaN and the largest finite value). The Fortran routine has no input NaN
// check, so the oracle accepts that garbage. |z| is an inexact hypot, so every
// norm takes the shared tolerance. Every suite stages data on the device, so it
// is REQUIRES_GPU (labeled `gpu`). Built only when calaman::lapack_reference
// exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lanhb;
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
using test::factorization_tol;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

constexpr Uplo kUplos[] = {Uplo::U, Uplo::L};

// Every LAPACK NORM char lanhb accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// Diagonal, one off-diagonal, a narrow band, a band whose 2k+1-row column spans
// more than one block's 256 threads, and k past n (a full matrix).
constexpr std::size_t kBandwidths[] = {0, 1, 3, 200, 1100};

// The n == 1 edge, small orders, around one block's stride, and many strides.
constexpr std::size_t kSizes[] = {1, 2, 3, 17, 255, 257, 700};

template<typename T>
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
  const std::size_t n = std::max<std::size_t>(host.size(), 1);
  HostBuffer<T> staging(n);
  std::copy(host.begin(), host.end(), staging.data());
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, staging, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return device;
}

template<typename T>
T scalar_from_device(const DeviceBuffer<T> &device) {
  const auto handle = shared_device();
  HostBuffer<T> host(1);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return host.data()[0];
}

char uplo_char(Uplo uplo) {
  return uplo == Uplo::U ? 'U' : 'L';
}

template<typename C>
struct complex_elem;

template<>
struct complex_elem<wwr::wwrFloatComplex> {
  using R = float;
  using C = wwr::wwrFloatComplex;
  static C make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, char uplo, std::size_t n, std::size_t k, const C *ab,
                   std::size_t ldab) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ki = static_cast<lapack_int>(k);
    const lapack_int ldi = static_cast<lapack_int>(ldab);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return static_cast<float>(LAPACK_clanhb(&norm, &uplo, &ni, &ki,
                                            reinterpret_cast<const lapack_complex_float *>(ab),
                                            &ldi, work.data()));
  }
};

template<>
struct complex_elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char uplo, std::size_t n, std::size_t k, const C *ab,
                    std::size_t ldab) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ki = static_cast<lapack_int>(k);
    const lapack_int ldi = static_cast<lapack_int>(ldab);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return LAPACK_zlanhb(&norm, &uplo, &ni, &ki,
                         reinterpret_cast<const lapack_complex_double *>(ab), &ldi, work.data());
  }
};

// The AB row of the diagonal entry in each column.
std::size_t diag_row(Uplo uplo, std::size_t k) {
  return uplo == Uplo::U ? k : 0;
}

// Row r of AB column j holds A(r+j-k, j) (Uplo::U) or A(r+j, j) (Uplo::L): an
// in-matrix band entry of the stored triangle iff r <= k and that row lies in
// [0, n).
bool in_band(Uplo uplo, std::size_t r, std::size_t j, std::size_t n, std::size_t k) {
  return r <= k && (uplo == Uplo::U ? r + j >= k : r + j < n);
}

// The ldab-by-n band storage whose in-band slot (r, j) is @p fill(r, j); every
// other slot alternates NaN and the largest finite component.
template<typename C, typename Fill>
std::vector<C> to_band(Uplo uplo, std::size_t n, std::size_t k, std::size_t ldab, Fill fill) {
  using R = typename complex_elem<C>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<R>::max();
  std::vector<C> ab(ldab * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t r = 0; r < ldab; ++r) {
      ab[r + j * ldab] = in_band(uplo, r, j, n, k) ? fill(r, j)
                         : (r + j) % 2 == 0        ? complex_elem<C>::make(nan, nan)
                                                   : complex_elem<C>::make(big, big);
    }
  }
  return ab;
}

// Random complex band entries over a few binades; each diagonal entry a random
// real part and the largest finite imaginary part (never read).
template<typename C>
std::vector<C> make_band(Uplo uplo, std::size_t n, std::size_t k, std::size_t ldab,
                         std::uint32_t seed) {
  using R = typename complex_elem<C>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double big = std::numeric_limits<R>::max();
  const std::size_t d = diag_row(uplo, k);
  return to_band<C>(uplo, n, k, ldab, [&](std::size_t r, std::size_t) {
    const double re = dist(gen);
    const double im = dist(gen);
    return complex_elem<C>::make(re, r == d ? big : im);
  });
}

template<typename C>
void expect_matches_reference(const NormCase nc, Uplo uplo, std::size_t n, std::size_t k,
                              std::size_t ldab) {
  using R = typename complex_elem<C>::R;
  const auto ab =
      make_band<C>(uplo, n, k, ldab, static_cast<std::uint32_t>(31 * n + 7 * ldab + 5 * k + 1));
  const R oracle = complex_elem<C>::ref(nc.c, uplo_char(uplo), n, k, ab.data(), ldab);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage or a diagonal imaginary part";

  auto d_AB = to_device(ab);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lanhb<C>(shared_device()->stream().get(), nc.which, uplo, n, k, d_AB.data(),
                            ldab, d_result.data());
  ASSERT_TRUE(s.ok()) << "lanhb returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, factorization_tol<R>(oracle, n, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n << " k=" << k
      << " ldab=" << ldab;
}

// Each case with the tight ldab == k+1 and a padded one.
template<typename C>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const std::size_t k : kBandwidths) {
        for (const std::size_t n : kSizes) {
          expect_matches_reference<C>(nc, uplo, n, k, k + 1);
          expect_matches_reference<C>(nc, uplo, n, k, k + 4);
        }
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanhb, per type
// ========================================================================

TEST(LanhbOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanhbOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// A dominant diagonal decides the max norm: |Re A(j,j)| must be the answer,
// not |A(j,j)| -- a guard that the diagonal's imaginary part is dropped even
// where it would not overflow.
TEST(LanhbOracleTests, MaxAbsIgnoresDiagonalImaginary) {
  using C = wwr::wwrDoubleComplex;
  const std::size_t k = 2;
  const std::size_t ldab = k + 2;
  for (const std::size_t n : {4u, 5u}) {
    for (const Uplo uplo : kUplos) {
      const std::size_t d = diag_row(uplo, k);
      const auto ab = to_band<C>(uplo, n, k, ldab, [&](std::size_t r, std::size_t j) {
        return r == d && j == 2 ? complex_elem<C>::make(-3.0, 4.0) // |z| = 5, |Re z| = 3
                                : complex_elem<C>::make(0.5, 0.5);
      });
      const double oracle = complex_elem<C>::ref('M', uplo_char(uplo), n, k, ab.data(), ldab);
      ASSERT_EQ(oracle, 3.0);
      auto d_AB = to_device(ab);
      DeviceBuffer<double> d_result(1, shared_device());
      ASSERT_TRUE(lanhb<C>(shared_device()->stream().get(), MatrixNorm::max_abs, uplo, n, k,
                           d_AB.data(), ldab, d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), oracle) << "uplo=" << uplo_char(uplo) << " n=" << n;
    }
  }
}

// n == 0 writes a real 0 for every norm and triangle, over a sentinel.
template<typename C>
void expect_empty_writes_zero() {
  using R = typename complex_elem<C>::R;
  auto dummy = to_device(std::vector<C>{complex_elem<C>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      auto d_result = to_device(std::vector<R>{R(-12345)});
      ASSERT_TRUE(lanhb<C>(shared_device()->stream().get(), nc.which, uplo, 0, 1, dummy.data(), 2,
                           d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), R{0})
          << "norm=" << nc.c << " uplo=" << uplo_char(uplo);
    }
  }
}

TEST(LanhbOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in an off-diagonal band entry (the outermost stored diagonal) or in a
// diagonal entry's real part propagates through every norm, as ZLANHB's
// DISNAN guard does for the max folds.
TEST(LanhbOracleTests, NaNPropagates) {
  using C = wwr::wwrDoubleComplex;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t n = 300;
  const std::size_t k = 4;
  const std::size_t ldab = k + 3;
  for (const Uplo uplo : kUplos) {
    const std::size_t d = diag_row(uplo, k);
    const std::size_t edge = uplo == Uplo::U ? 0 : k;
    const std::tuple<std::size_t, std::size_t, C> spots[] = {
        {edge, 150, complex_elem<C>::make(1.0, nan)}, {d, 77, complex_elem<C>::make(nan, 1.0)}};
    for (const auto &[r, j, value] : spots) {
      auto ab = make_band<C>(uplo, n, k, ldab, 9);
      ab[r + j * ldab] = value;
      auto d_AB = to_device(ab);
      DeviceBuffer<double> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lanhb<C>(shared_device()->stream().get(), nc.which, uplo, n, k, d_AB.data(),
                             ldab, d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " r=" << r << " j=" << j;
      }
    }
  }
}

} // namespace
} // namespace calaman
