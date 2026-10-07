// Oracle test for calaman.lanhp -- the ?lanhp norm of a complex Hermitian
// matrix in packed storage. The oracle is the reference ?lanhp in the SAME
// precision on the host, called through lapack.h's LAPACK_?lanhp (LAPACKE has
// no ?lanhp wrapper). The wwr complex types are layout-compatible with
// lapack_complex_*.
//
// Both UPLO values run. AP is built on the host by LAPACKE_?trttp from a random
// triangle of a padded (lda = n + 3) matrix whose diagonal carries the largest
// finite imaginary part -- ?lanhp reads only the real part, so a read of it
// overflows the result. The device copy carries trailing garbage (NaN and the
// largest finite value) past its n(n+1)/2 elements, packed storage's stand-in
// for a padded leading dimension. |z| is an inexact hypot, so every norm takes
// the shared tolerance. Every suite stages data on the device, so it is
// REQUIRES_GPU (labeled `gpu`). Built only when calaman::lapack_reference
// exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lanhp;
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

// Every LAPACK NORM char lanhp accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// Orders 1..5 (the n == 1 edge), around one block's stride, and many strides.
constexpr std::size_t kSizes[] = {1, 2, 3, 4, 5, 16, 17, 255, 256, 257, 600};

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
  static float ref(char norm, char uplo, std::size_t n, const C *ap) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return static_cast<float>(LAPACK_clanhp(
        &norm, &uplo, &ni, reinterpret_cast<const lapack_complex_float *>(ap), work.data()));
  }
  static lapack_int trttp(char uplo, std::size_t n, const C *a, std::size_t lda, C *ap) {
    return LAPACKE_ctrttp(LAPACK_COL_MAJOR, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_float *>(a),
                          static_cast<lapack_int>(lda),
                          reinterpret_cast<lapack_complex_float *>(ap));
  }
};

template<>
struct complex_elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char uplo, std::size_t n, const C *ap) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return LAPACK_zlanhp(&norm, &uplo, &ni, reinterpret_cast<const lapack_complex_double *>(ap),
                         work.data());
  }
  static lapack_int trttp(char uplo, std::size_t n, const C *a, std::size_t lda, C *ap) {
    return LAPACKE_ztrttp(LAPACK_COL_MAJOR, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_double *>(a),
                          static_cast<lapack_int>(lda),
                          reinterpret_cast<lapack_complex_double *>(ap));
  }
};

// The packed array of a padded (lda = n + 3) triangle produced by
// @p fill(i, j), converted by the reference ?trttp.
template<typename C, typename Fill>
std::vector<C> to_ap(Uplo uplo, std::size_t n, Fill fill) {
  const std::size_t lda = n + 3;
  std::vector<C> a(lda * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      a[i + j * lda] = fill(i, j);
    }
  }
  std::vector<C> ap(n * (n + 1) / 2);
  const lapack_int info = complex_elem<C>::trttp(uplo_char(uplo), n, a.data(), lda, ap.data());
  EXPECT_EQ(info, 0);
  return ap;
}

// Random complex entries over a few binades; each diagonal entry a random real
// part and the largest finite imaginary part (never read).
template<typename C>
std::vector<C> make_ap(Uplo uplo, std::size_t n, std::uint32_t seed) {
  using R = typename complex_elem<C>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double big = std::numeric_limits<R>::max();
  return to_ap<C>(uplo, n, [&](std::size_t i, std::size_t j) {
    const double re = dist(gen);
    return complex_elem<C>::make(re, i == j ? big : dist(gen));
  });
}

// AP followed by trailing garbage the kernel must never read.
template<typename C>
std::vector<C> with_slack(std::vector<C> ap) {
  using R = typename complex_elem<C>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  ap.push_back(complex_elem<C>::make(nan, nan));
  ap.push_back(complex_elem<C>::make(std::numeric_limits<R>::max(), 0.0));
  return ap;
}

template<typename C>
void expect_matches_reference(const NormCase nc, Uplo uplo, std::size_t n) {
  using R = typename complex_elem<C>::R;
  const auto ap = make_ap<C>(uplo, n, static_cast<std::uint32_t>(37 * n + 5));
  const R oracle = complex_elem<C>::ref(nc.c, uplo_char(uplo), n, ap.data());
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the diagonal's imaginary part";

  auto d_ap = to_device(with_slack(ap));
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s =
      lanhp<C>(shared_device()->stream().get(), nc.which, uplo, n, d_ap.data(), d_result.data());
  ASSERT_TRUE(s.ok()) << "lanhp returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, factorization_tol<R>(oracle, n, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n;
}

template<typename C>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const std::size_t n : kSizes) {
        expect_matches_reference<C>(nc, uplo, n);
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanhp, per type
// ========================================================================

TEST(LanhpOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanhpOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// A dominant diagonal decides the max norm: |Re A(j,j)| must be the answer,
// not |A(j,j)| -- a guard that the diagonal's imaginary part is dropped even
// where it would not overflow.
TEST(LanhpOracleTests, MaxAbsIgnoresDiagonalImaginary) {
  using C = wwr::wwrDoubleComplex;
  for (const std::size_t n : {4u, 5u}) {
    for (const Uplo uplo : kUplos) {
      const auto ap = to_ap<C>(uplo, n, [](std::size_t i, std::size_t j) {
        return i == 2 && j == 2 ? complex_elem<C>::make(-3.0, 4.0) // |z| = 5, |Re z| = 3
                                : complex_elem<C>::make(0.5, 0.5);
      });
      const double oracle = complex_elem<C>::ref('M', uplo_char(uplo), n, ap.data());
      ASSERT_EQ(oracle, 3.0);
      auto d_ap = to_device(ap);
      DeviceBuffer<double> d_result(1, shared_device());
      ASSERT_TRUE(lanhp<C>(shared_device()->stream().get(), MatrixNorm::max_abs, uplo, n,
                           d_ap.data(), d_result.data())
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
      ASSERT_TRUE(lanhp<C>(shared_device()->stream().get(), nc.which, uplo, 0, dummy.data(),
                           d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), R{0})
          << "norm=" << nc.c << " uplo=" << uplo_char(uplo);
    }
  }
}

TEST(LanhpOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in an off-diagonal AP entry propagates through every norm, as
// ZLANHP's DISNAN guard does for the max folds. Patched in after ?trttp, whose
// LAPACKE input check rejects a NaN; the diagonal slots are the ones holding
// the largest finite imaginary part, so the patch skips them.
TEST(LanhpOracleTests, NaNPropagates) {
  using C = wwr::wwrDoubleComplex;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<double>::max();
  for (const std::size_t n : {300u, 301u}) {
    for (const Uplo uplo : kUplos) {
      auto ap = make_ap<C>(uplo, n, 9);
      std::size_t k = ap.size() / 3;
      while (std::abs(wwr::wwrCimag(ap[k])) == big) {
        ++k;
      }
      ap[k] = complex_elem<C>::make(1.0, nan);
      auto d_ap = to_device(ap);
      DeviceBuffer<double> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lanhp<C>(shared_device()->stream().get(), nc.which, uplo, n, d_ap.data(),
                             d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n;
      }
    }
  }
}

} // namespace
} // namespace calaman
