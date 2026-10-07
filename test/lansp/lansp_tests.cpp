// Oracle test for calaman.lansp -- the ?lansp norm of a symmetric matrix in
// packed storage, s/d/c/z (complex symmetric, not Hermitian). The oracle is the
// reference ?lansp in the SAME precision on the host, called through lapack.h's
// LAPACK_?lansp (LAPACKE has no ?lansp wrapper). The wwr complex types are
// layout-compatible with lapack_complex_*.
//
// Both UPLO values run. AP is built on the host by LAPACKE_?trttp from a random
// triangle of a padded (lda = n + 3) matrix, and the device copy carries
// trailing garbage (NaN and the largest finite value) past its n(n+1)/2
// elements, so an over-read turns the result into NaN or a wrong max and fails
// the comparison -- packed storage has no leading dimension; that trailing
// slack is its stand-in. |x| is exact for a real T, so the real max norm is
// compared exactly; every other norm takes the shared tolerance. Every suite
// stages data on the device, so it is REQUIRES_GPU (labeled `gpu`). Built only
// when calaman::lapack_reference exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lansp;
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

// Every LAPACK NORM char lansp accepts, with the MatrixNorm it maps to ('O' is
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

// Per-type construction, the reference ?lansp and ?trttp.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool kComplex = false;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, char uplo, std::size_t n, const float *ap) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return static_cast<float>(LAPACK_slansp(&norm, &uplo, &ni, ap, work.data()));
  }
  static lapack_int trttp(char uplo, std::size_t n, const float *a, std::size_t lda, float *ap) {
    return LAPACKE_strttp(LAPACK_COL_MAJOR, uplo, static_cast<lapack_int>(n), a,
                          static_cast<lapack_int>(lda), ap);
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool kComplex = false;
  static double make(double re, double) { return re; }
  static double ref(char norm, char uplo, std::size_t n, const double *ap) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return LAPACK_dlansp(&norm, &uplo, &ni, ap, work.data());
  }
  static lapack_int trttp(char uplo, std::size_t n, const double *a, std::size_t lda, double *ap) {
    return LAPACKE_dtrttp(LAPACK_COL_MAJOR, uplo, static_cast<lapack_int>(n), a,
                          static_cast<lapack_int>(lda), ap);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  using C = wwr::wwrFloatComplex;
  static constexpr bool kComplex = true;
  static C make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, char uplo, std::size_t n, const C *ap) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return static_cast<float>(LAPACK_clansp(
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
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static constexpr bool kComplex = true;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char uplo, std::size_t n, const C *ap) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return LAPACK_zlansp(&norm, &uplo, &ni, reinterpret_cast<const lapack_complex_double *>(ap),
                         work.data());
  }
  static lapack_int trttp(char uplo, std::size_t n, const C *a, std::size_t lda, C *ap) {
    return LAPACKE_ztrttp(LAPACK_COL_MAJOR, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_double *>(a),
                          static_cast<lapack_int>(lda),
                          reinterpret_cast<lapack_complex_double *>(ap));
  }
};

// The n(n+1)/2-element packed array of a random symmetric matrix over a few
// binades, built by the reference ?trttp from a padded (lda = n + 3) triangle.
template<typename T>
std::vector<T> make_ap(Uplo uplo, std::size_t n, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const std::size_t lda = n + 3;
  std::vector<T> a(lda * n);
  for (T &x : a) {
    const double re = dist(gen);
    x = elem<T>::make(re, dist(gen));
  }
  std::vector<T> ap(n * (n + 1) / 2);
  const lapack_int info = elem<T>::trttp(uplo_char(uplo), n, a.data(), lda, ap.data());
  EXPECT_EQ(info, 0);
  return ap;
}

// AP followed by trailing garbage the kernel must never read.
template<typename T>
std::vector<T> with_slack(std::vector<T> ap) {
  using R = typename elem<T>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  ap.push_back(elem<T>::make(nan, nan));
  ap.push_back(elem<T>::make(std::numeric_limits<R>::max(), 0.0));
  return ap;
}

// |x| is exact for a real T, so the real max norm is compared exactly; the
// sums, and every complex norm (|z| is an inexact hypot), take the shared
// tolerance.
template<typename T>
typename elem<T>::R norm_tol(MatrixNorm which, typename elem<T>::R ref, std::size_t n) {
  using R = typename elem<T>::R;
  return (!elem<T>::kComplex && which == MatrixNorm::max_abs) ? R{0}
                                                              : factorization_tol<R>(ref, n, n);
}

template<typename T>
void expect_matches_reference(const NormCase nc, Uplo uplo, std::size_t n) {
  using R = typename elem<T>::R;
  const auto ap = make_ap<T>(uplo, n, static_cast<std::uint32_t>(37 * n + 5));
  const R oracle = elem<T>::ref(nc.c, uplo_char(uplo), n, ap.data());
  ASSERT_TRUE(std::isfinite(oracle));

  auto d_ap = to_device(with_slack(ap));
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s =
      lansp<T>(shared_device()->stream().get(), nc.which, uplo, n, d_ap.data(), d_result.data());
  ASSERT_TRUE(s.ok()) << "lansp returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol<T>(nc.which, oracle, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n;
}

template<typename T>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const std::size_t n : kSizes) {
        expect_matches_reference<T>(nc, uplo, n);
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lansp, per type
// ========================================================================

TEST(LanspOracleTests, Float) {
  run_sizes<float>();
}
TEST(LanspOracleTests, Double) {
  run_sizes<double>();
}
TEST(LanspOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanspOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// n == 0 writes 0 for every norm and triangle, over a sentinel.
template<typename T>
void expect_empty_writes_zero() {
  using R = typename elem<T>::R;
  auto dummy = to_device(std::vector<T>{elem<T>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      auto d_result = to_device(std::vector<R>{R(-12345)});
      ASSERT_TRUE(lansp<T>(shared_device()->stream().get(), nc.which, uplo, 0, dummy.data(),
                           d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), R{0})
          << "norm=" << nc.c << " uplo=" << uplo_char(uplo);
    }
  }
}

TEST(LanspOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN anywhere in AP propagates through every norm, as DLANSP's DISNAN guard
// does for the max folds. Patched in after ?trttp, whose LAPACKE input check
// rejects a NaN.
template<typename T>
void expect_nan_propagates() {
  using R = typename elem<T>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const std::size_t n : {300u, 301u}) {
    for (const Uplo uplo : kUplos) {
      auto ap = make_ap<T>(uplo, n, 9);
      ap[ap.size() / 3] = elem<T>::make(elem<T>::kComplex ? 1.0 : nan, nan);
      auto d_ap = to_device(ap);
      DeviceBuffer<R> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lansp<T>(shared_device()->stream().get(), nc.which, uplo, n, d_ap.data(),
                             d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n;
      }
    }
  }
}

TEST(LanspOracleTests, NaNPropagates) {
  expect_nan_propagates<double>();
  expect_nan_propagates<wwr::wwrDoubleComplex>();
}

} // namespace
} // namespace calaman
