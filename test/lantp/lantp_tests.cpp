// Oracle test for calaman.lantp -- the ?lantp norm of a triangular matrix in
// packed storage, s/d/c/z. The oracle is the reference ?lantp in the SAME
// precision on the host, called through lapack.h's LAPACK_?lantp (the lansp
// suite's route). The wwr complex types are layout-compatible with
// lapack_complex_*.
//
// Both UPLO and both DIAG values run. For Diag::U every diagonal slot of AP
// carries garbage (NaN and the largest finite value), and so does a short tail
// past the n(n+1)/2 packed elements: a single read of one turns the result into
// NaN or inf and fails the comparison. |x| is exact for a real T, so the real
// max norm is compared exactly; every other norm takes the shared tolerance.
// Every suite stages data on the device, so it is REQUIRES_GPU (labeled `gpu`).
// Built only when calaman::lapack_reference exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lantp;
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
constexpr Diag kDiags[] = {Diag::N, Diag::U};

// Every LAPACK NORM char lantp accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// The n == 1 edge, small orders, around one block's stride, and many strides.
constexpr std::size_t kSizes[] = {1, 2, 3, 17, 255, 257, 700};

// Garbage slots appended past the packed triangle.
constexpr std::size_t kTail = 3;

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
char diag_char(Diag diag) {
  return diag == Diag::U ? 'U' : 'N';
}

// Per-type construction and the reference ?lantp.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool kComplex = false;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, char uplo, char diag, lapack_int n, const float *ap, float *work) {
    return static_cast<float>(LAPACK_slantp(&norm, &uplo, &diag, &n, ap, work));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool kComplex = false;
  static double make(double re, double) { return re; }
  static double ref(char norm, char uplo, char diag, lapack_int n, const double *ap,
                    double *work) {
    return LAPACK_dlantp(&norm, &uplo, &diag, &n, ap, work);
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
  static float ref(char norm, char uplo, char diag, lapack_int n, const C *ap, float *work) {
    return static_cast<float>(LAPACK_clantp(
        &norm, &uplo, &diag, &n, reinterpret_cast<const lapack_complex_float *>(ap), work));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static constexpr bool kComplex = true;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char uplo, char diag, lapack_int n, const C *ap, double *work) {
    return LAPACK_zlantp(&norm, &uplo, &diag, &n,
                         reinterpret_cast<const lapack_complex_double *>(ap), work);
  }
};

// AP slot of the diagonal entry A(j,j).
std::size_t diag_slot(Uplo uplo, std::size_t n, std::size_t j) {
  return uplo == Uplo::U ? j + j * (j + 1) / 2 : j + j * (2 * n - j - 1) / 2;
}

// Packed entries are mixed-sign values over a few binades (complex: both
// parts); the Diag::U diagonal slots and the tail alternate NaN and the largest
// finite component.
template<typename T>
std::vector<T> make_packed(Uplo uplo, Diag diag, std::size_t n, std::uint32_t seed) {
  using R = typename elem<T>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<R>::max();
  const std::size_t len = n * (n + 1) / 2;
  std::vector<T> ap(len + kTail);
  for (std::size_t s = 0; s < ap.size(); ++s) {
    const double re = dist(gen);
    const double im = dist(gen);
    ap[s] = elem<T>::make(re, im);
  }
  const auto poison = [&](std::size_t s) {
    ap[s] = s % 2 == 0 ? elem<T>::make(nan, nan) : elem<T>::make(big, big);
  };
  for (std::size_t s = len; s < ap.size(); ++s) {
    poison(s);
  }
  if (diag == Diag::U) {
    for (std::size_t j = 0; j < n; ++j) {
      poison(diag_slot(uplo, n, j));
    }
  }
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
void expect_matches_reference(const NormCase nc, Uplo uplo, Diag diag, std::size_t n) {
  using R = typename elem<T>::R;
  const auto ap = make_packed<T>(uplo, diag, n, static_cast<std::uint32_t>(31 * n + 1));
  std::vector<R> work(std::max<std::size_t>(n, 1));
  const R oracle = elem<T>::ref(nc.c, uplo_char(uplo), diag_char(diag),
                                static_cast<lapack_int>(n), ap.data(), work.data());
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_AP = to_device(ap);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lantp<T>(shared_device()->stream().get(), nc.which, uplo, diag, n, d_AP.data(),
                            d_result.data());
  ASSERT_TRUE(s.ok()) << "lantp returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol<T>(nc.which, oracle, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " diag=" << diag_char(diag)
      << " n=" << n;
}

template<typename T>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const Diag diag : kDiags) {
        for (const std::size_t n : kSizes) {
          expect_matches_reference<T>(nc, uplo, diag, n);
        }
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lantp, per type
// ========================================================================

TEST(LantpOracleTests, Float) {
  run_sizes<float>();
}
TEST(LantpOracleTests, Double) {
  run_sizes<double>();
}
TEST(LantpOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LantpOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// n == 0 writes 0 for every norm, triangle and diagonal, over a sentinel.
template<typename T>
void expect_empty_writes_zero() {
  using R = typename elem<T>::R;
  auto dummy = to_device(std::vector<T>{elem<T>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const Diag diag : kDiags) {
        auto d_result = to_device(std::vector<R>{R(-12345)});
        ASSERT_TRUE(lantp<T>(shared_device()->stream().get(), nc.which, uplo, diag, 0,
                             dummy.data(), d_result.data())
                        .ok());
        EXPECT_EQ(scalar_from_device(d_result), R{0})
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " diag=" << diag_char(diag);
      }
    }
  }
}

TEST(LantpOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in the packed triangle -- a diagonal slot (Diag::N) or the last slot --
// propagates through every norm, as DLANTP's DISNAN guard does for the max.
template<typename T>
void expect_nan_propagates() {
  using R = typename elem<T>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t n = 300;
  for (const Uplo uplo : kUplos) {
    for (const std::size_t s : {diag_slot(uplo, n, 77), n * (n + 1) / 2 - 1}) {
      auto ap = make_packed<T>(uplo, Diag::N, n, 9);
      ap[s] = elem<T>::make(elem<T>::kComplex ? 1.0 : nan, nan);
      auto d_AP = to_device(ap);
      DeviceBuffer<R> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lantp<T>(shared_device()->stream().get(), nc.which, uplo, Diag::N, n,
                             d_AP.data(), d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " slot=" << s;
      }
    }
  }
}

TEST(LantpOracleTests, NaNPropagates) {
  expect_nan_propagates<double>();
  expect_nan_propagates<wwr::wwrDoubleComplex>();
}

} // namespace
} // namespace calaman
