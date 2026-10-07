// Oracle test for calaman.lanhf -- the ?lanhf norm of a complex Hermitian
// matrix in Rectangular Full Packed storage. The oracle is the reference
// ?lanhf in the SAME precision on the host; LAPACKE has no ?lanhf wrapper and
// lapack.h no declaration, so the Fortran symbol is declared here (with its
// three hidden CHARACTER lengths). The wwr complex types are layout-compatible
// with lapack_complex_*.
//
// Every TRANSR x UPLO x n-parity case runs. ARF is built on the host by
// LAPACKE_?trttf from a random triangle whose diagonal carries the largest
// finite imaginary part -- ?lanhf reads only the real part, so a read of it
// overflows the result. The device copy carries trailing garbage (NaN and the
// largest finite value) past its n(n+1)/2 elements, RFP's stand-in for a
// padded leading dimension. |z| is an inexact hypot, so every norm takes the
// shared tolerance. Every suite stages data on the device, so it is
// REQUIRES_GPU (labeled `gpu`). Built only when calaman::lapack_reference
// exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lanhf;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// The reference ?lanhf, Fortran-mangled REAL / DOUBLE PRECISION functions.
extern "C" {
float clanhf_(const char *norm, const char *transr, const char *uplo, const lapack_int *n,
              const lapack_complex_float *a, float *work, std::size_t, std::size_t, std::size_t);
double zlanhf_(const char *norm, const char *transr, const char *uplo, const lapack_int *n,
               const lapack_complex_double *a, double *work, std::size_t, std::size_t, std::size_t);
}

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
constexpr Trans kTransrs[] = {Trans::N, Trans::C};

// Every LAPACK NORM char lanhf accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// Orders 1..5 (both parities, the n == 1 edge), around one block's stride, and
// many strides.
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

char trans_char(Trans t) {
  return t == Trans::N ? 'N' : 'C';
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
  static float ref(char norm, char transr, char uplo, std::size_t n, const C *arf) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return clanhf_(&norm, &transr, &uplo, &ni, reinterpret_cast<const lapack_complex_float *>(arf),
                   work.data(), 1, 1, 1);
  }
  static lapack_int trttf(char transr, char uplo, std::size_t n, const C *a, std::size_t lda,
                          C *arf) {
    return LAPACKE_ctrttf(LAPACK_COL_MAJOR, transr, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_float *>(a),
                          static_cast<lapack_int>(lda),
                          reinterpret_cast<lapack_complex_float *>(arf));
  }
};

template<>
struct complex_elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char transr, char uplo, std::size_t n, const C *arf) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return zlanhf_(&norm, &transr, &uplo, &ni, reinterpret_cast<const lapack_complex_double *>(arf),
                   work.data(), 1, 1, 1);
  }
  static lapack_int trttf(char transr, char uplo, std::size_t n, const C *a, std::size_t lda,
                          C *arf) {
    return LAPACKE_ztrttf(LAPACK_COL_MAJOR, transr, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_double *>(a),
                          static_cast<lapack_int>(lda),
                          reinterpret_cast<lapack_complex_double *>(arf));
  }
};

// The RFP array of a padded (lda = n + 3) triangle produced by @p fill(i, j),
// converted by the reference ?trttf.
template<typename C, typename Fill>
std::vector<C> to_arf(Trans transr, Uplo uplo, std::size_t n, Fill fill) {
  const std::size_t lda = n + 3;
  std::vector<C> a(lda * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      a[i + j * lda] = fill(i, j);
    }
  }
  std::vector<C> arf(n * (n + 1) / 2);
  const lapack_int info =
      complex_elem<C>::trttf(trans_char(transr), uplo_char(uplo), n, a.data(), lda, arf.data());
  EXPECT_EQ(info, 0);
  return arf;
}

// Random complex entries over a few binades; each diagonal entry a random real
// part and the largest finite imaginary part (never read).
template<typename C>
std::vector<C> make_arf(Trans transr, Uplo uplo, std::size_t n, std::uint32_t seed) {
  using R = typename complex_elem<C>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double big = std::numeric_limits<R>::max();
  return to_arf<C>(transr, uplo, n, [&](std::size_t i, std::size_t j) {
    const double re = dist(gen);
    return complex_elem<C>::make(re, i == j ? big : dist(gen));
  });
}

// ARF followed by trailing garbage the kernel must never read.
template<typename C>
std::vector<C> with_slack(std::vector<C> arf) {
  using R = typename complex_elem<C>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  arf.push_back(complex_elem<C>::make(nan, nan));
  arf.push_back(complex_elem<C>::make(std::numeric_limits<R>::max(), 0.0));
  return arf;
}

template<typename C>
void expect_matches_reference(const NormCase nc, Trans transr, Uplo uplo, std::size_t n) {
  using R = typename complex_elem<C>::R;
  const auto arf = make_arf<C>(transr, uplo, n, static_cast<std::uint32_t>(37 * n + 5));
  const R oracle = complex_elem<C>::ref(nc.c, trans_char(transr), uplo_char(uplo), n, arf.data());
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the diagonal's imaginary part";

  auto d_arf = to_device(with_slack(arf));
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lanhf<C>(shared_device()->stream().get(), nc.which, transr, uplo, n,
                            d_arf.data(), d_result.data());
  ASSERT_TRUE(s.ok()) << "lanhf returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, factorization_tol<R>(oracle, n, n))
      << "norm=" << nc.c << " transr=" << trans_char(transr) << " uplo=" << uplo_char(uplo)
      << " n=" << n;
}

template<typename C>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        for (const std::size_t n : kSizes) {
          expect_matches_reference<C>(nc, transr, uplo, n);
        }
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanhf, per type
// ========================================================================

TEST(LanhfOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanhfOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// A dominant diagonal decides the max norm: |Re A(j,j)| must be the answer,
// not |A(j,j)| -- a guard that the diagonal's imaginary part is dropped even
// where it would not overflow, in every layout.
TEST(LanhfOracleTests, MaxAbsIgnoresDiagonalImaginary) {
  using C = wwr::wwrDoubleComplex;
  for (const std::size_t n : {4u, 5u}) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        const auto arf = to_arf<C>(transr, uplo, n, [](std::size_t i, std::size_t j) {
          return i == 2 && j == 2 ? complex_elem<C>::make(-3.0, 4.0) // |z| = 5, |Re z| = 3
                                  : complex_elem<C>::make(0.5, 0.5);
        });
        const double oracle =
            complex_elem<C>::ref('M', trans_char(transr), uplo_char(uplo), n, arf.data());
        ASSERT_EQ(oracle, 3.0);
        auto d_arf = to_device(arf);
        DeviceBuffer<double> d_result(1, shared_device());
        ASSERT_TRUE(lanhf<C>(shared_device()->stream().get(), MatrixNorm::max_abs, transr, uplo, n,
                             d_arf.data(), d_result.data())
                        .ok());
        EXPECT_EQ(scalar_from_device(d_result), oracle)
            << "transr=" << trans_char(transr) << " uplo=" << uplo_char(uplo) << " n=" << n;
      }
    }
  }
}

// n == 0 writes a real 0 for every norm and layout, over a sentinel.
template<typename C>
void expect_empty_writes_zero() {
  using R = typename complex_elem<C>::R;
  auto dummy = to_device(std::vector<C>{complex_elem<C>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        auto d_result = to_device(std::vector<R>{R(-12345)});
        ASSERT_TRUE(lanhf<C>(shared_device()->stream().get(), nc.which, transr, uplo, 0,
                             dummy.data(), d_result.data())
                        .ok());
        EXPECT_EQ(scalar_from_device(d_result), R{0})
            << "norm=" << nc.c << " transr=" << trans_char(transr) << " uplo=" << uplo_char(uplo);
      }
    }
  }
}

TEST(LanhfOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// TRANSR = 'T' is not a complex RFP layout; ZLANHF accepts only 'N' and 'C'.
// Rejected before anything is enqueued, n == 0 included.
TEST(LanhfOracleTests, TransposeTransrIsInvalidValue) {
  const auto stream = shared_device()->stream().get();
  EXPECT_EQ(
      lanhf<wwr::wwrDoubleComplex>(stream, MatrixNorm::one, Trans::T, Uplo::U, 4, nullptr, nullptr),
      wwr::wwrErrorInvalidValue);
  EXPECT_EQ(lanhf<wwr::wwrFloatComplex>(stream, MatrixNorm::max_abs, Trans::T, Uplo::L, 0, nullptr,
                                        nullptr),
            wwr::wwrErrorInvalidValue);
}

// A NaN in an off-diagonal ARF entry propagates through every norm, as
// ZLANHF's DISNAN guard does for the max folds. Patched in after ?trttf, whose
// LAPACKE input check rejects a NaN; the diagonal slots are the ones holding
// the largest finite imaginary part, so the patch skips them.
TEST(LanhfOracleTests, NaNPropagates) {
  using C = wwr::wwrDoubleComplex;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<double>::max();
  for (const std::size_t n : {300u, 301u}) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        auto arf = make_arf<C>(transr, uplo, n, 9);
        std::size_t k = arf.size() / 3;
        while (std::abs(wwr::wwrCimag(arf[k])) == big) {
          ++k;
        }
        arf[k] = complex_elem<C>::make(1.0, nan);
        auto d_arf = to_device(arf);
        DeviceBuffer<double> d_result(1, shared_device());
        for (const NormCase nc : kNormCases) {
          ASSERT_TRUE(lanhf<C>(shared_device()->stream().get(), nc.which, transr, uplo, n,
                               d_arf.data(), d_result.data())
                          .ok());
          EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
              << "norm=" << nc.c << " transr=" << trans_char(transr) << " uplo=" << uplo_char(uplo)
              << " n=" << n;
        }
      }
    }
  }
}

} // namespace
} // namespace calaman
