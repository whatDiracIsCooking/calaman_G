// Oracle test for calaman.lansf -- the ?lansf norm of a real symmetric matrix in
// Rectangular Full Packed storage. The oracle is the reference ?lansf in the
// SAME precision on the host; LAPACKE has no ?lansf wrapper and lapack.h no
// declaration, so the Fortran symbol is declared here (with its three hidden
// CHARACTER lengths).
//
// Every TRANSR x UPLO x n-parity case runs. ARF is built on the host by
// LAPACKE_?trttf from a random triangle, and the device copy carries trailing
// garbage (NaN and the largest finite value) past its n(n+1)/2 elements, so an
// over-read turns the result into NaN or a wrong max and fails the comparison.
// RFP has no leading dimension; that trailing slack is its stand-in. Every
// suite stages data on the device, so it is REQUIRES_GPU (labeled `gpu`).
// Built only when calaman::lapack_reference exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lansf;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// The reference ?lansf, Fortran-mangled REAL / DOUBLE PRECISION functions.
extern "C" {
float slansf_(const char *norm, const char *transr, const char *uplo, const lapack_int *n,
              const float *a, float *work, std::size_t, std::size_t, std::size_t);
double dlansf_(const char *norm, const char *transr, const char *uplo, const lapack_int *n,
               const double *a, double *work, std::size_t, std::size_t, std::size_t);
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
constexpr Trans kTransrs[] = {Trans::N, Trans::T};

// Every LAPACK NORM char lansf accepts, with the MatrixNorm it maps to ('O' is
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
  return t == Trans::N ? 'N' : 'T';
}

template<typename T>
struct elem;

template<>
struct elem<float> {
  static float ref(char norm, char transr, char uplo, std::size_t n, const float *arf) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return slansf_(&norm, &transr, &uplo, &ni, arf, work.data(), 1, 1, 1);
  }
  static lapack_int trttf(char transr, char uplo, std::size_t n, const float *a, std::size_t lda,
                          float *arf) {
    return LAPACKE_strttf(LAPACK_COL_MAJOR, transr, uplo, static_cast<lapack_int>(n), a,
                          static_cast<lapack_int>(lda), arf);
  }
};

template<>
struct elem<double> {
  static double ref(char norm, char transr, char uplo, std::size_t n, const double *arf) {
    const lapack_int ni = static_cast<lapack_int>(n);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return dlansf_(&norm, &transr, &uplo, &ni, arf, work.data(), 1, 1, 1);
  }
  static lapack_int trttf(char transr, char uplo, std::size_t n, const double *a, std::size_t lda,
                          double *arf) {
    return LAPACKE_dtrttf(LAPACK_COL_MAJOR, transr, uplo, static_cast<lapack_int>(n), a,
                          static_cast<lapack_int>(lda), arf);
  }
};

// The n(n+1)/2-element RFP array of a random symmetric matrix over a few
// binades, built by the reference ?trttf from a padded (lda = n + 3) triangle.
template<typename T>
std::vector<T> make_arf(Trans transr, Uplo uplo, std::size_t n, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const std::size_t lda = n + 3;
  std::vector<T> a(lda * n);
  for (T &x : a) {
    x = static_cast<T>(dist(gen));
  }
  std::vector<T> arf(n * (n + 1) / 2);
  const lapack_int info =
      elem<T>::trttf(trans_char(transr), uplo_char(uplo), n, a.data(), lda, arf.data());
  EXPECT_EQ(info, 0);
  return arf;
}

// ARF followed by trailing garbage the kernel must never read.
template<typename T>
std::vector<T> with_slack(std::vector<T> arf) {
  arf.push_back(std::numeric_limits<T>::quiet_NaN());
  arf.push_back(std::numeric_limits<T>::max());
  return arf;
}

// |x| is exact for a real T, so the max norm is compared exactly; the sums
// take the shared tolerance.
template<typename T>
T norm_tol(MatrixNorm which, T ref, std::size_t n) {
  return which == MatrixNorm::max_abs ? T{0} : factorization_tol<T>(ref, n, n);
}

template<typename T>
void expect_matches_reference(const NormCase nc, Trans transr, Uplo uplo, std::size_t n) {
  const auto arf = make_arf<T>(transr, uplo, n, static_cast<std::uint32_t>(37 * n + 5));
  const T oracle = elem<T>::ref(nc.c, trans_char(transr), uplo_char(uplo), n, arf.data());
  ASSERT_TRUE(std::isfinite(oracle));

  auto d_arf = to_device(with_slack(arf));
  DeviceBuffer<T> d_result(1, shared_device());
  const Status s = lansf<T>(shared_device()->stream().get(), nc.which, transr, uplo, n,
                            d_arf.data(), d_result.data());
  ASSERT_TRUE(s.ok()) << "lansf returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol(nc.which, oracle, n))
      << "norm=" << nc.c << " transr=" << trans_char(transr) << " uplo=" << uplo_char(uplo)
      << " n=" << n;
}

template<typename T>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        for (const std::size_t n : kSizes) {
          expect_matches_reference<T>(nc, transr, uplo, n);
        }
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lansf, per type
// ========================================================================

TEST(LansfOracleTests, Float) {
  run_sizes<float>();
}
TEST(LansfOracleTests, Double) {
  run_sizes<double>();
}

// n == 0 writes 0 for every norm and layout, over a sentinel.
TEST(LansfOracleTests, EmptyWritesZero) {
  auto dummy = to_device(std::vector<double>{42.0});
  for (const NormCase nc : kNormCases) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        auto d_result = to_device(std::vector<double>{-12345.0});
        ASSERT_TRUE(lansf<double>(shared_device()->stream().get(), nc.which, transr, uplo, 0,
                                  dummy.data(), d_result.data())
                        .ok());
        EXPECT_EQ(scalar_from_device(d_result), 0.0)
            << "norm=" << nc.c << " transr=" << trans_char(transr) << " uplo=" << uplo_char(uplo);
      }
    }
  }
}

// TRANSR = 'C' is not a real RFP layout; DLANSF accepts only 'N' and 'T'.
// Rejected before anything is enqueued, n == 0 included.
TEST(LansfOracleTests, ConjugateTransrIsInvalidValue) {
  const auto stream = shared_device()->stream().get();
  EXPECT_EQ(lansf<double>(stream, MatrixNorm::one, Trans::C, Uplo::U, 4, nullptr, nullptr),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(lansf<float>(stream, MatrixNorm::max_abs, Trans::C, Uplo::L, 0, nullptr, nullptr),
            wwr::wwrErrorInvalidValue);
}

// A NaN anywhere in ARF propagates through every norm, as DLANSF's DISNAN
// guard does for the max folds.
TEST(LansfOracleTests, NaNPropagates) {
  for (const std::size_t n : {300u, 301u}) {
    for (const Trans transr : kTransrs) {
      for (const Uplo uplo : kUplos) {
        auto arf = make_arf<double>(transr, uplo, n, 9);
        arf[arf.size() / 3] = std::numeric_limits<double>::quiet_NaN();
        auto d_arf = to_device(arf);
        DeviceBuffer<double> d_result(1, shared_device());
        for (const NormCase nc : kNormCases) {
          ASSERT_TRUE(lansf<double>(shared_device()->stream().get(), nc.which, transr, uplo, n,
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
