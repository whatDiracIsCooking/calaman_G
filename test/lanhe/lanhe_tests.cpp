// Oracle test for calaman.lanhe -- the ?lanhe norm of a complex Hermitian
// matrix read from one triangle. The oracle is the reference ?lanhe
// (LAPACKE_{c,z}lanhe) in the SAME precision on the host, over the same buffer
// (the wwr complex types are layout-compatible with lapack_complex_*).
//
// Every case fills the triangle `uplo` does NOT name, and the lda > n padding
// rows, with garbage (NaN and the largest finite value), and gives every
// diagonal entry the largest finite imaginary part: ?lanhe reads only the real
// part of the diagonal, so a read of any of these turns the result into NaN or
// inf (or a wrong max) and fails the comparison. |z| is an inexact hypot, so
// every norm takes the shared tolerance. All suites stage data on the device,
// so they are REQUIRES_GPU (labeled `gpu`). Built only when
// calaman::lapack_reference exists; see this directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lanhe;
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

// Every LAPACK NORM char lanhe accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},
                                   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},
                                   {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}};

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

bool stored(Uplo uplo, std::size_t i, std::size_t j) {
  return uplo == Uplo::U ? i <= j : i >= j;
}

template<typename C>
struct complex_elem;

template<>
struct complex_elem<wwr::wwrFloatComplex> {
  using R = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, char uplo, std::size_t n, const wwr::wwrFloatComplex *a,
                   std::size_t lda) {
    return LAPACKE_clanhe(LAPACK_COL_MAJOR, norm, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_float *>(a),
                          static_cast<lapack_int>(lda));
  }
};

template<>
struct complex_elem<wwr::wwrDoubleComplex> {
  using R = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double ref(char norm, char uplo, std::size_t n, const wwr::wwrDoubleComplex *a,
                    std::size_t lda) {
    return LAPACKE_zlanhe(LAPACK_COL_MAJOR, norm, uplo, static_cast<lapack_int>(n),
                          reinterpret_cast<const lapack_complex_double *>(a),
                          static_cast<lapack_int>(lda));
  }
};

// The stored off-diagonal holds random complex entries over a few binades; the
// diagonal a random real part and the largest finite imaginary part (never
// read); every other slot alternates NaN and the largest finite component.
// Finite, not NaN, on the diagonal: LAPACKE's input NaN check scans it.
template<typename C>
std::vector<C> make_matrix(Uplo uplo, std::size_t n, std::size_t lda, std::uint32_t seed) {
  using R = typename complex_elem<C>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<R>::max();
  std::vector<C> a(lda * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      const bool in = i < n && stored(uplo, i, j);
      a[i + j * lda] = i == j             ? complex_elem<C>::make(dist(gen), big)
                       : in               ? complex_elem<C>::make(dist(gen), dist(gen))
                       : (i + j) % 2 == 0 ? complex_elem<C>::make(nan, nan)
                                          : complex_elem<C>::make(big, big);
    }
  }
  return a;
}

template<typename C>
void expect_matches_reference(const NormCase nc, Uplo uplo, std::size_t n, std::size_t lda) {
  using R = typename complex_elem<C>::R;
  const auto a = make_matrix<C>(uplo, n, lda, static_cast<std::uint32_t>(37 * n + 13 * lda + 5));
  const R oracle = complex_elem<C>::ref(nc.c, uplo_char(uplo), n, a.data(), lda);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_A = to_device(a);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lanhe<C>(shared_device()->stream().get(), nc.which, uplo, n, d_A.data(), lda,
                            d_result.data());
  ASSERT_TRUE(s.ok()) << "lanhe returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, factorization_tol<R>(oracle, n, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n << " lda=" << lda;
}

template<typename C>
void run_sizes() {
  // n = 1 edge (the diagonal alone), small orders, around one block's stride,
  // and many strides, each with lda == n and a padded lda > n.
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const std::size_t n : {1u, 2u, 3u, 17u, 255u, 257u, 600u}) {
        expect_matches_reference<C>(nc, uplo, n, n);
        expect_matches_reference<C>(nc, uplo, n, n + 5);
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanhe, per type
// ========================================================================

TEST(LanheOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanheOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// A dominant real diagonal decides the max norm: |Re A(j,j)| must be the
// answer, not |A(j,j)| -- a guard that the diagonal's imaginary part is dropped
// even where it would not overflow.
TEST(LanheOracleTests, MaxAbsIgnoresDiagonalImaginary) {
  using C = wwr::wwrDoubleComplex;
  const std::size_t n = 4;
  for (const Uplo uplo : kUplos) {
    std::vector<C> a(n * n, complex_elem<C>::make(0.5, 0.5));
    a[2 + 2 * n] = complex_elem<C>::make(-3.0, 4.0); // |z| = 5, |Re z| = 3
    const double oracle = complex_elem<C>::ref('M', uplo_char(uplo), n, a.data(), n);
    ASSERT_EQ(oracle, 3.0);
    auto d_A = to_device(a);
    DeviceBuffer<double> d_result(1, shared_device());
    ASSERT_TRUE(lanhe<C>(shared_device()->stream().get(), MatrixNorm::max_abs, uplo, n,
                         d_A.data(), n, d_result.data())
                    .ok());
    EXPECT_EQ(scalar_from_device(d_result), oracle) << "uplo=" << uplo_char(uplo);
  }
}

// n == 0 writes a real 0 for every norm and both triangles, over a sentinel.
template<typename C>
void expect_empty_writes_zero() {
  using R = typename complex_elem<C>::R;
  auto dummy = to_device(std::vector<C>{complex_elem<C>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      auto d_result = to_device(std::vector<R>{R(-12345)});
      ASSERT_TRUE(lanhe<C>(shared_device()->stream().get(), nc.which, uplo, 0, dummy.data(), 1,
                           d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), R{0})
          << "norm=" << nc.c << " uplo=" << uplo_char(uplo);
    }
  }
}

TEST(LanheOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in the STORED off-diagonal propagates through every norm, as ZLANHE's
// DISNAN guard does for the max folds.
TEST(LanheOracleTests, NaNInStoredTrianglePropagates) {
  using C = wwr::wwrDoubleComplex;
  const std::size_t n = 300;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const Uplo uplo : kUplos) {
    auto a = make_matrix<C>(uplo, n, n, 9);
    const std::size_t off = uplo == Uplo::U ? 2 + 150 * n : 150 + 2 * n; // (2,150) / (150,2)
    a[off] = complex_elem<C>::make(1.0, nan);
    auto d_A = to_device(a);
    DeviceBuffer<double> d_result(1, shared_device());
    for (const NormCase nc : kNormCases) {
      ASSERT_TRUE(lanhe<C>(shared_device()->stream().get(), nc.which, uplo, n, d_A.data(), n,
                           d_result.data())
                      .ok());
      EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
          << "norm=" << nc.c << " uplo=" << uplo_char(uplo);
    }
  }
}

} // namespace
} // namespace calaman
