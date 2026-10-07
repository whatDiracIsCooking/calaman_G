// Oracle test for calaman.lantr -- the ?lantr norm of an m-by-n upper or lower
// trapezoid, s/d/c/z. The oracle is the reference ?lantr in the SAME precision
// on the host, called through lapack.h's LAPACK_?lantr (the lansb suite's
// route). The wwr complex types are layout-compatible with lapack_complex_*.
//
// Both UPLO and both DIAG values run, on square, wide and tall shapes. Every
// slot of A that the routine must not read -- the opposite triangle, the lda > m
// padding rows and, for Diag::U, the diagonal -- carries garbage (NaN and the
// largest finite value): a single read of one turns the result into NaN or inf
// and fails the comparison. The Fortran routine has no input NaN check, so the
// oracle accepts that garbage. |x| is exact for a real T, so the real max norm
// is compared exactly; every other norm takes the shared tolerance. Every suite
// stages data on the device, so it is REQUIRES_GPU (labeled `gpu`). Built only
// when calaman::lapack_reference exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lantr;
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

// Every LAPACK NORM char lantr accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// The 1-by-1 edge, tiny squares, wide and tall trapezoids, single rows and
// columns longer than one block's 256 threads, and a large square.
constexpr std::pair<std::size_t, std::size_t> kShapes[] = {
    {1, 1}, {2, 2}, {3, 5}, {5, 3}, {17, 17}, {1, 300}, {300, 1}, {255, 400}, {400, 255}, {700, 700}};

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

// Per-type construction and the reference ?lantr.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool kComplex = false;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, char uplo, char diag, lapack_int m, lapack_int n, const float *a,
                   lapack_int lda, float *work) {
    return static_cast<float>(LAPACK_slantr(&norm, &uplo, &diag, &m, &n, a, &lda, work));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool kComplex = false;
  static double make(double re, double) { return re; }
  static double ref(char norm, char uplo, char diag, lapack_int m, lapack_int n, const double *a,
                    lapack_int lda, double *work) {
    return LAPACK_dlantr(&norm, &uplo, &diag, &m, &n, a, &lda, work);
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
  static float ref(char norm, char uplo, char diag, lapack_int m, lapack_int n, const C *a,
                   lapack_int lda, float *work) {
    return static_cast<float>(LAPACK_clantr(&norm, &uplo, &diag, &m, &n,
                                            reinterpret_cast<const lapack_complex_float *>(a),
                                            &lda, work));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static constexpr bool kComplex = true;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char uplo, char diag, lapack_int m, lapack_int n, const C *a,
                    lapack_int lda, double *work) {
    return LAPACK_zlantr(&norm, &uplo, &diag, &m, &n,
                         reinterpret_cast<const lapack_complex_double *>(a), &lda, work);
  }
};

template<typename T>
typename elem<T>::R reference(char norm, Uplo uplo, Diag diag, std::size_t m, std::size_t n,
                              const std::vector<T> &a, std::size_t lda) {
  std::vector<typename elem<T>::R> work(std::max<std::size_t>(m, 1));
  return elem<T>::ref(norm, uplo_char(uplo), diag_char(diag), static_cast<lapack_int>(m),
                      static_cast<lapack_int>(n), a.data(), static_cast<lapack_int>(lda),
                      work.data());
}

// Whether lantr may read A(i,j): inside the m-by-n trapezoid, off the diagonal
// when it is implicitly unit.
bool is_read(Uplo uplo, Diag diag, std::size_t i, std::size_t j, std::size_t m) {
  if (i >= m || (diag == Diag::U && i == j)) {
    return false;
  }
  return uplo == Uplo::U ? i <= j : i >= j;
}

// Trapezoid entries are mixed-sign values over a few binades (complex: both
// parts); every other slot of the lda-by-n array alternates NaN and the largest
// finite component.
template<typename T>
std::vector<T> make_matrix(Uplo uplo, Diag diag, std::size_t m, std::size_t n, std::size_t lda,
                           std::uint32_t seed) {
  using R = typename elem<T>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<R>::max();
  std::vector<T> a(lda * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      const double re = dist(gen);
      const double im = dist(gen);
      a[i + j * lda] = is_read(uplo, diag, i, j, m) ? elem<T>::make(re, im)
                       : (i + j) % 2 == 0           ? elem<T>::make(nan, nan)
                                                    : elem<T>::make(big, big);
    }
  }
  return a;
}

// |x| is exact for a real T, so the real max norm is compared exactly; the
// sums, and every complex norm (|z| is an inexact hypot), take the shared
// tolerance over the longer dimension.
template<typename T>
typename elem<T>::R norm_tol(MatrixNorm which, typename elem<T>::R ref, std::size_t m,
                             std::size_t n) {
  using R = typename elem<T>::R;
  const std::size_t len = std::max(m, n);
  return (!elem<T>::kComplex && which == MatrixNorm::max_abs) ? R{0}
                                                              : factorization_tol<R>(ref, len, len);
}

template<typename T>
void expect_matches_reference(const NormCase nc, Uplo uplo, Diag diag, std::size_t m,
                              std::size_t n, std::size_t lda) {
  using R = typename elem<T>::R;
  const auto a =
      make_matrix<T>(uplo, diag, m, n, lda, static_cast<std::uint32_t>(31 * m + 7 * n + lda));
  const R oracle = reference<T>(nc.c, uplo, diag, m, n, a, lda);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_A = to_device(a);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lantr<T>(shared_device()->stream().get(), nc.which, uplo, diag, m, n,
                            d_A.data(), lda, d_result.data());
  ASSERT_TRUE(s.ok()) << "lantr returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol<T>(nc.which, oracle, m, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " diag=" << diag_char(diag)
      << " m=" << m << " n=" << n << " lda=" << lda;
}

// Each case with the tight lda == m and a padded one.
template<typename T>
void run_shapes() {
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const Diag diag : kDiags) {
        for (const auto &[m, n] : kShapes) {
          expect_matches_reference<T>(nc, uplo, diag, m, n, m);
          expect_matches_reference<T>(nc, uplo, diag, m, n, m + 3);
        }
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lantr, per type
// ========================================================================

TEST(LantrOracleTests, Float) {
  run_shapes<float>();
}
TEST(LantrOracleTests, Double) {
  run_shapes<double>();
}
TEST(LantrOracleTests, ComplexFloat) {
  run_shapes<wwr::wwrFloatComplex>();
}
TEST(LantrOracleTests, ComplexDouble) {
  run_shapes<wwr::wwrDoubleComplex>();
}

// m == 0 or n == 0 writes 0 for every norm, triangle and diagonal, over a sentinel.
template<typename T>
void expect_empty_writes_zero() {
  using R = typename elem<T>::R;
  auto dummy = to_device(std::vector<T>{elem<T>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const Diag diag : kDiags) {
        for (const auto &[m, n] : {std::pair<std::size_t, std::size_t>{0, 0}, {0, 4}, {4, 0}}) {
          auto d_result = to_device(std::vector<R>{R(-12345)});
          ASSERT_TRUE(lantr<T>(shared_device()->stream().get(), nc.which, uplo, diag, m, n,
                               dummy.data(), 4, d_result.data())
                          .ok());
          EXPECT_EQ(scalar_from_device(d_result), R{0})
              << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " m=" << m << " n=" << n;
        }
      }
    }
  }
}

TEST(LantrOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN inside the trapezoid -- on the diagonal (Diag::N) or at the far corner
// of a column -- propagates through every norm, as DLANTR's DISNAN guard does.
template<typename T>
void expect_nan_propagates() {
  using R = typename elem<T>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t m = 300;
  const std::size_t n = 260;
  const std::size_t lda = m + 2;
  for (const Uplo uplo : kUplos) {
    const std::size_t corner_row = uplo == Uplo::U ? 0 : m - 1;
    for (const auto &[i, j] : {std::pair<std::size_t, std::size_t>{77, 77}, {corner_row, 150}}) {
      auto a = make_matrix<T>(uplo, Diag::N, m, n, lda, 9);
      a[i + j * lda] = elem<T>::make(elem<T>::kComplex ? 1.0 : nan, nan);
      auto d_A = to_device(a);
      DeviceBuffer<R> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lantr<T>(shared_device()->stream().get(), nc.which, uplo, Diag::N, m, n,
                             d_A.data(), lda, d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " i=" << i << " j=" << j;
      }
    }
  }
}

TEST(LantrOracleTests, NaNPropagates) {
  expect_nan_propagates<double>();
  expect_nan_propagates<wwr::wwrDoubleComplex>();
}

} // namespace
} // namespace calaman
