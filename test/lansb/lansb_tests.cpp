// Oracle test for calaman.lansb -- the ?lansb norm of a symmetric band matrix
// in LAPACK band storage, s/d/c/z (complex symmetric, not Hermitian). The
// oracle is the reference ?lansb in the SAME precision on the host, called
// through lapack.h's LAPACK_?lansb (LAPACKE has no ?lansb wrapper). The wwr
// complex types are layout-compatible with lapack_complex_*.
//
// Both UPLO values run. Every slot of AB that holds no in-matrix band entry of
// the stored triangle -- the unused corner and the ldab > k+1 padding rows --
// carries garbage (NaN and the largest finite value): a single read of one
// turns the result into NaN or inf and fails the comparison. The Fortran
// routine has no input NaN check, so the oracle accepts that garbage. |x| is
// exact for a real T, so the real max norm is compared exactly; every other
// norm takes the shared tolerance. Every suite stages data on the device, so it
// is REQUIRES_GPU (labeled `gpu`). Built only when calaman::lapack_reference
// exists; see CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lansb;
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

// Every LAPACK NORM char lansb accepts, with the MatrixNorm it maps to ('O' is
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

// Per-type construction and the reference ?lansb.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool kComplex = false;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, char uplo, std::size_t n, std::size_t k, const float *ab,
                   std::size_t ldab) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ki = static_cast<lapack_int>(k);
    const lapack_int ldi = static_cast<lapack_int>(ldab);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return static_cast<float>(LAPACK_slansb(&norm, &uplo, &ni, &ki, ab, &ldi, work.data()));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool kComplex = false;
  static double make(double re, double) { return re; }
  static double ref(char norm, char uplo, std::size_t n, std::size_t k, const double *ab,
                    std::size_t ldab) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ki = static_cast<lapack_int>(k);
    const lapack_int ldi = static_cast<lapack_int>(ldab);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return LAPACK_dlansb(&norm, &uplo, &ni, &ki, ab, &ldi, work.data());
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
  static float ref(char norm, char uplo, std::size_t n, std::size_t k, const C *ab,
                   std::size_t ldab) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ki = static_cast<lapack_int>(k);
    const lapack_int ldi = static_cast<lapack_int>(ldab);
    std::vector<float> work(std::max<std::size_t>(n, 1));
    return static_cast<float>(LAPACK_clansb(&norm, &uplo, &ni, &ki,
                                            reinterpret_cast<const lapack_complex_float *>(ab),
                                            &ldi, work.data()));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static constexpr bool kComplex = true;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, char uplo, std::size_t n, std::size_t k, const C *ab,
                    std::size_t ldab) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ki = static_cast<lapack_int>(k);
    const lapack_int ldi = static_cast<lapack_int>(ldab);
    std::vector<double> work(std::max<std::size_t>(n, 1));
    return LAPACK_zlansb(&norm, &uplo, &ni, &ki,
                         reinterpret_cast<const lapack_complex_double *>(ab), &ldi, work.data());
  }
};

// Row r of AB column j holds A(r+j-k, j) (Uplo::U) or A(r+j, j) (Uplo::L): an
// in-matrix band entry of the stored triangle iff r <= k and that row lies in
// [0, n).
bool in_band(Uplo uplo, std::size_t r, std::size_t j, std::size_t n, std::size_t k) {
  return r <= k && (uplo == Uplo::U ? r + j >= k : r + j < n);
}

// Band entries are mixed-sign values over a few binades (complex: both parts);
// every other slot of the ldab-by-n array alternates NaN and the largest finite
// component.
template<typename T>
std::vector<T> make_band(Uplo uplo, std::size_t n, std::size_t k, std::size_t ldab,
                         std::uint32_t seed) {
  using R = typename elem<T>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<R>::max();
  std::vector<T> ab(ldab * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t r = 0; r < ldab; ++r) {
      const double re = dist(gen);
      const double im = dist(gen);
      ab[r + j * ldab] = in_band(uplo, r, j, n, k) ? elem<T>::make(re, im)
                         : (r + j) % 2 == 0        ? elem<T>::make(nan, nan)
                                                   : elem<T>::make(big, big);
    }
  }
  return ab;
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
void expect_matches_reference(const NormCase nc, Uplo uplo, std::size_t n, std::size_t k,
                              std::size_t ldab) {
  using R = typename elem<T>::R;
  const auto ab =
      make_band<T>(uplo, n, k, ldab, static_cast<std::uint32_t>(31 * n + 7 * ldab + 5 * k + 1));
  const R oracle = elem<T>::ref(nc.c, uplo_char(uplo), n, k, ab.data(), ldab);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_AB = to_device(ab);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lansb<T>(shared_device()->stream().get(), nc.which, uplo, n, k, d_AB.data(),
                            ldab, d_result.data());
  ASSERT_TRUE(s.ok()) << "lansb returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol<T>(nc.which, oracle, n))
      << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " n=" << n << " k=" << k
      << " ldab=" << ldab;
}

// Each case with the tight ldab == k+1 and a padded one.
template<typename T>
void run_sizes() {
  for (const NormCase nc : kNormCases) {
    for (const Uplo uplo : kUplos) {
      for (const std::size_t k : kBandwidths) {
        for (const std::size_t n : kSizes) {
          expect_matches_reference<T>(nc, uplo, n, k, k + 1);
          expect_matches_reference<T>(nc, uplo, n, k, k + 4);
        }
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lansb, per type
// ========================================================================

TEST(LansbOracleTests, Float) {
  run_sizes<float>();
}
TEST(LansbOracleTests, Double) {
  run_sizes<double>();
}
TEST(LansbOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LansbOracleTests, ComplexDouble) {
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
      ASSERT_TRUE(lansb<T>(shared_device()->stream().get(), nc.which, uplo, 0, 1, dummy.data(), 2,
                           d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), R{0})
          << "norm=" << nc.c << " uplo=" << uplo_char(uplo);
    }
  }
}

TEST(LansbOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in the stored band -- on the diagonal or the outermost stored diagonal
// -- propagates through every norm, as DLANSB's DISNAN guard does for the max.
template<typename T>
void expect_nan_propagates() {
  using R = typename elem<T>::R;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::size_t n = 300;
  const std::size_t k = 4;
  const std::size_t ldab = k + 3;
  for (const Uplo uplo : kUplos) {
    // AB slots (row, column): the diagonal of column 77 and the outermost
    // stored diagonal of column 150.
    const std::size_t diag_row = uplo == Uplo::U ? k : 0;
    const std::size_t edge_row = uplo == Uplo::U ? 0 : k;
    for (const auto &[r, j] : {std::pair{diag_row, 77u}, std::pair{edge_row, 150u}}) {
      auto ab = make_band<T>(uplo, n, k, ldab, 9);
      ab[r + j * ldab] = elem<T>::make(elem<T>::kComplex ? 1.0 : nan, nan);
      auto d_AB = to_device(ab);
      DeviceBuffer<R> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lansb<T>(shared_device()->stream().get(), nc.which, uplo, n, k, d_AB.data(),
                             ldab, d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " uplo=" << uplo_char(uplo) << " r=" << r << " j=" << j;
      }
    }
  }
}

TEST(LansbOracleTests, NaNPropagates) {
  expect_nan_propagates<double>();
  expect_nan_propagates<wwr::wwrDoubleComplex>();
}

} // namespace
} // namespace calaman
