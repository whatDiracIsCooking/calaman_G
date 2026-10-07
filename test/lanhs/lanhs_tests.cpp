// Oracle test for calaman.lanhs -- the ?lanhs norm of an upper Hessenberg
// matrix, s/d/c/z. The oracle is the reference ?lanhs (LAPACK_?lanhs from
// lapack.h, which lapacke.h includes; LAPACKE has no ?lanhs wrapper) in the SAME
// precision on the host, over the same buffer (the wwr complex types are
// layout-compatible with lapack_complex_*).
//
// Every case fills the entries below the subdiagonal, and the lda > n padding
// rows, with garbage (NaN and the largest finite value): a single read of either
// would turn the result into NaN or inf and fail the comparison. All suites
// stage data on the device, so they are REQUIRES_GPU (labeled `gpu`). Built only
// when calaman::lapack_reference exists; see this directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lanhs;
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

// Per-type element construction and the same-precision reference ?lanhs.
// `exact_max` is whether the max norm is a pure compare-and-copy (real T), so
// it must match bit for bit; a complex |z| is an inexact hypot.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool exact_max = true;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, lapack_int n, const float *a, lapack_int lda, float *work) {
    return static_cast<float>(LAPACK_slanhs(&norm, &n, a, &lda, work));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool exact_max = true;
  static double make(double re, double) { return re; }
  static double ref(char norm, lapack_int n, const double *a, lapack_int lda, double *work) {
    return LAPACK_dlanhs(&norm, &n, a, &lda, work);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static constexpr bool exact_max = false;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, lapack_int n, const wwr::wwrFloatComplex *a, lapack_int lda,
                   float *work) {
    return static_cast<float>(LAPACK_clanhs(
        &norm, &n, reinterpret_cast<const lapack_complex_float *>(a), &lda, work));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static constexpr bool exact_max = false;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double ref(char norm, lapack_int n, const wwr::wwrDoubleComplex *a, lapack_int lda,
                    double *work) {
    return LAPACK_zlanhs(&norm, &n, reinterpret_cast<const lapack_complex_double *>(a), &lda,
                         work);
  }
};

// Every NORM char DLANHS accepts, with the MatrixNorm it maps to ('O' and 'E'
// are the 1-norm's and Frobenius norm's synonyms).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs}, {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},     {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

template<typename T>
typename elem<T>::R reference(char norm, std::size_t n, const std::vector<T> &a,
                              std::size_t lda) {
  std::vector<typename elem<T>::R> work(n == 0 ? 1 : n);
  return elem<T>::ref(norm, static_cast<lapack_int>(n), a.data(), static_cast<lapack_int>(lda),
                      work.data());
}

// The referenced part (i <= j+1) holds mixed-sign entries over a few binades;
// every other slot of the lda-by-n array (below the subdiagonal, and the
// padding rows) alternates NaN and the largest finite component.
template<typename T>
std::vector<T> make_hessenberg(std::size_t n, std::size_t lda, std::uint32_t seed) {
  using R = typename elem<T>::R;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double big = std::numeric_limits<R>::max();
  std::vector<T> a(lda * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      const bool in = i < n && i <= j + 1;
      if (in) {
        const double re = dist(gen);
        a[i + j * lda] = elem<T>::make(re, dist(gen));
      } else {
        a[i + j * lda] = (i + j) % 2 == 0 ? elem<T>::make(nan, nan) : elem<T>::make(big, big);
      }
    }
  }
  return a;
}

template<typename T>
void expect_matches_reference(const NormCase nc, std::size_t n, std::size_t lda) {
  using R = typename elem<T>::R;
  const auto a = make_hessenberg<T>(n, lda, static_cast<std::uint32_t>(31 * n + 7 * lda + 1));
  const R oracle = reference(nc.c, n, a, lda);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_A = to_device(a);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s =
      lanhs<T>(shared_device()->stream().get(), nc.which, n, d_A.data(), lda, d_result.data());
  ASSERT_TRUE(s.ok()) << "lanhs returned " << s.name() << ": " << s.message();
  // The real max norm compares and copies only, so it must match exactly; the
  // sums differ from the reference in order (and, for 'F', by ?lassq's
  // scaling), so they get the shared eps-relative tolerance, n terms per sum.
  const R tol = (elem<T>::exact_max && nc.which == MatrixNorm::max_abs)
                    ? R{0}
                    : factorization_tol<R>(oracle, n, n);
  EXPECT_NEAR(scalar_from_device(d_result), oracle, tol)
      << "norm=" << nc.c << " n=" << n << " lda=" << lda;
}

template<typename T>
void run_sizes() {
  // 1..3: the edge columns (n = 1 has no subdiagonal); 255..257: around the
  // block's thread count, so a column or row straddles one stride; 1000: many
  // strides. Each with lda == n and a padded lda > n.
  for (const NormCase nc : kNormCases) {
    for (const std::size_t n : {1u, 2u, 3u, 17u, 64u, 255u, 256u, 257u, 1000u}) {
      expect_matches_reference<T>(nc, n, n);
      expect_matches_reference<T>(nc, n, n + 5);
    }
  }
}

// n == 0 writes 0 over a pre-seeded sentinel for every norm, as DLANHS returns 0.
template<typename T>
void expect_empty_writes_zero() {
  using R = typename elem<T>::R;
  auto dummy = to_device(std::vector<T>{elem<T>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    ASSERT_EQ(reference<T>(nc.c, 0, std::vector<T>(1), 1), R{0});
    auto d_result = to_device(std::vector<R>{R(-12345)});
    ASSERT_TRUE(
        lanhs<T>(shared_device()->stream().get(), nc.which, 0, dummy.data(), 1, d_result.data())
            .ok());
    EXPECT_EQ(scalar_from_device(d_result), R{0}) << "norm=" << nc.c;
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanhs, per type
// ========================================================================

TEST(LanhsOracleTests, Float) {
  run_sizes<float>();
}
TEST(LanhsOracleTests, Double) {
  run_sizes<double>();
}
TEST(LanhsOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanhsOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

TEST(LanhsOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in the referenced part -- on the subdiagonal, the diagonal, or the top
// row -- propagates through every norm, as DLANHS's DISNAN guard does for the
// max folds.
TEST(LanhsOracleTests, NaNInReferencedPartPropagates) {
  const std::size_t n = 300;
  const std::size_t lda = n + 3;
  // (151,150) subdiagonal, (77,77) diagonal, (0,299) top-right corner.
  for (const std::size_t pos : {151 + 150 * lda, 77 + 77 * lda, 0 + 299 * lda}) {
    auto a = make_hessenberg<double>(n, lda, 9);
    a[pos] = std::numeric_limits<double>::quiet_NaN();
    auto d_A = to_device(a);
    DeviceBuffer<double> d_result(1, shared_device());
    for (const NormCase nc : kNormCases) {
      ASSERT_TRUE(lanhs<double>(shared_device()->stream().get(), nc.which, n, d_A.data(), lda,
                                d_result.data())
                      .ok());
      EXPECT_TRUE(std::isnan(scalar_from_device(d_result))) << "norm=" << nc.c << " pos=" << pos;
    }
  }
}

} // namespace
} // namespace calaman
