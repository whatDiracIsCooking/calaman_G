// Oracle test for calaman.lanht -- the ?lanht norm of a complex Hermitian
// tridiagonal (real d, complex e). The oracle is the reference ?lanht
// (LAPACK_{c,z}lanht) in the SAME precision on the host, over the same buffers
// (the wwr complex types are layout-compatible with lapack_complex_*).
//
// ?lanht has no leading dimension; its analogue here is a sub-range: d and e
// are views at an offset into larger buffers whose slots outside the view hold
// garbage (NaN and the largest finite value), so a read outside [0, n) of d or
// [0, n-1) of e turns the result into NaN or inf and fails the comparison. |z|
// is an inexact hypot, so every norm takes the shared tolerance. All suites
// stage data on the device, so they are REQUIRES_GPU (labeled `gpu`). Built
// only when calaman::lapack_reference exists; see this directory's
// CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lanht;
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

// Every LAPACK NORM char lanht accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// Slots of garbage before and after each view: the "leading dimension" padding.
constexpr std::size_t kPad = 3;

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

template<typename C>
struct complex_elem;

// LAPACKE has no ?lanht wrapper, so call the Fortran routine through lapack.h's
// LAPACK_?lanht (which lapacke.h includes; it appends the hidden string length).
template<>
struct complex_elem<wwr::wwrFloatComplex> {
  using R = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, std::size_t n, const float *d, const wwr::wwrFloatComplex *e) {
    const lapack_int ni = static_cast<lapack_int>(n);
    return static_cast<float>(
        LAPACK_clanht(&norm, &ni, d, reinterpret_cast<const lapack_complex_float *>(e)));
  }
};

template<>
struct complex_elem<wwr::wwrDoubleComplex> {
  using R = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double ref(char norm, std::size_t n, const double *d, const wwr::wwrDoubleComplex *e) {
    const lapack_int ni = static_cast<lapack_int>(n);
    return LAPACK_zlanht(&norm, &ni, d, reinterpret_cast<const lapack_complex_double *>(e));
  }
};

// (d, e) as views at offset kPad into buffers of n + 2*kPad slots; the view
// holds mixed-sign values over a few binades, every other slot alternates NaN
// and the largest finite value.
template<typename C>
struct Tridiagonal {
  using R = typename complex_elem<C>::R;
  std::vector<R> d;
  std::vector<C> e;

  Tridiagonal(std::size_t n, std::uint32_t seed) : d(n + 2 * kPad), e(n + 2 * kPad) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double big = std::numeric_limits<R>::max();
    const std::size_t ne = n == 0 ? 0 : n - 1;
    for (std::size_t k = 0; k < d.size(); ++k) {
      const bool in_d = k >= kPad && k < kPad + n;
      const bool in_e = k >= kPad && k < kPad + ne;
      const double junk = k % 2 == 0 ? nan : big;
      d[k] = static_cast<R>(in_d ? dist(gen) : junk);
      e[k] = in_e ? complex_elem<C>::make(dist(gen), dist(gen)) : complex_elem<C>::make(junk, junk);
    }
  }
};

template<typename C>
void expect_matches_reference(const NormCase nc, std::size_t n) {
  using R = typename complex_elem<C>::R;
  const Tridiagonal<C> t(n, static_cast<std::uint32_t>(41 * n + 7));
  const R oracle = complex_elem<C>::ref(nc.c, n, t.d.data() + kPad, t.e.data() + kPad);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_d = to_device(t.d);
  auto d_e = to_device(t.e);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = lanht<C>(shared_device()->stream().get(), nc.which, n, d_d.data() + kPad,
                            d_e.data() + kPad, d_result.data());
  ASSERT_TRUE(s.ok()) << "lanht returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, factorization_tol<R>(oracle, 4, 4))
      << "norm=" << nc.c << " n=" << n;
}

template<typename C>
void run_sizes() {
  // 1..3: the edge rows (n = 1 reads d alone); 255..257: around the block's
  // thread count (chunk 1 vs 2); 1000/4099: multi-row chunks, ragged last one.
  for (const NormCase nc : kNormCases) {
    for (const std::size_t n : {1u, 2u, 3u, 7u, 64u, 255u, 256u, 257u, 1000u, 4099u}) {
      expect_matches_reference<C>(nc, n);
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lanht, per type
// ========================================================================

TEST(LanhtOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LanhtOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// A dominant off-diagonal decides the max norm by its MODULUS: |3+4i| = 5, not
// max(|Re|, |Im|) = 4 or |Re| + |Im| = 7.
TEST(LanhtOracleTests, MaxAbsUsesComplexModulus) {
  using C = wwr::wwrDoubleComplex;
  const std::vector<double> d = {1.0, -2.0, 0.5, 1.5};
  const std::vector<C> e = {complex_elem<C>::make(0.5, 0.5), complex_elem<C>::make(3.0, -4.0),
                            complex_elem<C>::make(-1.0, 0.0)};
  const double oracle = complex_elem<C>::ref('M', d.size(), d.data(), e.data());
  ASSERT_EQ(oracle, 5.0);
  auto d_d = to_device(d);
  auto d_e = to_device(e);
  DeviceBuffer<double> d_result(1, shared_device());
  ASSERT_TRUE(lanht<C>(shared_device()->stream().get(), MatrixNorm::max_abs, d.size(), d_d.data(),
                       d_e.data(), d_result.data())
                  .ok());
  EXPECT_NEAR(scalar_from_device(d_result), oracle, factorization_tol<double>(oracle, 4, 4));
}

// n == 0 writes a real 0 for every norm, over a sentinel (ZLANHT returns 0).
template<typename C>
void expect_empty_writes_zero() {
  using R = typename complex_elem<C>::R;
  auto dummy_d = to_device(std::vector<R>{R(42)});
  auto dummy_e = to_device(std::vector<C>{complex_elem<C>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    auto d_result = to_device(std::vector<R>{R(-12345)});
    ASSERT_TRUE(lanht<C>(shared_device()->stream().get(), nc.which, 0, dummy_d.data(),
                         dummy_e.data(), d_result.data())
                    .ok());
    EXPECT_EQ(scalar_from_device(d_result), R{0}) << "norm=" << nc.c;
    EXPECT_EQ(complex_elem<C>::ref(nc.c, 0, nullptr, nullptr), R{0}) << "norm=" << nc.c;
  }
}

TEST(LanhtOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in either component of an off-diagonal -- including the one a chunk
// boundary hands to its left neighbour -- propagates through every norm, as
// ZLANHT's DISNAN guard does for the max folds.
TEST(LanhtOracleTests, NaNPropagates) {
  using C = wwr::wwrDoubleComplex;
  const std::size_t n = 600; // chunk 3: e[2] links thread 0 to thread 1
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const std::size_t pos : {std::size_t{2}, n - 2}) {
    for (const C bad : {complex_elem<C>::make(1.0, nan), complex_elem<C>::make(nan, 1.0)}) {
      Tridiagonal<C> t(n, 9);
      t.e[kPad + pos] = bad;
      auto d_d = to_device(t.d);
      auto d_e = to_device(t.e);
      DeviceBuffer<double> d_result(1, shared_device());
      for (const NormCase nc : kNormCases) {
        ASSERT_TRUE(lanht<C>(shared_device()->stream().get(), nc.which, n, d_d.data() + kPad,
                             d_e.data() + kPad, d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << nc.c << " pos=" << pos;
      }
    }
  }
}

} // namespace
} // namespace calaman
