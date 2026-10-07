// Oracle test for calaman.langb -- the ?langb norm of an n-by-n general band
// matrix in LAPACK band storage. The oracle is the reference ?langb
// (LAPACKE_?langb, s/d/c/z) in the SAME precision on the host.
//
// Every case fills the slots of AB that hold no in-matrix band entry -- the
// unused top-left and bottom-right corners and the ldab > kl+ku+1 padding rows
// -- with garbage (NaN and the largest finite value): a single read of one
// would turn the result into NaN or inf and fail the comparison. LAPACKE's NaN
// check reads only the in-matrix band, so the oracle accepts that garbage. All
// suites stage data on the device, so they are REQUIRES_GPU (labeled `gpu`).
// Built only when calaman::lapack_reference exists; see this directory's
// CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.langb;
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

// Per-type element construction and the same-precision oracle. R is the real
// type of the norm; `exact_max` is whether 'M' compares and copies only (real),
// or takes an inexact |z| hypot (complex).
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool exact_max = true;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, std::size_t n, std::size_t kl, std::size_t ku, const float *ab,
                   std::size_t ldab) {
    return LAPACKE_slangb(LAPACK_COL_MAJOR, norm, static_cast<lapack_int>(n),
                          static_cast<lapack_int>(kl), static_cast<lapack_int>(ku), ab,
                          static_cast<lapack_int>(ldab));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool exact_max = true;
  static double make(double re, double) { return re; }
  static double ref(char norm, std::size_t n, std::size_t kl, std::size_t ku, const double *ab,
                    std::size_t ldab) {
    return LAPACKE_dlangb(LAPACK_COL_MAJOR, norm, static_cast<lapack_int>(n),
                          static_cast<lapack_int>(kl), static_cast<lapack_int>(ku), ab,
                          static_cast<lapack_int>(ldab));
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static constexpr bool exact_max = false;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, std::size_t n, std::size_t kl, std::size_t ku,
                   const wwr::wwrFloatComplex *ab, std::size_t ldab) {
    return LAPACKE_clangb(LAPACK_COL_MAJOR, norm, static_cast<lapack_int>(n),
                          static_cast<lapack_int>(kl), static_cast<lapack_int>(ku),
                          reinterpret_cast<const lapack_complex_float *>(ab),
                          static_cast<lapack_int>(ldab));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static constexpr bool exact_max = false;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double ref(char norm, std::size_t n, std::size_t kl, std::size_t ku,
                    const wwr::wwrDoubleComplex *ab, std::size_t ldab) {
    return LAPACKE_zlangb(LAPACK_COL_MAJOR, norm, static_cast<lapack_int>(n),
                          static_cast<lapack_int>(kl), static_cast<lapack_int>(ku),
                          reinterpret_cast<const lapack_complex_double *>(ab),
                          static_cast<lapack_int>(ldab));
  }
};

// Every LAPACK NORM char langb accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' Frobenius's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// Row r of AB column j holds A(r + j - ku, j): an in-matrix band entry iff that
// row index lies in [0, n) and r < kl+ku+1.
bool in_band(std::size_t r, std::size_t j, std::size_t n, std::size_t kl, std::size_t ku) {
  return r <= kl + ku && r + j >= ku && r + j - ku < n;
}

// Band entries are mixed-sign values over a few binades (complex: both parts);
// every other slot of the ldab-by-n array alternates NaN and the largest finite
// component.
template<typename T>
std::vector<T> make_band(std::size_t n, std::size_t kl, std::size_t ku, std::size_t ldab,
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
      ab[r + j * ldab] = in_band(r, j, n, kl, ku) ? elem<T>::make(re, im)
                         : (r + j) % 2 == 0       ? elem<T>::make(nan, nan)
                                                  : elem<T>::make(big, big);
    }
  }
  return ab;
}

// The real max norm compares and copies only, so it must match exactly; the
// sums differ from the reference in order (and, for 'F', by ?lassq's scaling),
// and a complex |z| is inexact, so those get the shared eps-relative tolerance.
template<typename T>
void expect_matches_reference(const NormCase nc, std::size_t n, std::size_t kl, std::size_t ku,
                              std::size_t ldab) {
  using R = typename elem<T>::R;
  const auto ab = make_band<T>(n, kl, ku, ldab,
                               static_cast<std::uint32_t>(31 * n + 7 * ldab + 5 * kl + 3 * ku + 1));
  const R oracle = elem<T>::ref(nc.c, n, kl, ku, ab.data(), ldab);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_AB = to_device(ab);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = langb<T>(shared_device()->stream().get(), nc.which, n, kl, ku, d_AB.data(), ldab,
                            d_result.data());
  ASSERT_TRUE(s.ok()) << "langb returned " << s.name() << ": " << s.message();
  const R tol = (nc.which == MatrixNorm::max_abs && elem<T>::exact_max)
                    ? R{0}
                    : factorization_tol<R>(oracle, n, n);
  EXPECT_NEAR(scalar_from_device(d_result), oracle, tol)
      << "norm=" << nc.c << " n=" << n << " kl=" << kl << " ku=" << ku << " ldab=" << ldab;
}

struct Band {
  std::size_t kl;
  std::size_t ku;
};

// Diagonal, bidiagonal both ways, narrow and lopsided bands, a band wider than
// one block's 256 threads (so a column or row straddles a stride), and kl/ku
// past n (a full matrix in band storage).
constexpr Band kBands[] = {{0, 0}, {1, 0}, {0, 1}, {2, 3}, {7, 1}, {200, 120}, {1100, 1100}};

template<typename T>
void run_sizes() {
  // n = 1 edge, small orders, around one block's stride, and many strides; each
  // with the tight ldab == kl+ku+1 and a padded one.
  for (const NormCase nc : kNormCases) {
    for (const Band b : kBands) {
      for (const std::size_t n : {1u, 2u, 3u, 17u, 255u, 257u, 700u}) {
        const std::size_t tight = b.kl + b.ku + 1;
        expect_matches_reference<T>(nc, n, b.kl, b.ku, tight);
        expect_matches_reference<T>(nc, n, b.kl, b.ku, tight + 3);
      }
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?langb, per type
// ========================================================================

TEST(LangbOracleTests, Float) {
  run_sizes<float>();
}
TEST(LangbOracleTests, Double) {
  run_sizes<double>();
}
TEST(LangbOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LangbOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// n == 0 writes a real 0 (DLANGB returns 0) over a pre-seeded sentinel, for
// every norm.
template<typename T>
void expect_empty_writes_zero() {
  using R = typename elem<T>::R;
  auto dummy = to_device(std::vector<T>{elem<T>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    auto d_result = to_device(std::vector<R>{R(-12345)});
    ASSERT_TRUE(langb<T>(shared_device()->stream().get(), nc.which, 0, 1, 1, dummy.data(), 3,
                         d_result.data())
                    .ok());
    EXPECT_EQ(scalar_from_device(d_result), R{0}) << "norm=" << nc.c;
  }
}

TEST(LangbOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in the band -- on the diagonal, a sub- or a super-diagonal --
// propagates through every norm, as DLANGB's DISNAN guard does for the max.
TEST(LangbOracleTests, NaNInBandPropagates) {
  const std::size_t n = 300;
  const std::size_t kl = 4;
  const std::size_t ku = 2;
  const std::size_t ldab = kl + ku + 3;
  // (i, j) in the band: diagonal, last sub-diagonal, last super-diagonal.
  const std::pair<std::size_t, std::size_t> spots[] = {{77, 77}, {150 + kl, 150}, {10, 10 + ku}};
  for (const auto &[i, j] : spots) {
    auto ab = make_band<double>(n, kl, ku, ldab, 9);
    ab[ku + i - j + j * ldab] = std::numeric_limits<double>::quiet_NaN();
    auto d_AB = to_device(ab);
    DeviceBuffer<double> d_result(1, shared_device());
    for (const NormCase nc : kNormCases) {
      ASSERT_TRUE(langb<double>(shared_device()->stream().get(), nc.which, n, kl, ku, d_AB.data(),
                                ldab, d_result.data())
                      .ok());
      EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
          << "norm=" << nc.c << " i=" << i << " j=" << j;
    }
  }
}

} // namespace
} // namespace calaman
