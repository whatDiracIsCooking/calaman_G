// Oracle test for calaman.lasr -- apply a sequence of plane rotations (?lasr).
// The oracle is the reference ?lasr in the SAME precision on the host, run over
// the identical A, c and s, for all 12 SIDE x PIVOT x DIRECT combinations and
// all four element types (the complex ones with real c, s, as ZLASR takes them;
// the wwr complex types are layout-compatible with the Fortran COMPLEX types).
//
// Every case stages A on the device and runs the kernel (whose blocks each call
// the block-cooperative lasr_block of lapack/lasr/lasr.h), so the suite is
// REQUIRES_GPU. Built only when calaman::lapack_reference exists; see this
// directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h> // lapack_complex_float, lapack_complex_double

#include <cstddef>

// Neither LAPACKE nor lapack.h declares ?lasr, so call the Fortran symbol: every
// argument by reference, plus one hidden length per CHARACTER argument. The
// symbols live in LAPACK::LAPACK, linked by calaman::lapack_reference.
extern "C" {
void slasr_(const char *side, const char *pivot, const char *direct, const int *m, const int *n,
            const float *c, const float *s, float *a, const int *lda, std::size_t side_len,
            std::size_t pivot_len, std::size_t direct_len);
void dlasr_(const char *side, const char *pivot, const char *direct, const int *m, const int *n,
            const double *c, const double *s, double *a, const int *lda, std::size_t side_len,
            std::size_t pivot_len, std::size_t direct_len);
void clasr_(const char *side, const char *pivot, const char *direct, const int *m, const int *n,
            const float *c, const float *s, lapack_complex_float *a, const int *lda,
            std::size_t side_len, std::size_t pivot_len, std::size_t direct_len);
void zlasr_(const char *side, const char *pivot, const char *direct, const int *m, const int *n,
            const double *c, const double *s, lapack_complex_double *a, const int *lda,
            std::size_t side_len, std::size_t pivot_len, std::size_t direct_len);
}

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lasr;
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
using test::frobenius_norm;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// ========================================================================
// Per-element-type traits: the real component type, the builder, the
// component accessors (host wwrC* -- .x/.y is not portable to hipComplex),
// and the oracle
// ========================================================================

template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr const char *kName = "s";
  static float make(double re, double) { return static_cast<float>(re); }
  static float re(float v) { return v; }
  static float im(float) { return 0.0F; }
  static void ref(char side, char pivot, char direct, int m, int n, const float *c,
                  const float *s, float *a, int lda) {
    slasr_(&side, &pivot, &direct, &m, &n, c, s, a, &lda, 1, 1, 1);
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr const char *kName = "d";
  static double make(double re, double) { return re; }
  static double re(double v) { return v; }
  static double im(double) { return 0.0; }
  static void ref(char side, char pivot, char direct, int m, int n, const double *c,
                  const double *s, double *a, int lda) {
    dlasr_(&side, &pivot, &direct, &m, &n, c, s, a, &lda, 1, 1, 1);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static constexpr const char *kName = "c";
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float re(wwr::wwrFloatComplex v) { return wwr::wwrCrealf(v); }
  static float im(wwr::wwrFloatComplex v) { return wwr::wwrCimagf(v); }
  static void ref(char side, char pivot, char direct, int m, int n, const float *c,
                  const float *s, wwr::wwrFloatComplex *a, int lda) {
    clasr_(&side, &pivot, &direct, &m, &n, c, s, reinterpret_cast<lapack_complex_float *>(a),
           &lda, 1, 1, 1);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static constexpr const char *kName = "z";
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double re(wwr::wwrDoubleComplex v) { return wwr::wwrCreal(v); }
  static double im(wwr::wwrDoubleComplex v) { return wwr::wwrCimag(v); }
  static void ref(char side, char pivot, char direct, int m, int n, const double *c,
                  const double *s, wwr::wwrDoubleComplex *a, int lda) {
    zlasr_(&side, &pivot, &direct, &m, &n, c, s, reinterpret_cast<lapack_complex_double *>(a),
           &lda, 1, 1, 1);
  }
};

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
std::vector<T> to_host(const DeviceBuffer<T> &device, std::size_t n) {
  const auto handle = shared_device();
  HostBuffer<T> host(n == 0 ? 1 : n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

constexpr std::array kSides = {Side::L, Side::R};
constexpr std::array kPivots = {Pivot::V, Pivot::T, Pivot::B};
constexpr std::array kDirects = {Direct::F, Direct::B};

char side_char(Side v) {
  return v == Side::L ? 'L' : 'R';
}
char pivot_char(Pivot v) {
  return v == Pivot::V ? 'V' : (v == Pivot::T ? 'T' : 'B');
}
char direct_char(Direct v) {
  return v == Direct::F ? 'F' : 'B';
}

// A sentinel in the lda padding rows (both components for a complex T): lasr
// must never write it.
constexpr double kPad = 12345.0;

template<typename T>
void expect_matches_reference(Side side, Pivot pivot, Direct direct, std::size_t m,
                              std::size_t n) {
  using E = elem<T>;
  using R = typename E::R;
  const std::size_t lda = m + 3;
  const std::size_t k = side == Side::L ? m : n;
  const std::size_t nrot = k < 2 ? 0 : k - 1;
  const auto seed = static_cast<std::uint32_t>(131 * m + 7 * n + 3 * side_char(side) +
                                               5 * pivot_char(pivot) + direct_char(direct));
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> val(-4.0, 4.0);
  std::uniform_real_distribution<double> angle(-3.14159, 3.14159);

  const T pad = E::make(kPad, kPad);
  std::vector<T> A(lda * n, pad);
  // The live entries' real and imaginary parts, flattened: ||A||_F is the
  // Frobenius norm of that real vector.
  std::vector<R> live;
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      const double re = val(gen);
      const double im = val(gen);
      A[i + j * lda] = E::make(re, im);
      live.push_back(E::re(A[i + j * lda]));
      live.push_back(E::im(A[i + j * lda]));
    }
  }
  std::vector<R> c(nrot);
  std::vector<R> s(nrot);
  for (std::size_t j = 0; j < nrot; ++j) {
    const double t = angle(gen);
    c[j] = static_cast<R>(std::cos(t));
    s[j] = static_cast<R>(std::sin(t));
  }
  if (nrot > 2) { // one identity rotation, which DLASR skips
    c[1] = R{1};
    s[1] = R{0};
  }

  std::vector<T> want = A;
  E::ref(side_char(side), pivot_char(pivot), direct_char(direct), static_cast<int>(m),
         static_cast<int>(n), c.data(), s.data(), want.data(), static_cast<int>(lda));

  auto d_A = to_device(A);
  auto d_c = to_device(c);
  auto d_s = to_device(s);
  const Status st = lasr<T>(shared_device()->stream().get(), side, pivot, direct, m, n,
                            d_c.data(), d_s.data(), d_A.data(), lda);
  ASSERT_TRUE(st.ok()) << "lasr returned " << st.name() << ": " << st.message();
  const std::vector<T> got = to_host(d_A, A.size());

  // Each line passes through up to k-1 rotations: the chain length plays the
  // role min(m, n) plays for a factorization.
  const R tol = factorization_tol<R>(frobenius_norm(live), k, k);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      const std::size_t at = i + j * lda;
      if (i >= m) {
        ASSERT_EQ(E::re(got[at]), E::re(pad)) << "padding written at (" << i << "," << j << ")";
        ASSERT_EQ(E::im(got[at]), E::im(pad)) << "padding written at (" << i << "," << j << ")";
      } else {
        ASSERT_NEAR(E::re(got[at]), E::re(want[at]), tol)
            << E::kName << side_char(side) << pivot_char(pivot) << direct_char(direct)
            << " m=" << m << " n=" << n << " re at (" << i << "," << j << ")";
        ASSERT_NEAR(E::im(got[at]), E::im(want[at]), tol)
            << E::kName << side_char(side) << pivot_char(pivot) << direct_char(direct)
            << " m=" << m << " n=" << n << " im at (" << i << "," << j << ")";
      }
    }
  }
}

// m or n of 0/1 (including k == 1, no rotations at all), tiny, around the
// 256-thread block and the 1024-line slab, and multi-block on either side.
constexpr std::array<std::pair<std::size_t, std::size_t>, 14> kShapes = {{
    {0, 5}, {5, 0}, {0, 0}, {1, 1}, {1, 9}, {9, 1}, {2, 2}, {3, 7},
    {37, 53}, {300, 129}, {129, 300}, {64, 1100}, {1100, 64}, {2, 2500},
}};

template<typename T>
void run_all_combinations() {
  for (const Side side : kSides) {
    for (const Pivot pivot : kPivots) {
      for (const Direct direct : kDirects) {
        for (const auto &[m, n] : kShapes) {
          expect_matches_reference<T>(side, pivot, direct, m, n);
        }
      }
    }
  }
}

TEST(LasrOracleTests, AllCombinationsFloat) {
  run_all_combinations<float>();
}
TEST(LasrOracleTests, AllCombinationsDouble) {
  run_all_combinations<double>();
}
TEST(LasrOracleTests, AllCombinationsComplexFloat) {
  run_all_combinations<wwr::wwrFloatComplex>();
}
TEST(LasrOracleTests, AllCombinationsComplexDouble) {
  run_all_combinations<wwr::wwrDoubleComplex>();
}

// lda < max(1, m) is rejected before anything is enqueued, as DLASR's INFO = 9.
TEST(LasrOracleTests, RejectsShortLeadingDimension) {
  const Status st = lasr<double>(shared_device()->stream().get(), Side::L, Pivot::V, Direct::F,
                                 4, 3, nullptr, nullptr, nullptr, 3);
  EXPECT_FALSE(st.ok());
  const Status zero = lasr<double>(shared_device()->stream().get(), Side::R, Pivot::B, Direct::B,
                                   0, 3, nullptr, nullptr, nullptr, 0);
  EXPECT_FALSE(zero.ok());
  const Status cplx = lasr<wwr::wwrDoubleComplex>(shared_device()->stream().get(), Side::L,
                                                  Pivot::V, Direct::F, 4, 3, nullptr, nullptr,
                                                  nullptr, 3);
  EXPECT_FALSE(cplx.ok());
}

} // namespace
} // namespace calaman
