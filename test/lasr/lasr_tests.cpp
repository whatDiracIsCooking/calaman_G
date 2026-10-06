// Oracle test for calaman.lasr -- apply a sequence of plane rotations (?lasr).
// The oracle is the reference ?lasr in the SAME precision on the host, run over
// the identical A, c and s, for all 12 SIDE x PIVOT x DIRECT combinations.
//
// Every case stages A on the device and runs the kernel (whose blocks each call
// the block-cooperative lasr_block of lapack/lasr/lasr.h), so the suite is
// REQUIRES_GPU. Built only when calaman::lapack_reference exists; see this
// directory's CMakeLists.txt.

#include <gtest/gtest.h>

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
}

import std;

import wwr.runtime_api;
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

void ref_lasr(char side, char pivot, char direct, int m, int n, const float *c, const float *s,
              float *a, int lda) {
  slasr_(&side, &pivot, &direct, &m, &n, c, s, a, &lda, 1, 1, 1);
}
void ref_lasr(char side, char pivot, char direct, int m, int n, const double *c,
              const double *s, double *a, int lda) {
  dlasr_(&side, &pivot, &direct, &m, &n, c, s, a, &lda, 1, 1, 1);
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

// A sentinel in the lda padding rows: lasr must never write it.
constexpr double kPad = 12345.0;

template<typename T>
void expect_matches_reference(Side side, Pivot pivot, Direct direct, std::size_t m,
                              std::size_t n) {
  const std::size_t lda = m + 3;
  const std::size_t k = side == Side::L ? m : n;
  const std::size_t nrot = k < 2 ? 0 : k - 1;
  const auto seed = static_cast<std::uint32_t>(131 * m + 7 * n + 3 * side_char(side) +
                                               5 * pivot_char(pivot) + direct_char(direct));
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> val(-4.0, 4.0);
  std::uniform_real_distribution<double> angle(-3.14159, 3.14159);

  std::vector<T> A(lda * n, static_cast<T>(kPad));
  std::vector<T> live;
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      A[i + j * lda] = static_cast<T>(val(gen));
      live.push_back(A[i + j * lda]);
    }
  }
  std::vector<T> c(nrot);
  std::vector<T> s(nrot);
  for (std::size_t j = 0; j < nrot; ++j) {
    const double t = angle(gen);
    c[j] = static_cast<T>(std::cos(t));
    s[j] = static_cast<T>(std::sin(t));
  }
  if (nrot > 2) { // one identity rotation, which DLASR skips
    c[1] = T{1};
    s[1] = T{0};
  }

  std::vector<T> want = A;
  ref_lasr(side_char(side), pivot_char(pivot), direct_char(direct), static_cast<int>(m),
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
  const T tol = factorization_tol<T>(frobenius_norm(live), k, k);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      const std::size_t at = i + j * lda;
      if (i >= m) {
        ASSERT_EQ(got[at], static_cast<T>(kPad)) << "padding written at (" << i << "," << j << ")";
      } else {
        ASSERT_NEAR(got[at], want[at], tol)
            << side_char(side) << pivot_char(pivot) << direct_char(direct) << " m=" << m
            << " n=" << n << " at (" << i << "," << j << ")";
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

// lda < max(1, m) is rejected before anything is enqueued, as DLASR's INFO = 9.
TEST(LasrOracleTests, RejectsShortLeadingDimension) {
  const Status st = lasr<double>(shared_device()->stream().get(), Side::L, Pivot::V, Direct::F,
                                 4, 3, nullptr, nullptr, nullptr, 3);
  EXPECT_FALSE(st.ok());
  const Status zero = lasr<double>(shared_device()->stream().get(), Side::R, Pivot::B, Direct::B,
                                   0, 3, nullptr, nullptr, nullptr, 0);
  EXPECT_FALSE(zero.ok());
}

} // namespace
} // namespace calaman
