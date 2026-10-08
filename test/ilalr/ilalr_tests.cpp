// Oracle test for calaman.ilalr -- the last non-zero row of a column-major
// matrix, LAPACK's ila?lr, read as a count of leading rows.
//
// There is no LAPACKE wrapper and no LAPACK_ila?lr declaration in lapack.h, so
// the oracle is the reference Fortran itself (ilaslr_/iladlr_/ilaclr_/ilazlr_,
// exported by liblapack, hand-declared below). The result is an exact integer,
// so the comparison is equality. Every case fills the lda - m padding rows with
// a non-zero sentinel, so a scan that strays into the padding is caught.
//
// Cases: all-zero A, a single non-zero at every position of a small matrix
// (first/last row and column included), a matrix larger than one block with a
// zero trailing band, NaN / -0 / imaginary-only entries, m == 0 and n == 0
// (0 without calling the oracle, which reads A(M,1) when N = 0), and the
// argument checks.
//
// REQUIRES_GPU (see CMakeLists.txt): every case runs the scan on the device.

#include <gtest/gtest.h>

extern "C" {
int ilaslr_(const int *m, const int *n, const float *a, const int *lda);
int iladlr_(const int *m, const int *n, const double *a, const int *lda);
int ilaclr_(const int *m, const int *n, const void *a, const int *lda);
int ilazlr_(const int *m, const int *n, const void *a, const int *lda);
}

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.ilalr;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const std::vector<T> &host) {
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

int int_from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<int> &device) {
  HostBuffer<int> host(1);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return host.data()[0];
}

/// @brief Per-type element construction and the Fortran oracle
template<typename T>
struct Elem;

template<>
struct Elem<float> {
  static float make(double re, double /*im*/) { return static_cast<float>(re); }
  static int ref(int m, int n, const float *a, int lda) { return ilaslr_(&m, &n, a, &lda); }
};

template<>
struct Elem<double> {
  static double make(double re, double /*im*/) { return re; }
  static int ref(int m, int n, const double *a, int lda) { return iladlr_(&m, &n, a, &lda); }
};

template<>
struct Elem<wwr::wwrFloatComplex> {
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static int ref(int m, int n, const wwr::wwrFloatComplex *a, int lda) {
    return ilaclr_(&m, &n, a, &lda);
  }
};

template<>
struct Elem<wwr::wwrDoubleComplex> {
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static int ref(int m, int n, const wwr::wwrDoubleComplex *a, int lda) {
    return ilazlr_(&m, &n, a, &lda);
  }
};

/// @brief An lda-by-n buffer: zero in rows [0, m), a non-zero sentinel below
template<typename T>
std::vector<T> padded_zero(int m, int n, int lda) {
  std::vector<T> a(static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < lda; ++i) {
      a[static_cast<std::size_t>(j) * lda + i] = Elem<T>::make(i < m ? 0.0 : 7.0, 0.0);
    }
  }
  return a;
}

/// @brief Run calaman::ilalr on @p a; d_last is pre-filled with -1 so a missed write shows
template<typename T>
int run_device(int m, int n, int lda, const std::vector<T> &a) {
  auto handle = shared_device();
  auto d_a = to_device(handle, a);
  auto d_last = to_device(handle, std::vector<int>{-1});
  const std::size_t bytes = ilalr_bufferSize(m, n);
  DeviceBuffer<std::byte> d_work(bytes == 0 ? 1 : bytes, handle);
  const auto status =
      ilalr<T>(handle->stream().get(), m, n, d_a.data(), lda, d_last.data(), d_work.data(), bytes);
  EXPECT_TRUE(status.ok()) << "ilalr returned status=" << status.name();
  return int_from_device(handle, d_last);
}

/// @brief Device result equals the Fortran oracle on the same storage
template<typename T>
void check(int m, int n, int lda, const std::vector<T> &a, const std::string &ctx) {
  const int want = Elem<T>::ref(m, n, a.data(), lda);
  EXPECT_EQ(run_device<T>(m, n, lda, a), want) << ctx;
}

template<typename T>
void all_zero() {
  check<T>(5, 4, 7, padded_zero<T>(5, 4, 7), "all zero, padded");
  check<T>(1, 1, 1, padded_zero<T>(1, 1, 1), "1x1 zero");
}

template<typename T>
void single_nonzero_everywhere() {
  constexpr int m = 5;
  constexpr int n = 4;
  constexpr int lda = 7;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      auto a = padded_zero<T>(m, n, lda);
      a[static_cast<std::size_t>(j) * lda + i] = Elem<T>::make(-3.0, 0.0);
      check<T>(m, n, lda, a, "single non-zero at (" + std::to_string(i) + "," +
                                 std::to_string(j) + ")");
    }
  }
}

template<typename T>
void larger_than_a_block() {
  // Rows past last_row are zero; the block above is dense with a few holes.
  constexpr int m = 300;
  constexpr int n = 70;
  constexpr int lda = 303;
  for (const int last_row : {1, 129, 257, 300}) {
    auto a = padded_zero<T>(m, n, lda);
    for (int j = 0; j < 40; ++j) {
      for (int i = 0; i < last_row; ++i) {
        const int v = (i * 7 + j * 3) % 11 - 5; // some exact zeros
        a[static_cast<std::size_t>(j) * lda + i] = Elem<T>::make(v, 0.0);
      }
    }
    a[static_cast<std::size_t>(39) * lda + (last_row - 1)] = Elem<T>::make(1.0, 0.0);
    check<T>(m, n, lda, a, "dense block to row " + std::to_string(last_row));
  }
}

template<typename T>
void special_values() {
  constexpr int m = 6;
  constexpr int n = 5;
  constexpr int lda = 8;
  const double nan = std::numeric_limits<double>::quiet_NaN();

  auto a = padded_zero<T>(m, n, lda);
  a[1 * lda + 2] = Elem<T>::make(nan, 0.0);
  check<T>(m, n, lda, a, "NaN counts as non-zero");

  a = padded_zero<T>(m, n, lda);
  a[3 * lda + 5] = Elem<T>::make(-0.0, -0.0);
  check<T>(m, n, lda, a, "-0 counts as zero");

  a = padded_zero<T>(m, n, lda);
  a[0 * lda + 4] = Elem<T>::make(0.0, 2.0); // zero for a real T
  check<T>(m, n, lda, a, "imaginary-only entry");
}

template<typename T>
void empty_is_zero() {
  // No oracle: the reference reads A(M,1) when N = 0. 0 is this module's contract.
  EXPECT_EQ(run_device<T>(0, 4, 1, padded_zero<T>(0, 4, 1)), 0) << "m == 0";
  EXPECT_EQ(run_device<T>(5, 0, 5, padded_zero<T>(5, 1, 5)), 0) << "n == 0";
}

template<typename T>
void bad_arguments() {
  auto handle = shared_device();
  const auto a = padded_zero<T>(4, 3, 4);
  auto d_a = to_device(handle, a);
  auto d_last = to_device(handle, std::vector<int>{-1});
  const std::size_t bytes = ilalr_bufferSize(4, 3);
  ASSERT_GT(bytes, 0u);
  DeviceBuffer<std::byte> d_work(bytes, handle);
  const auto stream = handle->stream().get();
  EXPECT_FALSE(ilalr<T>(stream, 4, 3, d_a.data(), 3, d_last.data(), d_work.data(), bytes).ok())
      << "lda < m";
  EXPECT_FALSE(ilalr<T>(stream, -1, 3, d_a.data(), 4, d_last.data(), d_work.data(), bytes).ok())
      << "m < 0";
  EXPECT_FALSE(ilalr<T>(stream, 4, 3, d_a.data(), 4, d_last.data(), d_work.data(), bytes - 1).ok())
      << "undersized workspace";
  EXPECT_FALSE(ilalr<T>(stream, 4, 3, d_a.data(), 4, nullptr, d_work.data(), bytes).ok())
      << "null result";
}

template<typename T>
void all_cases() {
  all_zero<T>();
  single_nonzero_everywhere<T>();
  larger_than_a_block<T>();
  special_values<T>();
  empty_is_zero<T>();
  bad_arguments<T>();
}

} // namespace

TEST(IlalrOracleTests, Float) { all_cases<float>(); }
TEST(IlalrOracleTests, Double) { all_cases<double>(); }
TEST(IlalrOracleTests, ComplexFloat) { all_cases<wwr::wwrFloatComplex>(); }
TEST(IlalrOracleTests, ComplexDouble) { all_cases<wwr::wwrDoubleComplex>(); }

} // namespace calaman
