// Oracle test for calaman.tpttr: unpacking AP into A must agree with
// LAPACKE_?tpttr BIT FOR BIT -- the routine moves elements without arithmetic.
// Each case pre-fills the whole lda-by-n A with distinct sentinels and compares
// the whole buffer, so the opposite triangle and the lda > n padding rows are
// checked untouched as well as the unpacked triangle.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.tpttr;
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
DeviceBuffer<T> to_device(const std::shared_ptr<DeviceHandle> &handle, const std::vector<T> &host) {
  const std::size_t n = std::max<std::size_t>(host.size(), 1);
  HostBuffer<T> staging(n);
  std::copy(host.begin(), host.end(), staging.data());
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, staging, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return device;
}

template<typename T>
std::vector<T> from_device(const std::shared_ptr<DeviceHandle> &handle,
                           const DeviceBuffer<T> &device, const std::size_t n) {
  HostBuffer<T> host(std::max<std::size_t>(n, 1));
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

// make builds an element from (re, im) -- im is dropped for a real T; ref
// dispatches to the oracle in the matching precision. The wwr complex types are
// layout-compatible with lapack_complex_*.
template<typename T>
struct elem;

template<>
struct elem<float> {
  static float make(double re, double) { return static_cast<float>(re); }
  static lapack_int ref(char uplo, lapack_int n, const float *ap, float *a, lapack_int lda) {
    return LAPACKE_stpttr(LAPACK_COL_MAJOR, uplo, n, ap, a, lda);
  }
};

template<>
struct elem<double> {
  static double make(double re, double) { return re; }
  static lapack_int ref(char uplo, lapack_int n, const double *ap, double *a, lapack_int lda) {
    return LAPACKE_dtpttr(LAPACK_COL_MAJOR, uplo, n, ap, a, lda);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using T = wwr::wwrFloatComplex;
  static T make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static lapack_int ref(char uplo, lapack_int n, const T *ap, T *a, lapack_int lda) {
    return LAPACKE_ctpttr(LAPACK_COL_MAJOR, uplo, n,
                          reinterpret_cast<const lapack_complex_float *>(ap),
                          reinterpret_cast<lapack_complex_float *>(a), lda);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using T = wwr::wwrDoubleComplex;
  static T make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static lapack_int ref(char uplo, lapack_int n, const T *ap, T *a, lapack_int lda) {
    return LAPACKE_ztpttr(LAPACK_COL_MAJOR, uplo, n,
                          reinterpret_cast<const lapack_complex_double *>(ap),
                          reinterpret_cast<lapack_complex_double *>(a), lda);
  }
};

// Bit-exact: the conversion is a pure copy, so even the sign of zero must match.
template<typename T>
bool bits_equal(const T &a, const T &b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template<typename T>
void expect_matches_reference(const Uplo uplo, const std::size_t n, const std::size_t lda) {
  const char uplo_c = uplo == Uplo::U ? 'U' : 'L';
  const std::size_t np = n * (n + 1) / 2;

  std::vector<T> ap(np);
  for (std::size_t k = 0; k < np; ++k) {
    ap[k] = elem<T>::make(static_cast<double>(k) + 0.5, 3.0 - static_cast<double>(k));
  }
  std::vector<T> a(lda * n);
  for (std::size_t k = 0; k < a.size(); ++k) {
    a[k] = elem<T>::make(-1000.0 - static_cast<double>(k), 7.0 + static_cast<double>(k));
  }

  std::vector<T> ref = a;
  const lapack_int info =
      elem<T>::ref(uplo_c, static_cast<lapack_int>(n), ap.data(), ref.empty() ? nullptr : ref.data(),
                   static_cast<lapack_int>(lda));
  ASSERT_EQ(info, 0) << "uplo=" << uplo_c << " n=" << n << " lda=" << lda;

  auto handle = shared_device();
  auto d_ap = to_device(handle, ap);
  auto d_a = to_device(handle, a);
  const Status status = tpttr<T>(handle->stream().get(), uplo, n, d_ap.data(), d_a.data(), lda);
  ASSERT_EQ(status, wwr::wwrSuccess) << "uplo=" << uplo_c << " n=" << n << " lda=" << lda;

  const auto got = from_device(handle, d_a, a.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    EXPECT_TRUE(bits_equal(got[k], ref[k]))
        << "uplo=" << uplo_c << " n=" << n << " lda=" << lda << " (i,j)=(" << k % lda << ","
        << k / lda << ")";
  }
}

template<typename T>
void run_matches_reference() {
  for (const Uplo uplo : {Uplo::U, Uplo::L}) {
    for (const std::size_t n : {0UZ, 1UZ, 2UZ, 3UZ, 8UZ, 13UZ, 67UZ}) {
      expect_matches_reference<T>(uplo, n, std::max<std::size_t>(n, 1)); // lda == n
      expect_matches_reference<T>(uplo, n, n + 3);                       // lda > n
    }
  }
}

} // namespace

TEST(TpttrOracleTests, MatchesReferenceFloat) {
  run_matches_reference<float>();
}

TEST(TpttrOracleTests, MatchesReferenceDouble) {
  run_matches_reference<double>();
}

TEST(TpttrOracleTests, MatchesReferenceComplexFloat) {
  run_matches_reference<wwr::wwrFloatComplex>();
}

TEST(TpttrOracleTests, MatchesReferenceComplexDouble) {
  run_matches_reference<wwr::wwrDoubleComplex>();
}

// lda < max(1, n) is the one argument error left; it enqueues nothing.
TEST(TpttrOracleTests, BadLdaIsInvalidValue) {
  auto handle = shared_device();
  const Status status =
      tpttr<double>(handle->stream().get(), Uplo::U, 4, nullptr, nullptr, 3);
  EXPECT_EQ(status, wwr::wwrErrorInvalidValue);
  const Status zero_lda =
      tpttr<double>(handle->stream().get(), Uplo::L, 0, nullptr, nullptr, 0);
  EXPECT_EQ(zero_lda, wwr::wwrErrorInvalidValue);
}

} // namespace calaman
