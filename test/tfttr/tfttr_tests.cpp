// Oracle test for calaman.tfttr: converting RFP ARF into A must agree with
// LAPACKE_?tfttr BIT FOR BIT -- the routine only moves (and, for complex,
// conjugates) elements. Every TRANSR x UPLO x n-parity case runs. Each case
// pre-fills the whole lda-by-n A with distinct sentinels and compares the whole
// buffer, so the opposite triangle and the lda > n padding rows are checked
// untouched as well as the converted triangle.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.tfttr;
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
// dispatches to the oracle in the matching precision; kTransposed is the
// TRANSR LAPACK accepts for the type. The wwr complex types are
// layout-compatible with lapack_complex_*.
template<typename T>
struct elem;

template<>
struct elem<float> {
  static constexpr Trans kTransposed = Trans::T;
  static float make(double re, double) { return static_cast<float>(re); }
  static lapack_int ref(char tr, char uplo, lapack_int n, const float *arf, float *a,
                        lapack_int lda) {
    return LAPACKE_stfttr(LAPACK_COL_MAJOR, tr, uplo, n, arf, a, lda);
  }
};

template<>
struct elem<double> {
  static constexpr Trans kTransposed = Trans::T;
  static double make(double re, double) { return re; }
  static lapack_int ref(char tr, char uplo, lapack_int n, const double *arf, double *a,
                        lapack_int lda) {
    return LAPACKE_dtfttr(LAPACK_COL_MAJOR, tr, uplo, n, arf, a, lda);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using T = wwr::wwrFloatComplex;
  static constexpr Trans kTransposed = Trans::C;
  static T make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static lapack_int ref(char tr, char uplo, lapack_int n, const T *arf, T *a, lapack_int lda) {
    return LAPACKE_ctfttr(LAPACK_COL_MAJOR, tr, uplo, n,
                          reinterpret_cast<const lapack_complex_float *>(arf),
                          reinterpret_cast<lapack_complex_float *>(a), lda);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using T = wwr::wwrDoubleComplex;
  static constexpr Trans kTransposed = Trans::C;
  static T make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static lapack_int ref(char tr, char uplo, lapack_int n, const T *arf, T *a, lapack_int lda) {
    return LAPACKE_ztfttr(LAPACK_COL_MAJOR, tr, uplo, n,
                          reinterpret_cast<const lapack_complex_double *>(arf),
                          reinterpret_cast<lapack_complex_double *>(a), lda);
  }
};

// Bit-exact: the conversion is a copy or a conjugate, both exact.
template<typename T>
bool bits_equal(const T &a, const T &b) {
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

char trans_char(const Trans t) {
  return t == Trans::N ? 'N' : (t == Trans::T ? 'T' : 'C');
}

template<typename T>
void expect_matches_reference(const Trans transr, const Uplo uplo, const std::size_t n,
                              const std::size_t lda) {
  const char tr_c = trans_char(transr);
  const char uplo_c = uplo == Uplo::U ? 'U' : 'L';
  const std::size_t np = n * (n + 1) / 2;

  std::vector<T> arf(np);
  for (std::size_t k = 0; k < np; ++k) {
    arf[k] = elem<T>::make(static_cast<double>(k) + 0.5, 3.0 - static_cast<double>(k));
  }
  std::vector<T> a(lda * n);
  for (std::size_t k = 0; k < a.size(); ++k) {
    a[k] = elem<T>::make(-1000.0 - static_cast<double>(k), 7.0 + static_cast<double>(k));
  }

  std::vector<T> ref = a;
  const lapack_int info =
      elem<T>::ref(tr_c, uplo_c, static_cast<lapack_int>(n), arf.empty() ? nullptr : arf.data(),
                   ref.empty() ? nullptr : ref.data(), static_cast<lapack_int>(lda));
  ASSERT_EQ(info, 0) << "transr=" << tr_c << " uplo=" << uplo_c << " n=" << n << " lda=" << lda;

  auto handle = shared_device();
  auto d_arf = to_device(handle, arf);
  auto d_a = to_device(handle, a);
  const Status status =
      tfttr<T>(handle->stream().get(), transr, uplo, n, d_arf.data(), d_a.data(), lda);
  ASSERT_EQ(status, wwr::wwrSuccess)
      << "transr=" << tr_c << " uplo=" << uplo_c << " n=" << n << " lda=" << lda;

  const auto got = from_device(handle, d_a, a.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    EXPECT_TRUE(bits_equal(got[k], ref[k]))
        << "transr=" << tr_c << " uplo=" << uplo_c << " n=" << n << " lda=" << lda << " (i,j)=("
        << k % lda << "," << k / lda << ")";
  }
}

template<typename T>
void run_matches_reference() {
  for (const Trans transr : {Trans::N, elem<T>::kTransposed}) {
    for (const Uplo uplo : {Uplo::U, Uplo::L}) {
      // Both parities, each at a size where the corner triangle is non-trivial.
      for (const std::size_t n : {0UZ, 1UZ, 2UZ, 3UZ, 4UZ, 5UZ, 8UZ, 13UZ, 64UZ, 67UZ}) {
        expect_matches_reference<T>(transr, uplo, n, std::max<std::size_t>(n, 1)); // lda == n
        expect_matches_reference<T>(transr, uplo, n, n + 3);                       // lda > n
      }
    }
  }
}

} // namespace

TEST(TfttrOracleTests, MatchesReferenceFloat) {
  run_matches_reference<float>();
}

TEST(TfttrOracleTests, MatchesReferenceDouble) {
  run_matches_reference<double>();
}

TEST(TfttrOracleTests, MatchesReferenceComplexFloat) {
  run_matches_reference<wwr::wwrFloatComplex>();
}

TEST(TfttrOracleTests, MatchesReferenceComplexDouble) {
  run_matches_reference<wwr::wwrDoubleComplex>();
}

// The argument errors: lda < max(1, n), and the TRANSR LAPACK rejects for the
// type ('C' for real, 'T' for complex). Each enqueues nothing.
TEST(TfttrOracleTests, BadArgumentsAreInvalidValue) {
  auto handle = shared_device();
  const auto stream = handle->stream().get();
  EXPECT_EQ(tfttr<double>(stream, Trans::N, Uplo::U, 4, nullptr, nullptr, 3),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(tfttr<double>(stream, Trans::N, Uplo::L, 0, nullptr, nullptr, 0),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(tfttr<float>(stream, Trans::C, Uplo::U, 4, nullptr, nullptr, 4),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(tfttr<wwr::wwrDoubleComplex>(stream, Trans::T, Uplo::L, 4, nullptr, nullptr, 4),
            wwr::wwrErrorInvalidValue);
}

} // namespace calaman
