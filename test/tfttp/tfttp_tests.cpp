// Oracle test for calaman.tfttp: converting RFP ARF to packed AP must agree
// with LAPACKE_?tfttp BIT FOR BIT -- the routine only moves (and, for complex,
// conjugates) elements. Every TRANSR x UPLO x n-parity case runs. AP starts as
// distinct sentinels, so a slot left unwritten shows up; ARF is checked
// unchanged.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.tfttp;
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
  static lapack_int ref(char tr, char uplo, lapack_int n, const float *arf, float *ap) {
    return LAPACKE_stfttp(LAPACK_COL_MAJOR, tr, uplo, n, arf, ap);
  }
};

template<>
struct elem<double> {
  static constexpr Trans kTransposed = Trans::T;
  static double make(double re, double) { return re; }
  static lapack_int ref(char tr, char uplo, lapack_int n, const double *arf, double *ap) {
    return LAPACKE_dtfttp(LAPACK_COL_MAJOR, tr, uplo, n, arf, ap);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using T = wwr::wwrFloatComplex;
  static constexpr Trans kTransposed = Trans::C;
  static T make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static lapack_int ref(char tr, char uplo, lapack_int n, const T *arf, T *ap) {
    return LAPACKE_ctfttp(LAPACK_COL_MAJOR, tr, uplo, n,
                          reinterpret_cast<const lapack_complex_float *>(arf),
                          reinterpret_cast<lapack_complex_float *>(ap));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using T = wwr::wwrDoubleComplex;
  static constexpr Trans kTransposed = Trans::C;
  static T make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static lapack_int ref(char tr, char uplo, lapack_int n, const T *arf, T *ap) {
    return LAPACKE_ztfttp(LAPACK_COL_MAJOR, tr, uplo, n,
                          reinterpret_cast<const lapack_complex_double *>(arf),
                          reinterpret_cast<lapack_complex_double *>(ap));
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
void expect_matches_reference(const Trans transr, const Uplo uplo, const std::size_t n) {
  const char tr_c = trans_char(transr);
  const char uplo_c = uplo == Uplo::U ? 'U' : 'L';
  const std::size_t np = n * (n + 1) / 2;

  std::vector<T> arf(np);
  std::vector<T> ap(np);
  for (std::size_t k = 0; k < np; ++k) {
    arf[k] = elem<T>::make(static_cast<double>(k) + 0.5, 3.0 - static_cast<double>(k));
    ap[k] = elem<T>::make(-1000.0 - static_cast<double>(k), 7.0 + static_cast<double>(k));
  }

  std::vector<T> ref = ap;
  const lapack_int info = elem<T>::ref(tr_c, uplo_c, static_cast<lapack_int>(n),
                                       arf.empty() ? nullptr : arf.data(),
                                       ref.empty() ? nullptr : ref.data());
  ASSERT_EQ(info, 0) << "transr=" << tr_c << " uplo=" << uplo_c << " n=" << n;

  auto handle = shared_device();
  auto d_arf = to_device(handle, arf);
  auto d_ap = to_device(handle, ap);
  const Status status = tfttp<T>(handle->stream().get(), transr, uplo, n, d_arf.data(), d_ap.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << "transr=" << tr_c << " uplo=" << uplo_c << " n=" << n;

  const auto got = from_device(handle, d_ap, np);
  const auto arf_after = from_device(handle, d_arf, np);
  for (std::size_t k = 0; k < np; ++k) {
    EXPECT_TRUE(bits_equal(got[k], ref[k]))
        << "transr=" << tr_c << " uplo=" << uplo_c << " n=" << n << " ap slot=" << k;
    EXPECT_TRUE(bits_equal(arf_after[k], arf[k])) << "ARF modified at slot " << k;
  }
}

template<typename T>
void run_matches_reference() {
  for (const Trans transr : {Trans::N, elem<T>::kTransposed}) {
    for (const Uplo uplo : {Uplo::U, Uplo::L}) {
      // Both parities, each at a size where the corner triangle is non-trivial.
      for (const std::size_t n : {0UZ, 1UZ, 2UZ, 3UZ, 4UZ, 5UZ, 8UZ, 13UZ, 64UZ, 67UZ}) {
        expect_matches_reference<T>(transr, uplo, n);
      }
    }
  }
}

} // namespace

TEST(TfttpOracleTests, MatchesReferenceFloat) {
  run_matches_reference<float>();
}

TEST(TfttpOracleTests, MatchesReferenceDouble) {
  run_matches_reference<double>();
}

TEST(TfttpOracleTests, MatchesReferenceComplexFloat) {
  run_matches_reference<wwr::wwrFloatComplex>();
}

TEST(TfttpOracleTests, MatchesReferenceComplexDouble) {
  run_matches_reference<wwr::wwrDoubleComplex>();
}

// The one argument error: the TRANSR LAPACK rejects for the type ('C' for
// real, 'T' for complex). It enqueues nothing.
TEST(TfttpOracleTests, BadTransrIsInvalidValue) {
  auto handle = shared_device();
  const auto stream = handle->stream().get();
  EXPECT_EQ(tfttp<float>(stream, Trans::C, Uplo::U, 4, nullptr, nullptr),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(tfttp<double>(stream, Trans::C, Uplo::L, 4, nullptr, nullptr),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(tfttp<wwr::wwrDoubleComplex>(stream, Trans::T, Uplo::L, 4, nullptr, nullptr),
            wwr::wwrErrorInvalidValue);
}

} // namespace calaman
