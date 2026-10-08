// Oracle test for calaman.larscl2 -- reciprocal diagonal row-scaling of a
// column-major matrix, X(i,j) <-- X(i,j) / d(i), with d real for every T.
//
// There is no LAPACKE wrapper and no LAPACK_?larscl2 in lapack.h, and the
// image's liblapack does not export ?larscl2_ (an XBLAS-only routine there), so
// the oracle is a same-precision host transcription of the Fortran loop -- as in
// test/lascl2/. The loop is one IEEE divide per element (per component for
// complex, which is what COMPLEX / REAL reduces to), correctly rounded on host
// and device alike, so the comparison is BIT FOR BIT, not to a tolerance --
// with non-trivial quotients and a d spanning ~60 binades. Each case also checks
// that a reciprocal-then-multiply would have DIFFERED somewhere, so an exact
// pass is evidence the device kept the literal divide.
//
// Every case pre-fills the whole ldx-by-n buffer with distinct sentinels and
// compares all of it, so a write into the ldx - m padding rows is caught.
// Cases: square/tall/wide with and without padding, m == 1, 1x1, a matrix wider
// than one block, m == 0 / n == 0 (no launch, buffer unchanged), and ldx < m
// (wwrErrorInvalidValue, buffer unchanged).
//
// REQUIRES_GPU (see CMakeLists.txt): every case runs the kernel on the device.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.larscl2;
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

template<typename T>
std::vector<T> from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<T> &device,
                           std::size_t n) {
  HostBuffer<T> host(n == 0 ? 1 : n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  std::vector<T> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = host.data()[i];
  }
  return out;
}

/// @brief Per-type element construction, component access, and the host divide
template<typename T>
struct Elem;

template<typename R>
struct RealElem {
  using real = R;
  static R make(double re, double /*im*/) { return static_cast<R>(re); }
  static std::array<R, 2> parts(R a) { return {a, R(0)}; }
  static R div(R a, R d) { return a / d; }
  static R mul_recip(R a, R d) { return a * (R(1) / d); }
};

template<>
struct Elem<float> : RealElem<float> {};
template<>
struct Elem<double> : RealElem<double> {};

template<>
struct Elem<wwr::wwrFloatComplex> {
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static std::array<float, 2> parts(wwr::wwrFloatComplex a) {
    return {wwr::wwrCrealf(a), wwr::wwrCimagf(a)};
  }
  static wwr::wwrFloatComplex div(wwr::wwrFloatComplex a, float d) {
    return wwr::make_wwrFloatComplex(wwr::wwrCrealf(a) / d, wwr::wwrCimagf(a) / d);
  }
  static wwr::wwrFloatComplex mul_recip(wwr::wwrFloatComplex a, float d) {
    const float r = 1.0f / d;
    return wwr::make_wwrFloatComplex(wwr::wwrCrealf(a) * r, wwr::wwrCimagf(a) * r);
  }
};

template<>
struct Elem<wwr::wwrDoubleComplex> {
  using real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static std::array<double, 2> parts(wwr::wwrDoubleComplex a) {
    return {wwr::wwrCreal(a), wwr::wwrCimag(a)};
  }
  static wwr::wwrDoubleComplex div(wwr::wwrDoubleComplex a, double d) {
    return wwr::make_wwrDoubleComplex(wwr::wwrCreal(a) / d, wwr::wwrCimag(a) / d);
  }
  static wwr::wwrDoubleComplex mul_recip(wwr::wwrDoubleComplex a, double d) {
    const double r = 1.0 / d;
    return wwr::make_wwrDoubleComplex(wwr::wwrCreal(a) * r, wwr::wwrCimag(a) * r);
  }
};

/// @brief Bitwise equality, component by component (so -0 != +0)
template<typename T>
bool same_bits(T a, T b) {
  const auto pa = Elem<T>::parts(a);
  const auto pb = Elem<T>::parts(b);
  for (std::size_t c = 0; c < 2; ++c) {
    if (std::memcmp(&pa[c], &pb[c], sizeof(pa[c])) != 0) {
      return false;
    }
  }
  return true;
}

/// @brief A distinct non-trivial value per buffer slot: inexact under division
template<typename T>
T sentinel(std::size_t k) {
  const double re = 1.0 + 0.371 * static_cast<double>(k % 97) - 0.013 * static_cast<double>(k);
  const double im = -2.25 + 0.193 * static_cast<double>(k % 31);
  return Elem<T>::make(re, im);
}

/// @brief d(i): non-power-of-two mantissas of both signs across 2^-30 .. 2^30
template<typename R>
R divisor(std::size_t i) {
  constexpr std::array<double, 5> kMantissa{1.0, 3.0, 0.1, 7.25, -5.5};
  const int exponent = (static_cast<int>((i * 7) % 13) - 6) * 5;
  return static_cast<R>(std::ldexp(kMantissa[i % kMantissa.size()], exponent));
}

/// @brief Fill, run, and compare the whole ldx-by-n buffer bit for bit
template<typename T>
void check(int m, int n, int ldx, const std::string &ctx) {
  using R = typename Elem<T>::real;
  auto handle = shared_device();
  const std::size_t size = static_cast<std::size_t>(ldx) * static_cast<std::size_t>(n);

  std::vector<T> ref(size);
  for (std::size_t k = 0; k < size; ++k) {
    ref[k] = sentinel<T>(k);
  }
  std::vector<R> d(static_cast<std::size_t>(m == 0 ? 1 : m));
  for (int i = 0; i < m; ++i) {
    d[static_cast<std::size_t>(i)] = divisor<R>(static_cast<std::size_t>(i));
  }
  auto d_x = to_device(handle, ref);
  auto d_d = to_device(handle, d);

  // Host reference: the literal ?LARSCL2 loop over rows [0, m), padding untouched.
  std::size_t recip_differs = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * ldx + i;
      const R di = d[static_cast<std::size_t>(i)];
      recip_differs += same_bits(Elem<T>::div(ref[k], di), Elem<T>::mul_recip(ref[k], di)) ? 0 : 1;
      ref[k] = Elem<T>::div(ref[k], di);
    }
  }

  const auto status = larscl2<T>(handle->stream().get(), static_cast<std::size_t>(m),
                                 static_cast<std::size_t>(n), d_d.data(), d_x.data(),
                                 static_cast<std::size_t>(ldx));
  EXPECT_TRUE(status.ok()) << ctx << ": larscl2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_x, size);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < ldx; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * ldx + i;
      EXPECT_TRUE(same_bits(got[k], ref[k])) << ctx << ": mismatch at (" << i << "," << j << ")";
    }
  }
  // A matrix of more than a handful of elements must contain a quotient the
  // reciprocal-multiply gets wrong, or the exact pass above proves nothing.
  if (static_cast<std::size_t>(m) * static_cast<std::size_t>(n) >= 16) {
    EXPECT_GT(recip_differs, 0u) << ctx << ": inputs do not discriminate a reciprocal-multiply";
  }
}

template<typename T>
void shapes() {
  check<T>(4, 4, 4, "square");
  check<T>(5, 5, 7, "square/pad");
  check<T>(6, 3, 6, "tall");
  check<T>(6, 3, 8, "tall/pad");
  check<T>(3, 6, 3, "wide");
  check<T>(3, 6, 5, "wide/pad");
  check<T>(1, 9, 1, "m == 1");
  check<T>(1, 9, 4, "m == 1/pad");
  check<T>(1, 1, 1, "1x1");
  check<T>(300, 70, 303, "taller than one block/pad");
}

template<typename T>
void empty_is_noop() {
  // No launch: the sentinel buffer survives, as the empty host loop leaves it.
  check<T>(0, 4, 5, "m == 0");
  check<T>(4, 0, 5, "n == 0");
}

template<typename T>
void bad_ldx_is_invalid_value() {
  using R = typename Elem<T>::real;
  auto handle = shared_device();
  std::vector<T> x(12);
  for (std::size_t k = 0; k < x.size(); ++k) {
    x[k] = sentinel<T>(k);
  }
  auto d_x = to_device(handle, x);
  auto d_d = to_device(handle, std::vector<R>(4, R(3)));
  const auto status = larscl2<T>(handle->stream().get(), 4, 3, d_d.data(), d_x.data(), 3);
  EXPECT_EQ(status, wwr::wwrErrorInvalidValue) << "ldx < m";
  const auto got = from_device(handle, d_x, x.size());
  for (std::size_t k = 0; k < x.size(); ++k) {
    EXPECT_TRUE(same_bits(got[k], x[k])) << "ldx < m wrote at " << k;
  }
}

template<typename T>
void all_cases() {
  shapes<T>();
  empty_is_noop<T>();
  bad_ldx_is_invalid_value<T>();
}

} // namespace

TEST(Larscl2OracleTests, Float) { all_cases<float>(); }
TEST(Larscl2OracleTests, Double) { all_cases<double>(); }
TEST(Larscl2OracleTests, ComplexFloat) { all_cases<wwr::wwrFloatComplex>(); }
TEST(Larscl2OracleTests, ComplexDouble) { all_cases<wwr::wwrDoubleComplex>(); }

} // namespace calaman
