// Spec test for calaman.set_element -- read one element of a device array by a
// device-held 1-based index. The spec is its own oracle, so there is no
// reference LAPACK here:
//
//   1. Gather. set_element copies d_x[(idx-1)*incx] to the result, bit for bit
//      (the values are exactly representable, so == is right), for a manually
//      placed device index -- unit stride and a strided vector, real and complex.
//   2. Magnitude. set_element_abs writes |d_x[(idx-1)*incx]|; the expected value
//      is the host modulus of the same element (exact for a real element, within
//      a relative tolerance for a complex sqrt).
//   3. iamax integration. The real reason the module exists: with a BLAS handle
//      in DEVICE pointer mode, iamax writes its 1-based index to the device, and
//      set_element_abs turns that device index into the max-magnitude value --
//      which must equal the host's max |element|. This proves the 1-based
//      convention matches iamax end to end.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the arrays on the device
// and runs the kernel, so `ctest -LE gpu` excludes it. It needs no reference
// LAPACK -- the spec is its own oracle -- so unlike the linalg suites it does
// not guard on calaman::lapack_reference.

#include <gtest/gtest.h>

import std;

import wwr.blas;          // wwrblasHandle_t, device pointer mode, WWRBLAS_STATUS_*
import wwr.wrappers.blas; // iamax
import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.common; // ComplexToRealType
import wwr.extension.memory_buffer;
import calaman.set_element;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;

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

// ── host element helpers, one per element type ────────────────────────────────
//
// `real` is the component type; `make` builds a value from (re, im) -- im is
// ignored for a real element; `abs` is the host modulus; `eq` is bit-exact
// equality. Complex goes through the host wwrC* accessors in wwr.complex -- the
// member .x/.y is not portable to hipComplex.

template<typename T>
struct elem;

template<>
struct elem<float> {
  using real = float;
  static float make(double re, double /*im*/) { return static_cast<float>(re); }
  static float abs(float v) { return v < 0.0f ? -v : v; }
  static bool eq(float a, float b) { return a == b; }
};

template<>
struct elem<double> {
  using real = double;
  static double make(double re, double /*im*/) { return re; }
  static double abs(double v) { return v < 0.0 ? -v : v; }
  static bool eq(double a, double b) { return a == b; }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float abs(wwr::wwrFloatComplex z) {
    const double re = wwr::wwrCrealf(z);
    const double im = wwr::wwrCimagf(z);
    return static_cast<float>(std::sqrt(re * re + im * im));
  }
  static bool eq(wwr::wwrFloatComplex a, wwr::wwrFloatComplex b) {
    return wwr::wwrCrealf(a) == wwr::wwrCrealf(b) && wwr::wwrCimagf(a) == wwr::wwrCimagf(b);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double abs(wwr::wwrDoubleComplex z) {
    const double re = wwr::wwrCreal(z);
    const double im = wwr::wwrCimag(z);
    return std::sqrt(re * re + im * im);
  }
  static bool eq(wwr::wwrDoubleComplex a, wwr::wwrDoubleComplex b) {
    return wwr::wwrCreal(a) == wwr::wwrCreal(b) && wwr::wwrCimag(a) == wwr::wwrCimag(b);
  }
};

template<typename T>
using RealOf = typename elem<T>::real;

// A relative tolerance for the magnitude's sqrt (exact for a real element, so
// this only matters for complex); tight enough to catch a wrong element.
template<typename R>
R mag_tol(R ref) {
  const R rel = std::is_same_v<R, float> ? R(1e-5) : R(1e-12);
  const R atol = std::is_same_v<R, float> ? R(1e-6) : R(1e-13);
  return rel * (ref < R{0} ? -ref : ref) + atol;
}

// Place a 1-based index on the device for the manual cases.
DeviceBuffer<int> index_on_device(std::shared_ptr<DeviceHandle> handle, int one_based) {
  return to_device(handle, std::vector<int>{one_based});
}

// ── cases ─────────────────────────────────────────────────────────────────────

// set_element copies the indexed element exactly, for a chosen 1-based index and
// stride. Builds a length-(n*incx) buffer whose stride positions carry distinct
// values and whose gaps carry a sentinel, so a wrong stride would be caught.
template<typename T>
void gather_one(int n, int incx, int one_based) {
  using R = RealOf<T>;
  auto handle = std::make_shared<DeviceHandle>(0);
  auto stream = handle->stream().get();
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(incx);

  std::vector<T> x(total, elem<T>::make(-777.0, -888.0)); // gap sentinel
  for (int k = 0; k < n; ++k) {
    const std::size_t i = static_cast<std::size_t>(k) * static_cast<std::size_t>(incx);
    x[i] = elem<T>::make(static_cast<double>(k) - 4.0, 2.0 * static_cast<double>(k) + 1.0);
  }
  const T expected = x[static_cast<std::size_t>(one_based - 1) * static_cast<std::size_t>(incx)];

  auto d_x = to_device(handle, x);
  auto d_idx = index_on_device(handle, one_based);
  std::vector<T> result_init{elem<T>::make(0.0, 0.0)};
  auto d_result = to_device(handle, result_init);

  set_element<T>(stream, d_x.data(), incx, d_idx.data(), d_result.data());
  wwr::wwrStreamSynchronize(stream);

  const auto got = from_device(handle, d_result, 1);
  EXPECT_TRUE(elem<T>::eq(got[0], expected))
      << "gather n=" << n << " incx=" << incx << " idx=" << one_based;
}

// set_element_abs writes the host modulus of the indexed element.
template<typename T>
void magnitude_one(int n, int incx, int one_based) {
  using R = RealOf<T>;
  auto handle = std::make_shared<DeviceHandle>(0);
  auto stream = handle->stream().get();
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(incx);

  std::vector<T> x(total, elem<T>::make(0.0, 0.0));
  for (int k = 0; k < n; ++k) {
    const std::size_t i = static_cast<std::size_t>(k) * static_cast<std::size_t>(incx);
    x[i] = elem<T>::make(static_cast<double>(k) - 4.0, 3.0 - static_cast<double>(k));
  }
  const R expected =
      elem<T>::abs(x[static_cast<std::size_t>(one_based - 1) * static_cast<std::size_t>(incx)]);

  auto d_x = to_device(handle, x);
  auto d_idx = index_on_device(handle, one_based);
  std::vector<R> result_init{R{-1}};
  auto d_result = to_device(handle, result_init);

  set_element_abs<T>(stream, d_x.data(), incx, d_idx.data(), d_result.data());
  wwr::wwrStreamSynchronize(stream);

  const auto got = from_device(handle, d_result, 1);
  EXPECT_NEAR(got[0], expected, mag_tol<R>(expected))
      << "magnitude n=" << n << " incx=" << incx << " idx=" << one_based;
}

// The headline composition: iamax in DEVICE pointer mode writes its 1-based index
// to the device, set_element_abs turns it into the max-magnitude value, and that
// must equal the host's max |element|.
template<typename T>
void iamax_device_mode(int n, int incx) {
  using R = RealOf<T>;
  auto handle = std::make_shared<DeviceHandle>(0);
  auto stream = handle->stream().get();
  const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(incx);

  // A clear, unique maximum so iamax's choice is unambiguous.
  std::vector<T> x(total, elem<T>::make(0.0, 0.0));
  R host_max = R{-1};
  for (int k = 0; k < n; ++k) {
    const std::size_t i = static_cast<std::size_t>(k) * static_cast<std::size_t>(incx);
    const double base = static_cast<double>((k * 7) % 11) + 1.0; // 1..11, varied
    x[i] = elem<T>::make(base, base / 2.0);
    host_max = std::max(host_max, elem<T>::abs(x[i]));
  }

  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, stream), wwr::WWRBLAS_STATUS_SUCCESS);
  // DEVICE pointer mode: iamax's index lands on the device, where only a kernel
  // can read it -- which is exactly what set_element is for.
  ASSERT_EQ(wwr::wwrblasSetPointerMode(blas, wwr::WWRBLAS_POINTER_MODE_DEVICE),
            wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_x = to_device(handle, x);
  auto d_idx = to_device(handle, std::vector<int>{0});
  std::vector<R> result_init{R{-1}};
  auto d_result = to_device(handle, result_init);

  ASSERT_EQ(wwr::iamax<T>(blas, n, d_x.data(), incx, d_idx.data()), wwr::WWRBLAS_STATUS_SUCCESS);
  set_element_abs<T>(stream, d_x.data(), incx, d_idx.data(), d_result.data());
  wwr::wwrStreamSynchronize(stream);
  wwr::wwrblasDestroy(blas);

  const auto got = from_device(handle, d_result, 1);
  EXPECT_NEAR(got[0], host_max, mag_tol<R>(host_max)) << "iamax n=" << n << " incx=" << incx;
}

template<typename T>
void run_gather() {
  gather_one<T>(1, 1, 1);
  gather_one<T>(8, 1, 1);
  gather_one<T>(8, 1, 8);
  gather_one<T>(8, 1, 5);
  gather_one<T>(6, 3, 4); // strided
}

template<typename T>
void run_magnitude() {
  magnitude_one<T>(1, 1, 1);
  magnitude_one<T>(9, 1, 1);
  magnitude_one<T>(9, 1, 9);
  magnitude_one<T>(9, 1, 6);
  magnitude_one<T>(6, 2, 3); // strided
}

template<typename T>
void run_iamax() {
  iamax_device_mode<T>(1, 1);
  iamax_device_mode<T>(16, 1);
  iamax_device_mode<T>(1000, 1);
  iamax_device_mode<T>(100, 2); // strided
}

} // namespace

TEST(SetElementSpecTests, GatherReal) {
  run_gather<float>();
  run_gather<double>();
}

TEST(SetElementSpecTests, GatherComplex) {
  run_gather<wwr::wwrFloatComplex>();
  run_gather<wwr::wwrDoubleComplex>();
}

TEST(SetElementSpecTests, MagnitudeReal) {
  run_magnitude<float>();
  run_magnitude<double>();
}

TEST(SetElementSpecTests, MagnitudeComplex) {
  run_magnitude<wwr::wwrFloatComplex>();
  run_magnitude<wwr::wwrDoubleComplex>();
}

TEST(SetElementSpecTests, IamaxDevicePointerMode) {
  run_iamax<float>();
  run_iamax<double>();
  run_iamax<wwr::wwrFloatComplex>();
  run_iamax<wwr::wwrDoubleComplex>();
}

} // namespace calaman
