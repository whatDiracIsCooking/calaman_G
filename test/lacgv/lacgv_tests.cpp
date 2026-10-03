// Oracle test for calaman.lacgv: the conjugated vector must agree with reference
// LAPACK -- LAPACKE_clacgv / LAPACKE_zlacgv -- computed in the SAME precision on
// the host. Conjugation only negates the imaginary component, which is exact in
// floating point, so the comparison is BIT-EXACT (==), not a tolerance: a wrong
// sign, a missed element, or a touched gap all show immediately.
//
// Each case stages the WHOLE strided buffer (1 + (n-1)*|incx| elements) on the
// device and compares the full buffer back against the reference. That checks two
// things at once: the n strided elements are conjugated, and the gap elements
// between them (for non-unit stride) are left untouched -- the reference leaves
// them untouched too, so an identical buffer proves both. Both the reference and
// the device wrapper take the buffer's start pointer and the same incx, including
// negative incx (CLACGV's internal ioff walks from the far end).
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and runs
// the kernel, so the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise, so
// its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lacgv;
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

// ── host element helpers, one per complex element type ────────────────────────
//
// `make` builds a value from (re, im); `eq` is bit-exact equality; `ref_lacgv`
// dispatches to the reference oracle in the matching precision. The wwr complex
// types are layout-compatible with lapack_complex_*, so the oracle takes a
// reinterpret_cast of the same buffer (the cg_unitary suite does the same).

template<typename T>
struct elem;

template<>
struct elem<wwr::wwrFloatComplex> {
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static bool eq(wwr::wwrFloatComplex a, wwr::wwrFloatComplex b) {
    return wwr::wwrCrealf(a) == wwr::wwrCrealf(b) && wwr::wwrCimagf(a) == wwr::wwrCimagf(b);
  }
  static void ref_lacgv(int n, wwr::wwrFloatComplex *x, int incx) {
    LAPACKE_clacgv(n, reinterpret_cast<lapack_complex_float *>(x), incx);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static bool eq(wwr::wwrDoubleComplex a, wwr::wwrDoubleComplex b) {
    return wwr::wwrCreal(a) == wwr::wwrCreal(b) && wwr::wwrCimag(a) == wwr::wwrCimag(b);
  }
  static void ref_lacgv(int n, wwr::wwrDoubleComplex *x, int incx) {
    LAPACKE_zlacgv(n, reinterpret_cast<lapack_complex_double *>(x), incx);
  }
};

// ── cases ─────────────────────────────────────────────────────────────────────

// One (n, incx) case: lacgv on the device must match the reference bit for bit,
// over the whole strided buffer (so the untouched gap elements are checked too).
template<typename T>
void expect_matches_reference(int n, int incx) {
  ASSERT_GE(n, 1);
  ASSERT_NE(incx, 0);
  const std::size_t size =
      static_cast<std::size_t>(1 + (n - 1) * (incx < 0 ? -incx : incx));

  // Distinct value per slot, gaps included, so a touched gap or a wrong element
  // is visible. Non-trivial imaginary parts so a missed conjugation shows.
  std::vector<T> input(size);
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = elem<T>::make(static_cast<double>(i) - 4.0, 2.0 * static_cast<double>(i) + 1.0);
  }

  // Reference overwrites its own copy in place.
  std::vector<T> ref = input;
  elem<T>::ref_lacgv(n, ref.data(), incx);

  auto handle = shared_device();
  auto d_x = to_device(handle, input);
  const auto status = lacgv<T>(handle->stream().get(), n, d_x.data(), incx);
  ASSERT_EQ(status, wwr::wwrSuccess) << "n=" << n << " incx=" << incx;

  const auto got = from_device(handle, d_x, size);
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_TRUE(elem<T>::eq(got[i], ref[i])) << "n=" << n << " incx=" << incx << " slot=" << i;
  }
}

template<typename T>
void run_matches_reference() {
  expect_matches_reference<T>(1, 1);     // single element
  expect_matches_reference<T>(7, 1);     // unit stride
  expect_matches_reference<T>(64, 1);    // unit stride, multi-block
  expect_matches_reference<T>(129, 1);   // unit stride, odd multi-block
  expect_matches_reference<T>(7, 2);     // non-unit stride (gaps present)
  expect_matches_reference<T>(5, 3);     // non-unit stride
  expect_matches_reference<T>(7, -1);    // negative unit stride
  expect_matches_reference<T>(5, -2);    // negative non-unit stride
}

} // namespace

TEST(LacgvOracleTests, MatchesReferenceFloat) {
  run_matches_reference<wwr::wwrFloatComplex>();
}

TEST(LacgvOracleTests, MatchesReferenceDouble) {
  run_matches_reference<wwr::wwrDoubleComplex>();
}

// n <= 0: lacgv enqueues nothing, returns success, and leaves the buffer
// untouched -- the originals must survive bit for bit.
template<typename T>
void expect_nonpositive_n_is_noop(int n) {
  auto handle = shared_device();
  const std::size_t size = 4;
  std::vector<T> input(size);
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = elem<T>::make(1.0 + static_cast<double>(i), 7.0 - static_cast<double>(i));
  }
  auto d_x = to_device(handle, input);
  const auto status = lacgv<T>(handle->stream().get(), n, d_x.data(), 1);
  ASSERT_EQ(status, wwr::wwrSuccess) << "n=" << n;

  const auto got = from_device(handle, d_x, size);
  for (std::size_t i = 0; i < size; ++i) {
    EXPECT_TRUE(elem<T>::eq(got[i], input[i])) << "n=" << n << " slot=" << i;
  }
}

TEST(LacgvOracleTests, NonPositiveNIsNoop) {
  expect_nonpositive_n_is_noop<wwr::wwrFloatComplex>(0);
  expect_nonpositive_n_is_noop<wwr::wwrDoubleComplex>(0);
  expect_nonpositive_n_is_noop<wwr::wwrDoubleComplex>(-3);
}

} // namespace calaman
