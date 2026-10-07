// Spec test for calaman.complex_cast -- conversion between real and complex
// device arrays. There is no reference LAPACK here: the four operations are their
// own oracle, so this checks the SPECIFICATION directly, bit for bit (the values
// are exactly representable, so == is the right comparison):
//
//   1. Round-trip. Assemble a complex array from two real planes with
//      set_real_part + set_imag_part, then split it back with get_real_part /
//      get_imag_part -- the recovered planes equal the originals exactly. This is
//      the headline test: it validates all four operations together.
//   2. set_* preserves the other component. set_real_part over a pre-filled array
//      must leave every imaginary component untouched, and set_imag_part must
//      leave every real component untouched -- the read-modify-write contract.
//   3. count == 0 is a no-op. A call with count 0 touches nothing, for both a
//      set_* (output array unchanged) and a get_* (output plane unchanged).
//   4. Strided split_planes / merge_planes. A padded block round-trips exactly
//      across three different leading dimensions, padding untouched.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the arrays on the device
// and runs the cast kernels, so `ctest -LE gpu` excludes it. It needs no
// reference LAPACK -- the spec is its own oracle -- so unlike the linalg suites
// it does not guard on calaman::lapack_reference.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.complex_cast;
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
// `real` is the component type; `make` builds a value from (re, im); `re`/`im`
// read the components; `eq` is bit-exact equality. Complex goes through the host
// wwrC* accessors in wwr.complex -- the member .x/.y is not portable to hipComplex.

template<typename T>
struct elem;

template<>
struct elem<wwr::wwrFloatComplex> {
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float re(wwr::wwrFloatComplex z) { return wwr::wwrCrealf(z); }
  static float im(wwr::wwrFloatComplex z) { return wwr::wwrCimagf(z); }
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
  static double re(wwr::wwrDoubleComplex z) { return wwr::wwrCreal(z); }
  static double im(wwr::wwrDoubleComplex z) { return wwr::wwrCimag(z); }
  static bool eq(wwr::wwrDoubleComplex a, wwr::wwrDoubleComplex b) {
    return wwr::wwrCreal(a) == wwr::wwrCreal(b) && wwr::wwrCimag(a) == wwr::wwrCimag(b);
  }
};

template<typename T>
using RealOf = typename elem<T>::real;

// ── cases ─────────────────────────────────────────────────────────────────────

// Assemble a complex array from two real planes and split it back out: every
// recovered component equals its original exactly.
template<typename T>
void round_trip() {
  using R = RealOf<T>;
  auto handle = shared_device();
  auto stream = handle->stream().get();
  const std::size_t n = 7;

  std::vector<R> re(n);
  std::vector<R> im(n);
  for (std::size_t i = 0; i < n; ++i) {
    re[i] = static_cast<R>(static_cast<double>(i) - 3.0); // ..., -1, 0, 1, ...
    im[i] = static_cast<R>(2.0 * static_cast<double>(i) + 1.0);
  }

  // Start the complex array from a sentinel so an untouched component would show.
  std::vector<T> z_init(n, elem<T>::make(-99.0, -99.0));
  auto d_z = to_device(handle, z_init);
  auto d_re = to_device(handle, re);
  auto d_im = to_device(handle, im);

  set_real_part<T>(stream, d_z.data(), d_re.data(), n);
  wwr::wwrStreamSynchronize(stream);
  set_imag_part<T>(stream, d_z.data(), d_im.data(), n);
  wwr::wwrStreamSynchronize(stream);

  const auto z = from_device(handle, d_z, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(elem<T>::re(z[i]), re[i]) << "round_trip real[" << i << "]";
    EXPECT_EQ(elem<T>::im(z[i]), im[i]) << "round_trip imag[" << i << "]";
  }

  // Split back out through the get_* path.
  std::vector<R> re_out_init(n, R{0});
  std::vector<R> im_out_init(n, R{0});
  auto d_re_out = to_device(handle, re_out_init);
  auto d_im_out = to_device(handle, im_out_init);

  get_real_part<T>(stream, d_re_out.data(), d_z.data(), n);
  wwr::wwrStreamSynchronize(stream);
  get_imag_part<T>(stream, d_im_out.data(), d_z.data(), n);
  wwr::wwrStreamSynchronize(stream);

  const auto re_got = from_device(handle, d_re_out, n);
  const auto im_got = from_device(handle, d_im_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(re_got[i], re[i]) << "round_trip get_real[" << i << "]";
    EXPECT_EQ(im_got[i], im[i]) << "round_trip get_imag[" << i << "]";
  }
}

// set_real_part overwrites only the real component; set_imag_part only the
// imaginary one. The untouched component must survive bit for bit.
template<typename T>
void set_preserves_other() {
  using R = RealOf<T>;
  auto handle = shared_device();
  auto stream = handle->stream().get();
  const std::size_t n = 5;

  // Distinct original (re, im) per element.
  std::vector<T> z0(n);
  for (std::size_t i = 0; i < n; ++i) {
    z0[i] = elem<T>::make(10.0 + static_cast<double>(i), 100.0 + static_cast<double>(i));
  }

  // set_real_part leaves imag intact.
  {
    auto d_z = to_device(handle, z0);
    std::vector<R> new_re(n);
    for (std::size_t i = 0; i < n; ++i) {
      new_re[i] = static_cast<R>(-static_cast<double>(i) - 1.0);
    }
    auto d_re = to_device(handle, new_re);
    set_real_part<T>(stream, d_z.data(), d_re.data(), n);
    wwr::wwrStreamSynchronize(stream);
    const auto z = from_device(handle, d_z, n);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(elem<T>::re(z[i]), new_re[i]) << "set_real new real[" << i << "]";
      EXPECT_EQ(elem<T>::im(z[i]), elem<T>::im(z0[i])) << "set_real kept imag[" << i << "]";
    }
  }

  // set_imag_part leaves real intact.
  {
    auto d_z = to_device(handle, z0);
    std::vector<R> new_im(n);
    for (std::size_t i = 0; i < n; ++i) {
      new_im[i] = static_cast<R>(-2.0 * static_cast<double>(i) - 1.0);
    }
    auto d_im = to_device(handle, new_im);
    set_imag_part<T>(stream, d_z.data(), d_im.data(), n);
    wwr::wwrStreamSynchronize(stream);
    const auto z = from_device(handle, d_z, n);
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_EQ(elem<T>::im(z[i]), new_im[i]) << "set_imag new imag[" << i << "]";
      EXPECT_EQ(elem<T>::re(z[i]), elem<T>::re(z0[i])) << "set_imag kept real[" << i << "]";
    }
  }
}

// count == 0 launches nothing: neither a set_* nor a get_* touches its output.
template<typename T>
void zero_count_noop() {
  using R = RealOf<T>;
  auto handle = shared_device();
  auto stream = handle->stream().get();
  const std::size_t n = 3;

  std::vector<T> z0(n);
  for (std::size_t i = 0; i < n; ++i) {
    z0[i] = elem<T>::make(1.0 + static_cast<double>(i), 7.0 - static_cast<double>(i));
  }
  auto d_z = to_device(handle, z0);

  std::vector<R> in(n, static_cast<R>(42.0));
  auto d_in = to_device(handle, in);

  // A set_* with count 0 must not alter the complex array.
  set_real_part<T>(stream, d_z.data(), d_in.data(), 0);
  set_imag_part<T>(stream, d_z.data(), d_in.data(), 0);
  wwr::wwrStreamSynchronize(stream);
  const auto z = from_device(handle, d_z, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_TRUE(elem<T>::eq(z[i], z0[i])) << "zero_count set touched z[" << i << "]";
  }

  // A get_* with count 0 must not alter the real output plane.
  std::vector<R> out0(n, static_cast<R>(-5.0));
  auto d_out = to_device(handle, out0);
  get_real_part<T>(stream, d_out.data(), d_z.data(), 0);
  wwr::wwrStreamSynchronize(stream);
  const auto out = from_device(handle, d_out, n);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(out[i], out0[i]) << "zero_count get touched out[" << i << "]";
  }
}

// Strided pair: split a padded rows-by-cols block into planes with their own
// leading dimension, merge it into a third leading dimension; every in-range
// component comes back exactly and every padding slot of each output is untouched.
template<typename T>
void strided_round_trip() {
  using R = RealOf<T>;
  auto handle = shared_device();
  auto stream = handle->stream().get();
  const std::size_t rows = 5, cols = 3, lda = 7, ldp = 6, ldc = 9;
  const R pad{-55};

  std::vector<T> a(lda * cols, elem<T>::make(-99.0, -99.0));
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < rows; ++i) {
      a[i + j * lda] =
          elem<T>::make(static_cast<double>(i) - 2.0, 10.0 * static_cast<double>(j) + 1.0);
    }
  }
  auto d_a = to_device(handle, a);
  auto d_re = to_device(handle, std::vector<R>(ldp * cols, pad));
  auto d_im = to_device(handle, std::vector<R>(ldp * cols, pad));
  auto d_c = to_device(handle, std::vector<T>(ldc * cols, elem<T>::make(-7.0, -7.0)));

  split_planes<T>(stream, rows, cols, d_a.data(), lda, d_re.data(), d_im.data(), ldp);
  merge_planes<T>(stream, rows, cols, d_re.data(), d_im.data(), ldp, d_c.data(), ldc);
  wwr::wwrStreamSynchronize(stream);

  const auto re = from_device(handle, d_re, ldp * cols);
  const auto im = from_device(handle, d_im, ldp * cols);
  const auto c = from_device(handle, d_c, ldc * cols);
  for (std::size_t k = 0; k < ldp * cols; ++k) {
    const std::size_t i = k % ldp, j = k / ldp;
    const bool in = i < rows;
    EXPECT_EQ(re[k], in ? elem<T>::re(a[i + j * lda]) : pad) << "split re slot " << k;
    EXPECT_EQ(im[k], in ? elem<T>::im(a[i + j * lda]) : pad) << "split im slot " << k;
  }
  for (std::size_t k = 0; k < ldc * cols; ++k) {
    const std::size_t i = k % ldc, j = k / ldc;
    const T want = i < rows ? a[i + j * lda] : elem<T>::make(-7.0, -7.0);
    EXPECT_TRUE(elem<T>::eq(c[k], want)) << "merge slot " << k;
  }

  // rows == 0 or cols == 0 launches nothing.
  split_planes<T>(stream, 0, cols, d_c.data(), ldc, d_re.data(), d_im.data(), ldp);
  merge_planes<T>(stream, rows, 0, d_re.data(), d_im.data(), ldp, d_c.data(), ldc);
  wwr::wwrStreamSynchronize(stream);
  EXPECT_EQ(from_device(handle, d_re, ldp * cols), re);
}

} // namespace

TEST(ComplexCastSpecTests, StridedSplitMerge) {
  strided_round_trip<wwr::wwrFloatComplex>();
  strided_round_trip<wwr::wwrDoubleComplex>();
}

TEST(ComplexCastSpecTests, RoundTrip) {
  round_trip<wwr::wwrFloatComplex>();
  round_trip<wwr::wwrDoubleComplex>();
}

TEST(ComplexCastSpecTests, SetPreservesOther) {
  set_preserves_other<wwr::wwrFloatComplex>();
  set_preserves_other<wwr::wwrDoubleComplex>();
}

TEST(ComplexCastSpecTests, ZeroCountNoOp) {
  zero_count_noop<wwr::wwrFloatComplex>();
  zero_count_noop<wwr::wwrDoubleComplex>();
}

} // namespace calaman
