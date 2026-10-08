// Oracle test for calaman.la_wwaddw -- the double-word accumulate
// (x, y) <-- (x, y) + w of ?LA_WWADDW.
//
// There is no LAPACKE wrapper and no LAPACK_?la_wwaddw in lapack.h, and the
// image's liblapack does not export ?la_wwaddw_ (`nm -D` finds none; the
// extra-precise routines ship only with XBLAS). So the oracle is a host port of
// the Fortran loop, statement for statement, under `fp contract(off)`.
//
// The comparison is BIT FOR BIT on both x and y, never a tolerance: the output
// that matters is the low word y, which is ~2^-24 (float) or ~2^-53 (double)
// of x. A port that dropped the compensation entirely (y unchanged) would still
// agree with the reference to within any tolerance scaled to x, so a tolerance
// check would be a false pass. Each discriminating case also asserts that y
// really changed, so an exact pass is evidence the error term survived.
//
// Cases: n = 0 (no launch, buffers untouched), n = 1, w several binades below
// x (the low word's reason to exist), w cancelling x exactly, the roles swapped
// (|w| just above |x|, error kept; |w| >> |x|, error lost as in the reference),
// signed zeros, s + s overflowing (where (s + s) - s is NOT an
// identity, so folding it away is caught), and a mixed vector over many blocks.
// All four precisions; complex runs the body per component.
//
// REQUIRES_GPU (see CMakeLists.txt): every case runs the kernel on the device.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.la_wwaddw;
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
  if (n > 0) {
    wwr::extension::copy(device, staging, handle->stream().get());
  }
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

/// @brief Per-type element construction and component access
template<typename T>
struct Elem;

template<typename R>
struct RealElem {
  using real = R;
  static constexpr std::size_t kParts = 1;
  static R make(R re, R /*im*/) { return re; }
  static std::array<R, 2> parts(R a) { return {a, R(0)}; }
};

template<>
struct Elem<float> : RealElem<float> {};
template<>
struct Elem<double> : RealElem<double> {};

template<>
struct Elem<wwr::wwrFloatComplex> {
  using real = float;
  static constexpr std::size_t kParts = 2;
  static wwr::wwrFloatComplex make(float re, float im) { return wwr::make_wwrFloatComplex(re, im); }
  static std::array<float, 2> parts(wwr::wwrFloatComplex a) {
    return {wwr::wwrCrealf(a), wwr::wwrCimagf(a)};
  }
};

template<>
struct Elem<wwr::wwrDoubleComplex> {
  using real = double;
  static constexpr std::size_t kParts = 2;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static std::array<double, 2> parts(wwr::wwrDoubleComplex a) {
    return {wwr::wwrCreal(a), wwr::wwrCimag(a)};
  }
};

/// @brief Bitwise equality of one real (so -0 != +0)
template<typename R>
bool same_bits(R a, R b) {
  return std::memcmp(&a, &b, sizeof(R)) == 0;
}

/// @brief The ?LA_WWADDW loop body on one real component, statement for statement
///
/// `fp contract(off)` keeps the host port as strict as the kernel; x86-64 SSE
/// arithmetic carries no excess precision, and no fast-math reaches this TU.
template<typename R>
void host_wwaddw(R &x, R &y, const R w) {
#pragma clang fp contract(off)
  R s = x + w;
  s = (s + s) - s;
  y = ((x - s) + w) + y;
  x = s;
}

/// @brief One element's input as real components: (x, y, w) for re and for im
template<typename R>
struct Triple {
  R x, y, w;
};

/// @brief Run the device on @p re (and @p im, for complex) and compare bitwise
///
/// @p must_move: every element's low word must change -- the case is chosen so
/// the rounding error of x + w is non-zero, which a dropped compensation misses.
template<typename T>
void check(const std::vector<Triple<typename Elem<T>::real>> &re,
           const std::vector<Triple<typename Elem<T>::real>> &im, bool must_move,
           const std::string &ctx) {
  using R = typename Elem<T>::real;
  auto handle = shared_device();
  const std::size_t n = re.size();

  std::vector<T> x(n), y(n), w(n), ref_x(n), ref_y(n);
  for (std::size_t i = 0; i < n; ++i) {
    // Real types ignore the im part; complex pairs re[i] with im[i].
    const Triple<R> b = im.empty() ? Triple<R>{} : im[i];
    x[i] = Elem<T>::make(re[i].x, b.x);
    y[i] = Elem<T>::make(re[i].y, b.y);
    w[i] = Elem<T>::make(re[i].w, b.w);
    Triple<R> a = re[i];
    Triple<R> c = b;
    host_wwaddw(a.x, a.y, a.w);
    host_wwaddw(c.x, c.y, c.w);
    ref_x[i] = Elem<T>::make(a.x, c.x);
    ref_y[i] = Elem<T>::make(a.y, c.y);
    if (must_move) {
      EXPECT_FALSE(same_bits(a.y, re[i].y))
          << ctx << ": inputs give no rounding error at " << i << " (re)";
      if (Elem<T>::kParts == 2) {
        EXPECT_FALSE(same_bits(c.y, b.y))
            << ctx << ": inputs give no rounding error at " << i << " (im)";
      }
    }
  }

  auto d_x = to_device(handle, x);
  auto d_y = to_device(handle, y);
  auto d_w = to_device(handle, w);
  const auto status = la_wwaddw<T>(handle->stream().get(), n, d_x.data(), d_y.data(), d_w.data());
  EXPECT_TRUE(status.ok()) << ctx << ": la_wwaddw returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got_x = from_device(handle, d_x, n);
  const auto got_y = from_device(handle, d_y, n);
  const auto got_w = from_device(handle, d_w, n);

  for (std::size_t i = 0; i < n; ++i) {
    const auto gx = Elem<T>::parts(got_x[i]);
    const auto gy = Elem<T>::parts(got_y[i]);
    const auto gw = Elem<T>::parts(got_w[i]);
    const auto rx = Elem<T>::parts(ref_x[i]);
    const auto ry = Elem<T>::parts(ref_y[i]);
    const auto iw = Elem<T>::parts(w[i]);
    for (std::size_t c = 0; c < Elem<T>::kParts; ++c) {
      EXPECT_TRUE(same_bits(gx[c], rx[c]))
          << ctx << ": x mismatch at " << i << "/" << c << ": got " << gx[c] << " want " << rx[c];
      EXPECT_TRUE(same_bits(gy[c], ry[c]))
          << ctx << ": y mismatch at " << i << "/" << c << ": got " << gy[c] << " want " << ry[c];
      EXPECT_TRUE(same_bits(gw[c], iw[c])) << ctx << ": w written at " << i << "/" << c;
    }
  }
}

/// @brief Run @p cases as real, or as the real part with @p im_cases as the imaginary
template<typename T>
void check_cases(const std::vector<Triple<typename Elem<T>::real>> &cases,
                 const std::vector<Triple<typename Elem<T>::real>> &im_cases, bool must_move,
                 const std::string &ctx) {
  if (Elem<T>::kParts == 1) {
    check<T>(cases, {}, must_move, ctx);
  } else {
    check<T>(cases, im_cases, must_move, ctx + "/complex");
  }
}

template<typename R>
constexpr int kMantDigits = std::numeric_limits<R>::digits;

template<typename T>
void empty_is_noop() {
  using R = typename Elem<T>::real;
  auto handle = shared_device();
  const std::vector<T> x{Elem<T>::make(R(1.5), R(-2.5))};
  auto d_x = to_device(handle, x);
  auto d_y = to_device(handle, x);
  auto d_w = to_device(handle, x);
  const auto status = la_wwaddw<T>(handle->stream().get(), 0, d_x.data(), d_y.data(), d_w.data());
  EXPECT_TRUE(status.ok()) << "n == 0: status=" << status.name();
  const auto got = from_device(handle, d_x, 1);
  const auto gp = Elem<T>::parts(got[0]);
  const auto xp = Elem<T>::parts(x[0]);
  EXPECT_TRUE(same_bits(gp[0], xp[0]) && same_bits(gp[1], xp[1])) << "n == 0 wrote x";
  // A null pointer on n == 0 is a quick return, not an error, as the reference
  // never touches the arrays.
  EXPECT_TRUE(la_wwaddw<T>(handle->stream().get(), 0, nullptr, nullptr, nullptr).ok());
  EXPECT_EQ(la_wwaddw<T>(handle->stream().get(), 1, nullptr, d_y.data(), d_w.data()),
            wwr::wwrErrorInvalidValue);
}

template<typename T>
void one_element() {
  using R = typename Elem<T>::real;
  // 1 + 3 * 2^-(p+1): x + w rounds, leaving a non-zero error for y.
  const R w = std::ldexp(R(3), -(kMantDigits<R> + 1));
  check_cases<T>({{R(1), R(0), w}}, {{R(-1), R(0), -w}}, true, "n == 1");
}

template<typename T>
void w_binades_below() {
  using R = typename Elem<T>::real;
  constexpr int p = kMantDigits<R>;
  std::vector<Triple<R>> re, im;
  // w from just under one ulp of x to well below it, both signs, with a
  // non-zero running low word: the low word is what catches what x cannot hold.
  for (int k = 0; k < 12; ++k) {
    const R x = std::ldexp(R(1) + R(k) / R(16), k - 6);
    const R w = std::ldexp(R(5 + 2 * k), k - 6 - p - (k % 4)) * (k % 2 == 0 ? R(1) : R(-1));
    const R y = std::ldexp(R(1), k - 6 - 2 * p);
    re.push_back({x, y, w});
    // -3w keeps the imaginary sum's magnitude above |x|: a sum that dropped
    // below the power-of-two x at k == 0 could land exactly, with no error.
    im.push_back({-x, -y, R(-3) * w});
  }
  check_cases<T>(re, im, true, "w several binades below x");
}

template<typename T>
void w_cancels_x() {
  using R = typename Elem<T>::real;
  constexpr int p = kMantDigits<R>;
  // s == +0 exactly, so y gains ((x - 0) + -x) == +0 and keeps its old value.
  const R lo = std::ldexp(R(1), -2 * p);
  check_cases<T>({{R(1.5), lo, R(-1.5)}, {R(-3.25), -lo, R(3.25)}},
                 {{R(-0.75), R(0), R(0.75)}, {R(7), lo, R(-7)}}, false, "w cancels x");
}

template<typename T>
void roles_swap() {
  using R = typename Elem<T>::real;
  constexpr int p = kMantDigits<R>;
  // |w| >> |x|: s takes w's place. The reference's Fast2Sum assumes |x| >= |w|,
  // so here x is absorbed and its error lost too (x - s rounds to -w): y keeps
  // its value, which the device must reproduce rather than improve on.
  std::vector<Triple<R>> re, im;
  for (int k = 1; k <= 6; ++k) {
    const R x = std::ldexp(R(3 + 2 * k), -p - k);
    const R w = std::ldexp(R(1) + R(k) / R(8), k);
    re.push_back({x, R(0), w});
    im.push_back({-x, std::ldexp(R(1), -2 * p - k), -w});
  }
  check_cases<T>(re, im, false, "|w| >> |x|");

  // |w| just above |x|: x - s and (x - s) + w stay exact, so the error survives
  // the swap and y must move.
  std::vector<Triple<R>> near_re, near_im;
  for (int k = 1; k <= 4; ++k) {
    const R x = R(1) + std::ldexp(R(2 * k - 1), 1 - p);
    near_re.push_back({x, R(0), R(1.5)});
    near_im.push_back({-x, R(0), R(-1.75)});
  }
  check_cases<T>(near_re, near_im, true, "|w| just above |x|");
}

template<typename T>
void signed_zeros() {
  using R = typename Elem<T>::real;
  std::vector<Triple<R>> re, im;
  const std::array<R, 2> zeros{R(0), -R(0)};
  for (const R x : zeros) {
    for (const R y : zeros) {
      for (const R w : zeros) {
        re.push_back({x, y, w});
        im.push_back({w, x, y});
      }
    }
  }
  check_cases<T>(re, im, false, "signed zeros");
}

template<typename T>
void doubling_overflows() {
  using R = typename Elem<T>::real;
  // x + w finite but s + s overflows: (s + s) - s is +inf, not s, so a kernel
  // that folded the round-trip away would keep a finite x.
  const R big = std::numeric_limits<R>::max() / R(4) * R(3);
  check_cases<T>({{big, R(0), R(0)}}, {{-big, R(1), R(0)}}, false, "s + s overflows");
}

template<typename T>
void many_blocks() {
  using R = typename Elem<T>::real;
  constexpr int p = kMantDigits<R>;
  std::mt19937 rng(293);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::uniform_int_distribution<int> expo(-20, 20);
  std::vector<Triple<R>> re, im;
  for (int i = 0; i < 1000; ++i) {
    const auto draw = [&] {
      const int e = expo(rng);
      const R x = std::ldexp(static_cast<R>(unit(rng)), e);
      const R w = std::ldexp(static_cast<R>(unit(rng)), e - (i % 7) * 4);
      const R y = std::ldexp(static_cast<R>(unit(rng)), e - p - 2);
      return Triple<R>{x, y, w};
    };
    re.push_back(draw());
    im.push_back(draw());
  }
  check_cases<T>(re, im, false, "1000 mixed, many blocks");
}

template<typename T>
void all_cases() {
  empty_is_noop<T>();
  one_element<T>();
  w_binades_below<T>();
  w_cancels_x<T>();
  roles_swap<T>();
  signed_zeros<T>();
  doubling_overflows<T>();
  many_blocks<T>();
}

} // namespace

TEST(LaWwaddwOracleTests, Float) {
  all_cases<float>();
}
TEST(LaWwaddwOracleTests, Double) {
  all_cases<double>();
}
TEST(LaWwaddwOracleTests, ComplexFloat) {
  all_cases<wwr::wwrFloatComplex>();
}
TEST(LaWwaddwOracleTests, ComplexDouble) {
  all_cases<wwr::wwrDoubleComplex>();
}

} // namespace calaman
