// Oracle test for calaman.la_geamv -- y := alpha * |op(A)| * |x| + beta * |y|
// with ?LA_GEAMV's symbolic-zero nudge, y(i) += sign(safe1, y(i)) unless every
// product feeding y(i) had a zero multiplicand.
//
// There is no LAPACKE wrapper and no LAPACK_?la_geamv in lapack.h, and the
// image's liblapack does not export ?la_geamv_ (XBLAS-only there), so the
// oracle is a same-precision host transcription of the Fortran loop, statement
// for statement -- each multiply and add its own statement, so the host cannot
// contract them. The device rounds every operation the same way, so the
// comparison is BIT FOR BIT: an exact 0 against an exact safe1 is visible.
//
// The symbolic-zero path is targeted directly: A has an exactly zero row and
// column, and a row (and column) whose non-zeros meet only the exact zeros of
// x; y carries exact zeros too. A separate case drives every product into
// underflow, where the result must be safe1 exactly, not 0. x's stride gaps and
// A's padding rows are NaN, so a stray read shows; y's gaps hold sentinels that
// must survive. Also: all three Trans for every type, alpha/beta of 0, 1 and
// general values, both stride signs, m == 0 / n == 0, the alpha == 0 && beta ==
// 1 quick return, and the invalid-argument rejections.
//
// REQUIRES_GPU (see CMakeLists.txt): every case runs the kernel on the device.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.la_geamv;
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

/// @brief Per-type construction and the host CABS1
template<typename T>
struct Elem;

template<typename R>
struct RealElem {
  using real = R;
  static R make(double re, double /*im*/) { return static_cast<R>(re); }
  static R abs1(R a) { return std::fabs(a); }
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
  static float abs1(wwr::wwrFloatComplex a) {
    const float re = std::fabs(wwr::wwrCrealf(a));
    const float im = std::fabs(wwr::wwrCimagf(a));
    return re + im;
  }
};

template<>
struct Elem<wwr::wwrDoubleComplex> {
  using real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static double abs1(wwr::wwrDoubleComplex a) {
    const double re = std::fabs(wwr::wwrCreal(a));
    const double im = std::fabs(wwr::wwrCimag(a));
    return re + im;
  }
};

template<typename R>
bool same_bits(R a, R b) {
  return std::memcmp(&a, &b, sizeof(R)) == 0;
}

/// @brief Buffer length of a strided vector of @p len elements
std::size_t span(std::size_t len, int inc) {
  return len == 0 ? 1 : 1 + (len - 1) * static_cast<std::size_t>(std::abs(inc));
}

/// @brief 0-based buffer index of logical element k (KX/KY of the reference)
std::size_t at(std::size_t len, int inc, std::size_t k) {
  const std::ptrdiff_t first =
      inc > 0 ? 0 : static_cast<std::ptrdiff_t>(len - 1) * -static_cast<std::ptrdiff_t>(inc);
  return static_cast<std::size_t>(first + static_cast<std::ptrdiff_t>(k) * inc);
}

/// @brief ?LA_GEAMV, transcribed: one rounding per statement, no contraction
template<typename T>
void oracle(Trans trans, std::size_t m, std::size_t n, typename Elem<T>::real alpha,
            const std::vector<T> &a, std::size_t lda, const std::vector<T> &x, int incx,
            typename Elem<T>::real beta, std::vector<typename Elem<T>::real> &y, int incy) {
  using R = typename Elem<T>::real;
  if (m == 0 || n == 0 || (alpha == R(0) && beta == R(1))) {
    return;
  }
  const std::size_t lenx = trans == Trans::N ? n : m;
  const std::size_t leny = trans == Trans::N ? m : n;
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  for (std::size_t i = 0; i < leny; ++i) {
    R &yi = y[at(leny, incy, i)];
    bool symb_zero;
    if (beta == R(0)) {
      symb_zero = true;
      yi = R(0);
    } else if (yi == R(0)) {
      symb_zero = true;
    } else {
      symb_zero = false;
      const R ay = std::fabs(yi);
      yi = beta * ay;
    }
    if (alpha != R(0)) {
      for (std::size_t j = 0; j < lenx; ++j) {
        const T aij = trans == Trans::N ? a[i + j * lda] : a[j + i * lda];
        const R temp = Elem<T>::abs1(aij);
        const R xa = Elem<T>::abs1(x[at(lenx, incx, j)]);
        symb_zero = symb_zero && (xa == R(0) || temp == R(0));
        const R t1 = alpha * xa;
        const R t2 = t1 * temp;
        yi = yi + t2;
      }
    }
    if (!symb_zero) {
      const R nudge = std::copysign(safe1, yi);
      yi = yi + nudge;
    }
  }
}

/// @brief One fully specified call: inputs, host oracle, device run, bitwise compare
template<typename T>
struct Case {
  using R = typename Elem<T>::real;
  Trans trans;
  std::size_t m, n, lda;
  int incx, incy;
  R alpha, beta;
  std::vector<T> a, x;
  std::vector<R> y;

  std::size_t lenx() const { return trans == Trans::N ? n : m; }
  std::size_t leny() const { return trans == Trans::N ? m : n; }

  /// Runs both sides; returns the oracle y for the caller's extra assertions.
  std::vector<R> run(const std::string &ctx) const {
    auto handle = shared_device();
    std::vector<R> ref = y;
    oracle<T>(trans, m, n, alpha, a, lda, x, incx, beta, ref, incy);

    auto d_a = to_device(handle, a);
    auto d_x = to_device(handle, x);
    auto d_y = to_device(handle, y);
    const auto status = la_geamv<T>(handle->stream().get(), trans, m, n, alpha, d_a.data(), lda,
                                    d_x.data(), incx, beta, d_y.data(), incy);
    EXPECT_TRUE(status.ok()) << ctx << ": la_geamv returned " << status.name();
    wwr::wwrStreamSynchronize(handle->stream().get());
    const auto got = from_device(handle, d_y, y.size());
    for (std::size_t k = 0; k < y.size(); ++k) {
      EXPECT_TRUE(same_bits(got[k], ref[k]))
          << ctx << ": y buffer[" << k << "] got " << got[k] << " want " << ref[k];
    }
    return ref;
  }
};

const char *name(Trans t) {
  return t == Trans::N ? "N" : t == Trans::T ? "T" : "C";
}

/// @brief A with exact zeros: row 1, column 2, a diagonal band, and row/column 3
///        non-zero only where x is zero (k % 4 == 1) -- a symbolic zero that is
///        not a zero row. Padding rows are NaN.
template<typename T>
std::vector<T> make_a(std::size_t m, std::size_t n, std::size_t lda, std::mt19937 &gen) {
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<T> a(lda * n, Elem<T>::make(nan, nan));
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      const bool zero =
          i == 1 || j == 2 || (i == 3 && j % 4 != 1) || (j == 3 && i % 4 != 1) || (i + j) % 5 == 0;
      a[i + j * lda] = zero ? Elem<T>::make(0.0, 0.0) : Elem<T>::make(u(gen), u(gen));
    }
  }
  return a;
}

/// @brief x with exact zeros at k % 4 == 1 (one with a -0 component); gaps NaN
template<typename T>
std::vector<T> make_x(std::size_t len, int inc, std::mt19937 &gen) {
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<T> x(span(len, inc), Elem<T>::make(nan, nan));
  for (std::size_t k = 0; k < len; ++k) {
    x[at(len, inc, k)] =
        k % 4 == 1 ? Elem<T>::make(k % 8 == 1 ? -0.0 : 0.0, 0.0) : Elem<T>::make(u(gen), u(gen));
  }
  return x;
}

/// @brief y with exact zeros (+0 and -0) at k % 3 == 0; gaps hold sentinels
template<typename R>
std::vector<R> make_y(std::size_t len, int inc, std::mt19937 &gen) {
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::vector<R> y(span(len, inc), R(7.125));
  for (std::size_t k = 0; k < len; ++k) {
    y[at(len, inc, k)] = k % 3 == 0 ? R(k % 6 == 0 ? 0.0 : -0.0) : static_cast<R>(u(gen));
  }
  return y;
}

template<typename T>
void general() {
  using R = typename Elem<T>::real;
  struct Shape {
    std::size_t m, n, lda;
  };
  constexpr std::array<Shape, 6> kShapes{
      {{5, 7, 5}, {7, 5, 9}, {1, 1, 1}, {4, 1, 6}, {1, 4, 1}, {130, 260, 131}}};
  constexpr std::array<std::pair<int, int>, 3> kStrides{{{1, 1}, {2, -3}, {-1, 2}}};
  constexpr std::array<std::pair<double, double>, 7> kScalars{
      {{1, 0}, {1, 1}, {0.75, -1.25}, {0, 1}, {0, 0.5}, {-2, 0}, {1.5, 0}}};
  std::mt19937 gen(290);
  for (const Trans trans : {Trans::N, Trans::T, Trans::C}) {
    for (const auto &s : kShapes) {
      for (const auto &[incx, incy] : kStrides) {
        for (const auto &[alpha, beta] : kScalars) {
          Case<T> c{trans, s.m, s.n, s.lda, incx, incy, static_cast<R>(alpha), static_cast<R>(beta),
                    {},    {},  {}};
          c.a = make_a<T>(s.m, s.n, s.lda, gen);
          c.x = make_x<T>(c.lenx(), incx, gen);
          c.y = make_y<R>(c.leny(), incy, gen);
          const std::string ctx =
              std::format("trans={} m={} n={} lda={} incx={} incy={} "
                          "alpha={} beta={}",
                          name(trans), s.m, s.n, s.lda, incx, incy, alpha, beta);
          const auto ref = c.run(ctx);
          // With beta == 0 and m, n >= 4, the zero row and the x-masked row must
          // stay exactly 0 while the rest are nudged -- both paths exercised.
          if (alpha != 0 && beta == 0 && s.m >= 4 && s.n >= 4) {
            std::size_t zeros = 0;
            for (std::size_t k = 0; k < c.leny(); ++k) {
              zeros += ref[at(c.leny(), incy, k)] == R(0) ? 1 : 0;
            }
            EXPECT_GE(zeros, 2u) << ctx << ": no symbolic zeros exercised";
            EXPECT_LT(zeros, c.leny()) << ctx << ": no nudged entries exercised";
          }
        }
      }
    }
  }
}

/// @brief Every product underflows to 0 but none is symbolic: y must be safe1
template<typename T>
void underflow_is_nudged() {
  using R = typename Elem<T>::real;
  const double tiny = std::is_same_v<R, float> ? 1e-25 : 1e-200;
  const std::size_t m = 6, n = 5;
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  for (const Trans trans : {Trans::N, Trans::T, Trans::C}) {
    // alpha * |x| * |a| underflows; beta == 0.
    Case<T> c{trans, m, n, m, 1, 1, R(1), R(0), {}, {}, {}};
    c.a.assign(m * n, Elem<T>::make(tiny, -tiny));
    c.x.assign(c.lenx(), Elem<T>::make(tiny, 0.0));
    c.y.assign(c.leny(), R(1));
    auto ref = c.run(std::string("products underflow, trans=") + name(trans));
    for (const R v : ref) {
      EXPECT_TRUE(same_bits(v, safe1)) << "oracle gave " << v << ", want safe1 " << safe1;
    }
    // beta * |y| underflows and A is zero: still not symbolic.
    Case<T> d{trans, m, n, m, 1, 1, R(1), static_cast<R>(tiny), {}, {}, {}};
    d.a.assign(m * n, Elem<T>::make(0.0, 0.0));
    d.x.assign(d.lenx(), Elem<T>::make(1.0, 1.0));
    d.y.assign(d.leny(), static_cast<R>(-tiny));
    ref = d.run(std::string("beta*|y| underflows, trans=") + name(trans));
    for (const R v : ref) {
      EXPECT_TRUE(same_bits(v, safe1)) << "oracle gave " << v << ", want safe1 " << safe1;
    }
  }
}

/// @brief m == 0, n == 0 and alpha == 0 && beta == 1 leave y exactly as given
template<typename T>
void quick_returns() {
  using R = typename Elem<T>::real;
  std::mt19937 gen(7);
  const auto y = make_y<R>(5, 1, gen);
  for (const Trans trans : {Trans::N, Trans::T}) {
    for (const auto &[m, n, alpha, beta] :
         std::array<std::tuple<std::size_t, std::size_t, R, R>, 3>{
             {{0, 5, R(1), R(0)}, {5, 0, R(1), R(0)}, {5, 5, R(0), R(1)}}}) {
      Case<T> c{trans, m, n, 5, 1, 1, alpha, beta, {}, {}, y};
      c.a = make_a<T>(5, 5, 5, gen);
      c.x = make_x<T>(5, 1, gen);
      const auto ref = c.run(std::format("quick return m={} n={}", m, n));
      for (std::size_t k = 0; k < y.size(); ++k) {
        EXPECT_TRUE(same_bits(ref[k], y[k])) << "oracle touched y at " << k;
      }
    }
  }
}

template<typename T>
void invalid_arguments() {
  using R = typename Elem<T>::real;
  auto handle = shared_device();
  const std::vector<R> y(8, R(3));
  auto d_a = to_device(handle, std::vector<T>(16, Elem<T>::make(1.0, 0.0)));
  auto d_x = to_device(handle, std::vector<T>(8, Elem<T>::make(1.0, 0.0)));
  auto d_y = to_device(handle, y);
  const auto s = handle->stream().get();
  EXPECT_EQ(la_geamv<T>(s, Trans::N, 4, 3, R(1), d_a.data(), 3, d_x.data(), 1, R(0), d_y.data(), 1),
            wwr::wwrErrorInvalidValue)
      << "lda < m";
  EXPECT_EQ(la_geamv<T>(s, Trans::N, 0, 3, R(1), d_a.data(), 0, d_x.data(), 1, R(0), d_y.data(), 1),
            wwr::wwrErrorInvalidValue)
      << "lda < 1";
  EXPECT_EQ(la_geamv<T>(s, Trans::N, 4, 3, R(1), d_a.data(), 4, d_x.data(), 0, R(0), d_y.data(), 1),
            wwr::wwrErrorInvalidValue)
      << "incx == 0";
  EXPECT_EQ(la_geamv<T>(s, Trans::T, 4, 3, R(1), d_a.data(), 4, d_x.data(), 1, R(0), d_y.data(), 0),
            wwr::wwrErrorInvalidValue)
      << "incy == 0";
  EXPECT_EQ(la_geamv<T>(s, Trans::T, 4, 3, R(1), nullptr, 4, d_x.data(), 1, R(0), d_y.data(), 1),
            wwr::wwrErrorInvalidValue)
      << "null A";
  const auto got = from_device(handle, d_y, y.size());
  for (std::size_t k = 0; k < y.size(); ++k) {
    EXPECT_TRUE(same_bits(got[k], y[k])) << "a rejected call wrote y at " << k;
  }
}

template<typename T>
void all_cases() {
  general<T>();
  underflow_is_nudged<T>();
  quick_returns<T>();
  invalid_arguments<T>();
}

} // namespace

TEST(LaGeamvOracleTests, Float) {
  all_cases<float>();
}
TEST(LaGeamvOracleTests, Double) {
  all_cases<double>();
}
TEST(LaGeamvOracleTests, ComplexFloat) {
  all_cases<wwr::wwrFloatComplex>();
}
TEST(LaGeamvOracleTests, ComplexDouble) {
  all_cases<wwr::wwrDoubleComplex>();
}

} // namespace calaman
