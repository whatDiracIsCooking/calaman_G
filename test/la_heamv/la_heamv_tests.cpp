// Oracle test for calaman.la_heamv -- y := alpha * |A| * |x| + beta * |y| for
// a Hermitian A held in one triangle, with ?LA_HEAMV's symbolic-zero nudge.
//
// The oracle is test/la_syamv/'s: a same-precision host transcription of the
// loop over the DENSE full A, compared BIT FOR BIT (the image's liblapack does
// not export ?la_heamv_). The full A is Hermitian, A(j,i) == conj(A(i,j)), and
// its DIAGONAL CARRIES A NON-ZERO IMAGINARY PART: CLA_HEAMV reads every entry
// through CABS1, diagonal included, so that part counts -- where ?hemv would
// ignore it. That is stated in the interface and asserted here.
//
// The shared kernel really is shared: on every case la_syamv is run on the
// same stored triangle and must agree with la_heamv bit for bit.
//
// The unstored triangle, lda padding and x's stride gaps are NaN. Covered, for
// c/z and both Uplo: n = 1, 2, small and multi-block; padded lda; an entirely
// zero row; x with exact zeros; a symbolic-zero row that is not a zero row;
// the all-underflow case; both stride signs; the quick returns; the rejections.
//
// REQUIRES_GPU (see CMakeLists.txt): every case runs the kernel on the device.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.la_heamv;
import calaman.la_syamv;
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

/// @brief Per-type construction, conjugate and the host CABS1 (complex only)
template<typename T>
struct Elem;

template<>
struct Elem<wwr::wwrFloatComplex> {
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static wwr::wwrFloatComplex conj(wwr::wwrFloatComplex a) {
    return wwr::make_wwrFloatComplex(wwr::wwrCrealf(a), -wwr::wwrCimagf(a));
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
  static wwr::wwrDoubleComplex conj(wwr::wwrDoubleComplex a) {
    return wwr::make_wwrDoubleComplex(wwr::wwrCreal(a), -wwr::wwrCimag(a));
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

/// @brief ?LA_HEAMV over the dense full A (n-by-n, ld n): one rounding per statement
template<typename T>
void oracle(std::size_t n, typename Elem<T>::real alpha, const std::vector<T> &a,
            const std::vector<T> &x, int incx, typename Elem<T>::real beta,
            std::vector<typename Elem<T>::real> &y, int incy) {
  using R = typename Elem<T>::real;
  if (n == 0 || (alpha == R(0) && beta == R(1))) {
    return;
  }
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  for (std::size_t i = 0; i < n; ++i) {
    R &yi = y[at(n, incy, i)];
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
      for (std::size_t j = 0; j < n; ++j) {
        const R temp = Elem<T>::abs1(a[i + j * n]);
        const R xa = Elem<T>::abs1(x[at(n, incx, j)]);
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

/// @brief Pack the @p uplo triangle of the dense A into an lda-by-n buffer; all else NaN
template<typename T>
std::vector<T> pack(const std::vector<T> &a, std::size_t n, Uplo uplo, std::size_t lda) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<T> out(std::max<std::size_t>(1, lda * n), Elem<T>::make(nan, nan));
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      if (uplo == Uplo::U ? i <= j : i >= j) {
        out[i + j * lda] = a[i + j * n];
      }
    }
  }
  return out;
}

/// @brief One fully specified call: oracle, la_heamv and la_syamv, all compared bitwise
template<typename T>
struct Case {
  using R = typename Elem<T>::real;
  Uplo uplo;
  std::size_t n, lda;
  int incx, incy;
  R alpha, beta;
  std::vector<T> a, x; // a is the DENSE full Hermitian matrix, n-by-n, ld n
  std::vector<R> y;

  /// Runs all three; returns the oracle y for the caller's extra assertions.
  std::vector<R> run(const std::string &ctx) const {
    auto handle = shared_device();
    const auto s = handle->stream().get();
    std::vector<R> ref = y;
    oracle<T>(n, alpha, a, x, incx, beta, ref, incy);

    auto d_a = to_device(handle, pack(a, n, uplo, lda));
    auto d_x = to_device(handle, x);
    auto d_y = to_device(handle, y);
    auto d_ys = to_device(handle, y);
    const auto he =
        la_heamv<T>(s, uplo, n, alpha, d_a.data(), lda, d_x.data(), incx, beta, d_y.data(), incy);
    EXPECT_TRUE(he.ok()) << ctx << ": la_heamv returned " << he.name();
    const auto sy =
        la_syamv<T>(s, uplo, n, alpha, d_a.data(), lda, d_x.data(), incx, beta, d_ys.data(), incy);
    EXPECT_TRUE(sy.ok()) << ctx << ": la_syamv returned " << sy.name();
    wwr::wwrStreamSynchronize(s);
    const auto got = from_device(handle, d_y, y.size());
    const auto got_sy = from_device(handle, d_ys, y.size());
    for (std::size_t k = 0; k < y.size(); ++k) {
      EXPECT_TRUE(same_bits(got[k], ref[k]))
          << ctx << ": y buffer[" << k << "] got " << got[k] << " want " << ref[k];
      EXPECT_TRUE(same_bits(got_sy[k], got[k]))
          << ctx << ": la_syamv y buffer[" << k << "] " << got_sy[k] << " != la_heamv " << got[k];
    }
    return ref;
  }
};

const char *name(Uplo u) {
  return u == Uplo::U ? "U" : "L";
}

/// @brief Dense Hermitian A, A(j,i) = conj(A(i,j)), whose diagonal has NON-ZERO
///        imaginary parts, with exact zeros: row/column 1 entirely, every
///        (i + j) % 5 == 0, and row/column 3 non-zero only where x is zero
///        (k % 4 == 1) -- a symbolic zero that is not a zero row.
template<typename T>
std::vector<T> make_a(std::size_t n, std::mt19937 &gen) {
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::uniform_real_distribution<double> im(0.25, 2.0);
  std::vector<T> a(n * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i <= j; ++i) {
      const bool zero =
          i == 1 || j == 1 || (i == 3 && j % 4 != 1) || (j == 3 && i % 4 != 1) || (i + j) % 5 == 0;
      const T v = zero ? Elem<T>::make(0.0, 0.0) : Elem<T>::make(u(gen), i == j ? im(gen) : u(gen));
      a[i + j * n] = v;
      a[j + i * n] = Elem<T>::conj(v);
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
  constexpr std::array<std::pair<std::size_t, std::size_t>, 8> kShapes{
      {{1, 0}, {1, 2}, {2, 0}, {5, 0}, {8, 3}, {13, 1}, {130, 0}, {300, 7}}}; // n, lda - n
  constexpr std::array<std::pair<int, int>, 3> kStrides{{{1, 1}, {2, -3}, {-1, 2}}};
  constexpr std::array<std::pair<double, double>, 7> kScalars{
      {{1, 0}, {1, 1}, {0.75, -1.25}, {0, 1}, {0, 0.5}, {-2, 0}, {1.5, 0}}};
  std::mt19937 gen(2920);
  for (const Uplo uplo : {Uplo::U, Uplo::L}) {
    for (const auto &[n, pad] : kShapes) {
      for (const auto &[incx, incy] : kStrides) {
        for (const auto &[alpha, beta] : kScalars) {
          Case<T> c{uplo, n,  n + pad, incx, incy, static_cast<R>(alpha), static_cast<R>(beta),
                    {},   {}, {}};
          c.a = make_a<T>(n, gen);
          c.x = make_x<T>(n, incx, gen);
          c.y = make_y<R>(n, incy, gen);
          const std::string ctx =
              std::format("uplo={} n={} lda={} incx={} incy={} alpha={} beta={}", name(uplo), n,
                          c.lda, incx, incy, alpha, beta);
          const auto ref = c.run(ctx);
          if (alpha != 0 && beta == 0 && n >= 4) {
            EXPECT_TRUE(same_bits(ref[at(n, incy, 1)], R(0))) << ctx << ": zero row not 0";
            EXPECT_TRUE(same_bits(ref[at(n, incy, 3)], R(0))) << ctx << ": masked row not 0";
          }
        }
      }
    }
  }
}

/// @brief The diagonal's imaginary part counts: A = i * I gives y = alpha * |x| (+ nudge)
template<typename T>
void diagonal_imaginary_part_is_read() {
  using R = typename Elem<T>::real;
  const std::size_t n = 4;
  for (const Uplo uplo : {Uplo::U, Uplo::L}) {
    Case<T> c{uplo, n, n, 1, 1, R(1), R(0), {}, {}, {}};
    c.a.assign(n * n, Elem<T>::make(0.0, 0.0));
    for (std::size_t k = 0; k < n; ++k) {
      c.a[k + k * n] = Elem<T>::make(0.0, 1.0); // purely imaginary diagonal
    }
    c.x.assign(n, Elem<T>::make(2.0, 0.0));
    c.y.assign(n, R(0));
    const auto ref = c.run(std::string("imaginary diagonal, uplo=") + name(uplo));
    const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
    for (const R v : ref) {
      EXPECT_EQ(v, R(2) + safe1) << "the diagonal's imaginary part was dropped";
    }
  }
}

/// @brief Every product underflows to 0 but none is symbolic: y must be safe1
template<typename T>
void underflow_is_nudged() {
  using R = typename Elem<T>::real;
  const double tiny = std::is_same_v<R, float> ? 1e-25 : 1e-200;
  const std::size_t n = 6, lda = 7;
  const R safe1 = static_cast<R>(n + 1) * std::numeric_limits<R>::min();
  for (const Uplo uplo : {Uplo::U, Uplo::L}) {
    Case<T> c{uplo, n, lda, 1, 1, R(1), R(0), {}, {}, {}};
    c.a.assign(n * n, Elem<T>::make(tiny, 0.0));
    c.x.assign(n, Elem<T>::make(tiny, 0.0));
    c.y.assign(n, R(1));
    const auto ref = c.run(std::string("products underflow, uplo=") + name(uplo));
    for (const R v : ref) {
      EXPECT_TRUE(same_bits(v, safe1)) << "oracle gave " << v << ", want safe1 " << safe1;
    }
  }
}

/// @brief n == 0 and alpha == 0 && beta == 1 leave y exactly as given
template<typename T>
void quick_returns() {
  using R = typename Elem<T>::real;
  std::mt19937 gen(7);
  const auto y = make_y<R>(5, 1, gen);
  for (const Uplo uplo : {Uplo::U, Uplo::L}) {
    for (const auto &[n, alpha, beta] :
         std::array<std::tuple<std::size_t, R, R>, 2>{{{0, R(1), R(0)}, {5, R(0), R(1)}}}) {
      Case<T> c{uplo, n, std::max<std::size_t>(1, n), 1, 1, alpha, beta, {}, {}, y};
      c.a = make_a<T>(n, gen);
      c.x = make_x<T>(5, 1, gen);
      const auto ref = c.run(std::format("quick return n={}", n));
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
  auto d_a = to_device(handle, std::vector<T>(32, Elem<T>::make(1.0, 0.0)));
  auto d_x = to_device(handle, std::vector<T>(8, Elem<T>::make(1.0, 0.0)));
  auto d_y = to_device(handle, y);
  const auto s = handle->stream().get();
  const auto call = [&](Uplo uplo, std::size_t n, const T *a, std::size_t lda, int incx, int incy) {
    return la_heamv<T>(s, uplo, n, R(1), a, lda, d_x.data(), incx, R(0), d_y.data(), incy);
  };
  const T *a = d_a.data();
  EXPECT_EQ(call(Uplo::U, 4, a, 3, 1, 1), wwr::wwrErrorInvalidValue) << "lda < n";
  EXPECT_EQ(call(Uplo::L, 0, a, 0, 1, 1), wwr::wwrErrorInvalidValue) << "lda < 1, n == 0";
  EXPECT_EQ(call(Uplo::U, 4, a, 4, 0, 1), wwr::wwrErrorInvalidValue) << "incx == 0";
  EXPECT_EQ(call(Uplo::L, 4, a, 4, 1, 0), wwr::wwrErrorInvalidValue) << "incy == 0";
  EXPECT_EQ(call(Uplo::U, 4, nullptr, 4, 1, 1), wwr::wwrErrorInvalidValue) << "null A";
  EXPECT_EQ(la_heamv<T>(s, Uplo::U, 4, R(0), a, 2, d_x.data(), 1, R(1), d_y.data(), 1),
            wwr::wwrErrorInvalidValue)
      << "bad lda on a quick-return call";
  const auto got = from_device(handle, d_y, y.size());
  for (std::size_t k = 0; k < y.size(); ++k) {
    EXPECT_TRUE(same_bits(got[k], y[k])) << "a rejected call wrote y at " << k;
  }
}

template<typename T>
void all_cases() {
  general<T>();
  diagonal_imaginary_part_is_read<T>();
  underflow_is_nudged<T>();
  quick_returns<T>();
  invalid_arguments<T>();
}

} // namespace

TEST(LaHeamvOracleTests, ComplexFloat) {
  all_cases<wwr::wwrFloatComplex>();
}
TEST(LaHeamvOracleTests, ComplexDouble) {
  all_cases<wwr::wwrDoubleComplex>();
}

} // namespace calaman
