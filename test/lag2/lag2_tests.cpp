// Oracle test for calaman.lag2 -- general matrix, double <-> single, and
// through it the lag2_convert header. The oracle is LAPACKE_dlag2s / slag2d /
// zlag2c / clag2z over the identical inputs. A precision conversion rounds the
// same way on host and device, so SA must agree BIT FOR BIT, compared over the
// whole sentinel-filled ldsa-by-n buffer (so the ldsa - m padding rows must be
// untouched too), and lda != ldsa throughout.
//
// Overflow: one entry just above FLT_MAX (and, for complex, separately in the
// real and the imaginary part, both signs) must give INFO = 1 as the reference
// does; exactly FLT_MAX, and an overflow sitting in A's lda padding rows outside
// the m-by-n matrix, give INFO = 0. SA is unspecified on INFO = 1, so those
// cases compare INFO only. Plus the m == 0 / n == 0 case (INFO = 0, SA intact).
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lag2;
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

// Element traits: build from (re, im) -- im ignored for a real type -- and the
// bit pattern of every component.
template<typename T>
struct elem;

template<>
struct elem<float> {
  static float make(double re, double) { return static_cast<float>(re); }
  static std::array<std::uint64_t, 2> bits(float v) { return {std::bit_cast<std::uint32_t>(v), 0}; }
};
template<>
struct elem<double> {
  static double make(double re, double) { return re; }
  static std::array<std::uint64_t, 2> bits(double v) { return {std::bit_cast<std::uint64_t>(v), 0}; }
};
template<>
struct elem<wwr::wwrFloatComplex> {
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static std::array<std::uint64_t, 2> bits(wwr::wwrFloatComplex z) {
    return {std::bit_cast<std::uint32_t>(wwr::wwrCrealf(z)),
            std::bit_cast<std::uint32_t>(wwr::wwrCimagf(z))};
  }
};
template<>
struct elem<wwr::wwrDoubleComplex> {
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static std::array<std::uint64_t, 2> bits(wwr::wwrDoubleComplex z) {
    return {std::bit_cast<std::uint64_t>(wwr::wwrCreal(z)),
            std::bit_cast<std::uint64_t>(wwr::wwrCimag(z))};
  }
};

lapack_int ref_lag2(int m, int n, const double *a, int lda, float *sa, int ldsa) {
  return LAPACKE_dlag2s(LAPACK_COL_MAJOR, m, n, a, lda, sa, ldsa);
}
lapack_int ref_lag2(int m, int n, const float *a, int lda, double *sa, int ldsa) {
  return LAPACKE_slag2d(LAPACK_COL_MAJOR, m, n, a, lda, sa, ldsa);
}
lapack_int ref_lag2(int m, int n, const wwr::wwrDoubleComplex *a, int lda,
                    wwr::wwrFloatComplex *sa, int ldsa) {
  return LAPACKE_zlag2c(LAPACK_COL_MAJOR, m, n,
                        reinterpret_cast<const lapack_complex_double *>(a), lda,
                        reinterpret_cast<lapack_complex_float *>(sa), ldsa);
}
lapack_int ref_lag2(int m, int n, const wwr::wwrFloatComplex *a, int lda,
                    wwr::wwrDoubleComplex *sa, int ldsa) {
  return LAPACKE_clag2z(LAPACK_COL_MAJOR, m, n,
                        reinterpret_cast<const lapack_complex_float *>(a), lda,
                        reinterpret_cast<lapack_complex_double *>(sa), ldsa);
}

// Values that are NOT exact in single precision (so narrowing really rounds),
// spanning signs and magnitudes, distinct per slot.
double value(std::size_t k, int part) {
  const double base = (part == 0 ? 1.0 : -1.0) * (0.1 + 0.37 * static_cast<double>(k));
  return k % 3 == 0 ? base * 1e30 : (k % 3 == 1 ? base * 1e-30 : base);
}

// Run device and reference over the same A and sentinel SA. Compares SA bitwise
// when the reference reports INFO = 0; returns both INFOs to the caller.
template<typename From, typename To>
std::pair<int, int> run(std::vector<From> a, int m, int n, int lda, int ldsa, const char *ctx) {
  auto handle = shared_device();
  const std::size_t sa_size = static_cast<std::size_t>(ldsa) * static_cast<std::size_t>(n);
  std::vector<To> ref(sa_size);
  for (std::size_t k = 0; k < sa_size; ++k) {
    ref[k] = elem<To>::make(-1000.0 - static_cast<double>(k), 500.0 + static_cast<double>(k));
  }
  auto d_a = to_device(handle, a);
  auto d_sa = to_device(handle, ref);
  auto d_info = to_device(handle, std::vector<int>{-7});

  const lapack_int ref_info = ref_lag2(m, n, a.data(), lda, ref.data(), ldsa);

  const auto status = calaman::lag2<From, To>(
      handle->stream().get(), static_cast<std::size_t>(m), static_cast<std::size_t>(n),
      d_a.data(), static_cast<std::size_t>(lda), d_sa.data(), static_cast<std::size_t>(ldsa),
      d_info.data());
  EXPECT_TRUE(status.ok()) << ctx << ": lag2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const int info = from_device(handle, d_info, 1)[0];
  if (ref_info == 0) {
    const auto got = from_device(handle, d_sa, sa_size);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < ldsa; ++i) {
        const std::size_t k = static_cast<std::size_t>(j) * ldsa + i;
        EXPECT_EQ(elem<To>::bits(got[k]), elem<To>::bits(ref[k]))
            << ctx << ": SA mismatch at (" << i << "," << j << ")";
      }
    }
  }
  return {info, static_cast<int>(ref_info)};
}

template<typename From>
std::vector<From> filled(int lda, int n) {
  std::vector<From> a(static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
  for (std::size_t k = 0; k < a.size(); ++k) {
    a[k] = elem<From>::make(value(k, 0), value(k, 1));
  }
  return a;
}

template<typename From, typename To>
void matches_reference() {
  struct Shape {
    int m, n, lda, ldsa;
  };
  for (const Shape s : {Shape{5, 4, 7, 6}, Shape{3, 6, 4, 9}, Shape{1, 1, 2, 3},
                        Shape{130, 3, 131, 135}, Shape{0, 4, 5, 3}, Shape{4, 0, 5, 6}}) {
    const auto ctx = std::format("m={} n={} lda={} ldsa={}", s.m, s.n, s.lda, s.ldsa);
    const auto [info, ref_info] =
        run<From, To>(filled<From>(s.lda, s.n), s.m, s.n, s.lda, s.ldsa, ctx.c_str());
    EXPECT_EQ(ref_info, 0) << ctx;
    EXPECT_EQ(info, 0) << ctx;
  }
}

// Narrowing overflow: plant `bad` in the real (part 0) or imaginary (part 1)
// component of entry (i, j) and compare INFO with the reference.
template<typename From, typename To>
void expect_info(double bad, int part, int i, int j, int expected, const char *what) {
  const int m = 6, n = 5, lda = 8, ldsa = 7;
  auto a = filled<From>(lda, n);
  const std::size_t k = static_cast<std::size_t>(j) * lda + i;
  const double re = part == 0 ? bad : value(k, 0);
  const double im = part == 1 ? bad : value(k, 1);
  a[k] = elem<From>::make(re, im);
  const auto ctx = std::format("{} at ({},{}) part {}", what, i, j, part);
  const auto [info, ref_info] = run<From, To>(a, m, n, lda, ldsa, ctx.c_str());
  EXPECT_EQ(ref_info, expected) << ctx << ": reference";
  EXPECT_EQ(info, ref_info) << ctx;
}

template<typename From, typename To>
void overflow_info(const std::vector<int> &parts) {
  const double flt_max = std::numeric_limits<float>::max();
  const double above = std::nextafter(flt_max, std::numeric_limits<double>::infinity());
  for (const int part : parts) {
    expect_info<From, To>(above, part, 2, 3, 1, "just above FLT_MAX");
    expect_info<From, To>(-above, part, 5, 0, 1, "just below -FLT_MAX");
    expect_info<From, To>(std::numeric_limits<double>::infinity(), part, 0, 4, 1, "+inf");
    expect_info<From, To>(flt_max, part, 2, 3, 0, "exactly FLT_MAX");
    expect_info<From, To>(-flt_max, part, 1, 1, 0, "exactly -FLT_MAX");
    // Row 7 lies in A's lda padding (m = 6): outside the matrix, not checked.
    expect_info<From, To>(above, part, 7, 2, 0, "overflow in the lda padding");
  }
}

} // namespace

TEST(Lag2OracleTests, DoubleToSingle) {
  matches_reference<double, float>();
}

TEST(Lag2OracleTests, SingleToDouble) {
  matches_reference<float, double>();
}

TEST(Lag2OracleTests, ComplexDoubleToSingle) {
  matches_reference<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>();
}

TEST(Lag2OracleTests, ComplexSingleToDouble) {
  matches_reference<wwr::wwrFloatComplex, wwr::wwrDoubleComplex>();
}

TEST(Lag2OracleTests, OverflowInfoReal) {
  overflow_info<double, float>({0});
}

TEST(Lag2OracleTests, OverflowInfoComplexRealAndImagParts) {
  overflow_info<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>({0, 1});
}

} // namespace calaman
