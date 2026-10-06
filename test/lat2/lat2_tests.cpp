// Oracle test for calaman.lat2 -- triangular matrix, double to single. LAPACKE
// has no ?lat2? binding, so the oracle is the Fortran dlat2s_ / zlat2c_ via
// calaman::lapack_reference. A precision conversion rounds the same way on host
// and device, so SA must agree BIT FOR BIT over the whole sentinel-filled
// ldsa-by-n buffer: the converted triangle, the untouched opposite triangle and
// the ldsa - n padding rows. Both UPLO values, lda != ldsa throughout.
//
// Overflow: an entry just above FLT_MAX in the triangle (for complex, the real
// and the imaginary part separately; also on the diagonal) gives INFO = 1 as the
// reference does; the same entry only in the opposite triangle gives INFO = 0.
// SA is unspecified on INFO = 1, so those cases compare INFO only.
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lat2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

// The reference ?lat2?, Fortran-mangled; everything by reference, plus the
// hidden length of the CHARACTER argument.
extern "C" {
void dlat2s_(const char *uplo, const int *n, const double *a, const int *lda, float *sa,
             const int *ldsa, int *info, std::size_t uplo_len);
void zlat2c_(const char *uplo, const int *n, const void *a, const int *lda, void *sa,
             const int *ldsa, int *info, std::size_t uplo_len);
}

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

// Element traits: build from (re, im) -- im ignored for a real type -- the bit
// pattern of every component, and the reference call. The reference ?lat2?
// assigns INFO only on overflow (its callers, ?sposv/?cposv, pre-zero it), so
// the call starts from 0 as they do; calaman.lat2 writes the 0 itself.
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
  static int ref(char uplo, int n, const double *a, int lda, float *sa, int ldsa) {
    int info = 0;
    dlat2s_(&uplo, &n, a, &lda, sa, &ldsa, &info, 1);
    return info;
  }
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
  static int ref(char uplo, int n, const wwr::wwrDoubleComplex *a, int lda,
                 wwr::wwrFloatComplex *sa, int ldsa) {
    int info = 0;
    zlat2c_(&uplo, &n, a, &lda, sa, &ldsa, &info, 1);
    return info;
  }
};

char uplo_char(Uplo uplo) {
  return uplo == Uplo::U ? 'U' : 'L';
}

// Values that are NOT exact in single precision (so narrowing really rounds),
// spanning signs and magnitudes, distinct per slot.
double value(std::size_t k, int part) {
  const double base = (part == 0 ? 1.0 : -1.0) * (0.1 + 0.37 * static_cast<double>(k));
  return k % 3 == 0 ? base * 1e30 : (k % 3 == 1 ? base * 1e-30 : base);
}

template<typename From>
std::vector<From> filled(int lda, int n) {
  std::vector<From> a(static_cast<std::size_t>(lda) * static_cast<std::size_t>(n));
  for (std::size_t k = 0; k < a.size(); ++k) {
    a[k] = elem<From>::make(value(k, 0), value(k, 1));
  }
  return a;
}

// Run device and reference over the same A and sentinel SA; compare SA bitwise
// when the reference reports INFO = 0. Returns {device INFO, reference INFO}.
template<typename From, typename To>
std::pair<int, int> run(const std::vector<From> &a, Uplo uplo, int n, int lda, int ldsa,
                        const std::string &ctx) {
  auto handle = shared_device();
  const std::size_t sa_size = static_cast<std::size_t>(ldsa) * static_cast<std::size_t>(n);
  std::vector<To> ref(sa_size);
  for (std::size_t k = 0; k < sa_size; ++k) {
    ref[k] = elem<To>::make(-1000.0 - static_cast<double>(k), 500.0 + static_cast<double>(k));
  }
  auto d_a = to_device(handle, a);
  auto d_sa = to_device(handle, ref);
  auto d_info = to_device(handle, std::vector<int>{-7});

  const int ref_info = elem<From>::ref(uplo_char(uplo), n, a.data(), lda, ref.data(), ldsa);

  const auto status = calaman::lat2<From, To>(
      handle->stream().get(), uplo, static_cast<std::size_t>(n), d_a.data(),
      static_cast<std::size_t>(lda), d_sa.data(), static_cast<std::size_t>(ldsa), d_info.data());
  EXPECT_TRUE(status.ok()) << ctx << ": lat2 returned status=" << status.name();
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
  return {info, ref_info};
}

template<typename From, typename To>
void matches_reference() {
  struct Shape {
    int n, lda, ldsa;
  };
  for (const Uplo uplo : {Uplo::U, Uplo::L}) {
    for (const Shape s : {Shape{5, 7, 6}, Shape{6, 6, 9}, Shape{1, 2, 3}, Shape{130, 131, 135},
                          Shape{0, 3, 2}}) {
      const auto ctx =
          std::format("uplo={} n={} lda={} ldsa={}", uplo_char(uplo), s.n, s.lda, s.ldsa);
      const auto [info, ref_info] =
          run<From, To>(filled<From>(s.lda, s.n), uplo, s.n, s.lda, s.ldsa, ctx);
      EXPECT_EQ(ref_info, 0) << ctx;
      EXPECT_EQ(info, 0) << ctx;
    }
  }
}

// Plant `bad` in the real (part 0) or imaginary (part 1) component of entry
// (i, j) and compare INFO with the reference.
template<typename From, typename To>
void expect_info(Uplo uplo, double bad, int part, int i, int j, int expected, const char *what) {
  const int n = 6, lda = 8, ldsa = 7;
  auto a = filled<From>(lda, n);
  const std::size_t k = static_cast<std::size_t>(j) * lda + i;
  a[k] = elem<From>::make(part == 0 ? bad : value(k, 0), part == 1 ? bad : value(k, 1));
  const auto ctx = std::format("uplo={} {} at ({},{}) part {}", uplo_char(uplo), what, i, j, part);
  const auto [info, ref_info] = run<From, To>(a, uplo, n, lda, ldsa, ctx);
  EXPECT_EQ(ref_info, expected) << ctx << ": reference";
  EXPECT_EQ(info, ref_info) << ctx;
}

template<typename From, typename To>
void overflow_info(const std::vector<int> &parts) {
  const double flt_max = std::numeric_limits<float>::max();
  const double above = std::nextafter(flt_max, std::numeric_limits<double>::infinity());
  for (const int part : parts) {
    // (1, 4) is strictly upper, (4, 1) strictly lower, (3, 3) on the diagonal.
    expect_info<From, To>(Uplo::U, above, part, 1, 4, 1, "above FLT_MAX in U");
    expect_info<From, To>(Uplo::U, -above, part, 3, 3, 1, "below -FLT_MAX on the diagonal");
    expect_info<From, To>(Uplo::U, above, part, 4, 1, 0, "above FLT_MAX only in L");
    expect_info<From, To>(Uplo::L, above, part, 4, 1, 1, "above FLT_MAX in L");
    expect_info<From, To>(Uplo::L, above, part, 3, 3, 1, "above FLT_MAX on the diagonal");
    expect_info<From, To>(Uplo::L, -above, part, 1, 4, 0, "below -FLT_MAX only in U");
    expect_info<From, To>(Uplo::L, flt_max, part, 4, 1, 0, "exactly FLT_MAX in L");
  }
}

} // namespace

TEST(Lat2OracleTests, DoubleToSingle) {
  matches_reference<double, float>();
}

TEST(Lat2OracleTests, ComplexDoubleToSingle) {
  matches_reference<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>();
}

TEST(Lat2OracleTests, OverflowInfoReal) {
  overflow_info<double, float>({0});
}

TEST(Lat2OracleTests, OverflowInfoComplexRealAndImagParts) {
  overflow_info<wwr::wwrDoubleComplex, wwr::wwrFloatComplex>({0, 1});
}

} // namespace calaman
