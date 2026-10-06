// Oracle test for calaman.lacp2 -- copy all or part of a real matrix into a
// complex one, imaginary part 0. The oracle is LAPACKE_clacp2 / zlacp2 over the
// identical inputs. ?lacp2 only widens and stores, so device and reference
// agree BIT FOR BIT and every component is compared by its bit pattern.
//
// B starts as a distinct per-slot complex sentinel in both paths and the whole
// ldb-by-n buffer is compared, so the copied region, the untouched opposite
// triangle and the ldb - m padding rows are all checked. Cases: every Region
// over square, tall, wide and 1x1 shapes with lda > m and ldb > m (lda != ldb),
// plus the m == 0 / n == 0 no-op.
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lacp2;
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

template<typename C>
struct elem;

template<>
struct elem<wwr::wwrFloatComplex> {
  using Real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static std::array<std::uint32_t, 2> bits(wwr::wwrFloatComplex z) {
    return {std::bit_cast<std::uint32_t>(wwr::wwrCrealf(z)),
            std::bit_cast<std::uint32_t>(wwr::wwrCimagf(z))};
  }
  static lapack_int ref(char uplo, int m, int n, const float *a, int lda,
                        wwr::wwrFloatComplex *b, int ldb) {
    return LAPACKE_clacp2(LAPACK_COL_MAJOR, uplo, m, n, a, lda,
                          reinterpret_cast<lapack_complex_float *>(b), ldb);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using Real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static std::array<std::uint64_t, 2> bits(wwr::wwrDoubleComplex z) {
    return {std::bit_cast<std::uint64_t>(wwr::wwrCreal(z)),
            std::bit_cast<std::uint64_t>(wwr::wwrCimag(z))};
  }
  static lapack_int ref(char uplo, int m, int n, const double *a, int lda,
                        wwr::wwrDoubleComplex *b, int ldb) {
    return LAPACKE_zlacp2(LAPACK_COL_MAJOR, uplo, m, n, a, lda,
                          reinterpret_cast<lapack_complex_double *>(b), ldb);
  }
};

char uplo_char(Region region) {
  switch (region) {
  case Region::U:
    return 'U';
  case Region::L:
    return 'L';
  default:
    return 'A';
  }
}

template<typename C>
void check(Region region, int m, int n, int lda, int ldb, const char *ctx) {
  using R = typename elem<C>::Real;
  auto handle = shared_device();
  const std::size_t a_size = static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
  const std::size_t b_size = static_cast<std::size_t>(ldb) * static_cast<std::size_t>(n);

  // Exactly representable, distinct per slot; negative and fractional values.
  std::vector<R> a(a_size);
  for (std::size_t k = 0; k < a_size; ++k) {
    a[k] = static_cast<R>(0.25 * static_cast<double>(k) - 7.5);
  }
  std::vector<C> ref(b_size);
  for (std::size_t k = 0; k < b_size; ++k) {
    ref[k] = elem<C>::make(1000.0 + static_cast<double>(k), -1.0 - static_cast<double>(k));
  }
  auto d_a = to_device(handle, a);
  auto d_b = to_device(handle, ref);

  const lapack_int info = elem<C>::ref(uplo_char(region), m, n, a.data(), lda, ref.data(), ldb);
  ASSERT_EQ(info, 0) << ctx << ": reference ?lacp2 reported info=" << info;

  const auto status = calaman::lacp2<C>(
      handle->stream().get(), region, static_cast<std::size_t>(m), static_cast<std::size_t>(n),
      d_a.data(), static_cast<std::size_t>(lda), d_b.data(), static_cast<std::size_t>(ldb));
  EXPECT_TRUE(status.ok()) << ctx << ": lacp2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_b, b_size);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < ldb; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * ldb + i;
      EXPECT_EQ(elem<C>::bits(got[k]), elem<C>::bits(ref[k]))
          << ctx << ": mismatch at (" << i << "," << j << ")";
    }
  }
}

template<typename C>
void all_regions_all_shapes() {
  for (const Region r : {Region::A, Region::U, Region::L}) {
    check<C>(r, 5, 5, 7, 6, "square, lda > ldb > m");
    check<C>(r, 6, 3, 8, 9, "tall, ldb > lda > m");
    check<C>(r, 3, 6, 4, 5, "wide");
    check<C>(r, 1, 1, 2, 3, "1x1");
    check<C>(r, 130, 4, 131, 133, "taller than one block");
  }
}

template<typename C>
void zero_dim_is_noop() {
  check<C>(Region::A, 0, 4, 5, 3, "zero_m");
  check<C>(Region::A, 4, 0, 5, 6, "zero_n");
}

} // namespace

TEST(Lacp2OracleTests, AllRegionsAllShapes) {
  all_regions_all_shapes<wwr::wwrFloatComplex>();
  all_regions_all_shapes<wwr::wwrDoubleComplex>();
}

TEST(Lacp2OracleTests, ZeroDimIsNoop) {
  zero_dim_is_noop<wwr::wwrFloatComplex>();
  zero_dim_is_noop<wwr::wwrDoubleComplex>();
}

} // namespace calaman
