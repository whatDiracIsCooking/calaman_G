// Oracle test for calaman.laset (built under src/laset/) -- set all or part of a
// column-major matrix to constants, BETA on the diagonal and ALPHA off it.
//
// The oracle is the reference LAPACKE_?laset, run on the host over the identical
// inputs. ?laset writes only constant values -- no arithmetic -- so the device
// and the reference must agree BIT FOR BIT, and the comparison is `==`, not a
// tolerance. Each case pre-fills BOTH buffers with a distinct per-element
// sentinel before the call, so the whole lda-by-n buffer is compared: this
// checks the set region AND that every untouched element (the strictly-other
// triangle, and the lda-m padding rows beyond the leading m) is preserved,
// exactly as the reference preserves it.
//
// Cases cover all three regions (U/L/A) over square, tall and wide shapes, a
// leading dimension strictly greater than m (padding rows that must stay
// untouched), a 1x1 matrix, and the m==0 / n==0 early-return (the wrapper
// enqueues nothing and the buffer is unchanged -- compared against the reference,
// which is likewise a no-op).
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the matrix on the device
// and runs the fill kernel, so `ctest -LE gpu` excludes it. It needs the
// reference LAPACK, so the suite is guarded on calaman::lapack_reference at
// configure time (CMakeLists.txt) -- a missing oracle is a missing tier, not a
// silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.laset;
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

// Reference ?laset, column-major, into the leading m-by-n submatrix of a.
lapack_int ref_laset(char uplo, lapack_int m, lapack_int n, float alpha, float beta, float *a,
                     lapack_int lda) {
  return LAPACKE_slaset(LAPACK_COL_MAJOR, uplo, m, n, alpha, beta, a, lda);
}
lapack_int ref_laset(char uplo, lapack_int m, lapack_int n, double alpha, double beta, double *a,
                     lapack_int lda) {
  return LAPACKE_dlaset(LAPACK_COL_MAJOR, uplo, m, n, alpha, beta, a, lda);
}

// Region::U/L/A -> the LAPACK uplo char the reference wants ('A' is the
// "otherwise" full-matrix case ?laset documents).
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

// A distinct, exactly-representable sentinel per buffer slot, so a mis-indexed
// write (or a stray write into the untouched region or the lda padding) lands on
// a value no correct run would produce. Small integers are exact in both float
// and double.
template<typename T>
T sentinel(std::size_t k) {
  return static_cast<T>(1000 + static_cast<long>(k));
}

// Run both paths over the SAME sentinel-filled lda-by-n buffer and assert the
// whole buffer agrees bit for bit. alpha/beta are passed as small integers, so
// every written value is exact in T and `==` is the right comparison.
template<typename T>
void check(Region region, int m, int n, int lda, T alpha, T beta, const char *ctx) {
  auto handle = std::make_shared<DeviceHandle>(0);
  const std::size_t size = static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);

  std::vector<T> ref(size);
  for (std::size_t k = 0; k < size; ++k) {
    ref[k] = sentinel<T>(k);
  }
  auto d_a = to_device(handle, ref); // device starts from the identical sentinel fill

  const lapack_int info =
      ref_laset(uplo_char(region), m, n, alpha, beta, ref.data(), lda);
  ASSERT_EQ(info, 0) << ctx << ": reference ?laset reported info=" << info;

  calaman::laset<T>(handle->stream().get(), region, static_cast<std::size_t>(m),
                    static_cast<std::size_t>(n), alpha, beta, d_a.data(),
                    static_cast<std::size_t>(lda));
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_a, size);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < lda; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * lda + i;
      EXPECT_EQ(got[k], ref[k]) << ctx << ": mismatch at (" << i << "," << j << ")";
    }
  }
}

template<typename T>
void all_regions_all_shapes() {
  // Square, all three regions, with a leading dimension > m so the padding rows
  // [m, lda) must be left untouched.
  check<T>(Region::A, 4, 4, 4, static_cast<T>(-1), static_cast<T>(7), "square/A");
  check<T>(Region::U, 5, 5, 7, static_cast<T>(2), static_cast<T>(-3), "square/U/pad");
  check<T>(Region::L, 5, 5, 7, static_cast<T>(2), static_cast<T>(-3), "square/L/pad");

  // Tall (m > n): the diagonal is min(m,n) = n long; U is a trapezoid, L fills
  // the tall lower block.
  check<T>(Region::A, 6, 3, 8, static_cast<T>(4), static_cast<T>(-5), "tall/A");
  check<T>(Region::U, 6, 3, 6, static_cast<T>(4), static_cast<T>(-5), "tall/U");
  check<T>(Region::L, 6, 3, 8, static_cast<T>(4), static_cast<T>(-5), "tall/L");

  // Wide (m < n): the diagonal is min(m,n) = m long; the trapezoid flips.
  check<T>(Region::A, 3, 6, 5, static_cast<T>(-2), static_cast<T>(9), "wide/A");
  check<T>(Region::U, 3, 6, 3, static_cast<T>(-2), static_cast<T>(9), "wide/U");
  check<T>(Region::L, 3, 6, 5, static_cast<T>(-2), static_cast<T>(9), "wide/L");

  // 1x1: the single element is the diagonal, so it gets beta regardless of region.
  check<T>(Region::A, 1, 1, 1, static_cast<T>(8), static_cast<T>(6), "one/A");
  check<T>(Region::U, 1, 1, 1, static_cast<T>(8), static_cast<T>(6), "one/U");
}

template<typename T>
void zero_dim_is_noop() {
  // m == 0 and n == 0: the wrapper enqueues nothing. Compared against the
  // reference, which is likewise a no-op, the whole sentinel buffer survives. A
  // non-zero lda/n keeps the buffer (and the comparison) non-empty.
  check<T>(Region::A, 0, 4, 5, static_cast<T>(1), static_cast<T>(2), "zero_m");
  check<T>(Region::A, 4, 0, 5, static_cast<T>(1), static_cast<T>(2), "zero_n");
}

} // namespace

TEST(LasetOracleTests, AllRegionsAllShapes) {
  all_regions_all_shapes<float>();
  all_regions_all_shapes<double>();
}

TEST(LasetOracleTests, ZeroDimIsNoop) {
  zero_dim_is_noop<float>();
  zero_dim_is_noop<double>();
}

} // namespace calaman
