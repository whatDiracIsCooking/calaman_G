// Oracle test for calaman.lascl2 (built under src/lascl2/) -- diagonal
// row-scaling of a column-major matrix, X(i,j) <-- d(i) * X(i,j).
//
// There is no LAPACKE (or CBLAS) binding for ?lascl2 -- it is an auxiliary
// routine the reference destines for a future BLAS_?ge_diag_scale -- so the
// oracle is a same-precision host transcription of its double loop, run over the
// identical inputs. ?lascl2 does one multiply per element with no reordered
// summation, so with small exact-integer inputs every product is exact in T and
// the device and host agree BIT FOR BIT; the comparison is `==`, not a
// tolerance. Each case pre-fills BOTH buffers with a distinct per-element
// sentinel before the call, so the whole ldx-by-n buffer is compared: this
// checks the scaled region AND that every untouched element (the ldx-m padding
// rows beyond the leading m) is preserved, exactly as the host reference leaves
// them.
//
// Cases cover square, tall and wide shapes, a leading dimension strictly greater
// than m (padding rows that must stay untouched), a 1x1 matrix, a zero scale
// factor (the element is zeroed, exactly), and the m==0 / n==0 early-return (the
// wrapper enqueues nothing and the buffer is unchanged).
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the matrix on the device
// and runs the scaling kernel, so `ctest -LE gpu` excludes it. Unlike the laset
// suite it needs no reference LAPACK -- the oracle is the host loop below -- so
// it is not guarded on calaman::lapack_reference.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lascl2;
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

// A distinct, exactly-representable sentinel per matrix slot, so a mis-indexed
// write (or a stray write into the ldx padding) lands on a value no correct run
// would produce. Small integers are exact in both float and double.
template<typename T>
T sentinel(std::size_t k) {
  return static_cast<T>(1000 + static_cast<long>(k));
}

// The per-row scale factor d(i): small integers cycling through a zero (so one
// row is zeroed exactly) and both signs. Kept small so d(i) * sentinel stays an
// exact integer in float (< 2^24) as well as double.
template<typename T>
T scale(std::size_t i) {
  return static_cast<T>(static_cast<long>(i % 5) - 2); // -2, -1, 0, 1, 2, -2, ...
}

// Stage the SAME sentinel-filled ldx-by-n buffer and the SAME length-m scale
// vector on host and device, run both paths, and assert the whole buffer agrees
// bit for bit -- scaled region and untouched padding alike.
template<typename T>
void check(int m, int n, int ldx, const char *ctx) {
  auto handle = shared_device();
  const std::size_t size = static_cast<std::size_t>(ldx) * static_cast<std::size_t>(n);

  std::vector<T> ref(size);
  for (std::size_t k = 0; k < size; ++k) {
    ref[k] = sentinel<T>(k);
  }
  std::vector<T> d(static_cast<std::size_t>(m == 0 ? 1 : m));
  for (int i = 0; i < m; ++i) {
    d[static_cast<std::size_t>(i)] = scale<T>(static_cast<std::size_t>(i));
  }

  auto d_x = to_device(handle, ref); // device starts from the identical sentinel fill
  auto d_d = to_device(handle, d);

  // Host reference: the literal DLASCL2 loop over the leading m-by-n submatrix,
  // leaving the ldx-m padding rows untouched.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * ldx + i;
      ref[k] *= d[static_cast<std::size_t>(i)];
    }
  }

  const auto status =
      calaman::lascl2<T>(handle->stream().get(), static_cast<std::size_t>(m),
                         static_cast<std::size_t>(n), d_d.data(), d_x.data(),
                         static_cast<std::size_t>(ldx));
  EXPECT_TRUE(status.ok()) << ctx << ": lascl2 returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_x, size);

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < ldx; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * ldx + i;
      EXPECT_EQ(got[k], ref[k]) << ctx << ": mismatch at (" << i << "," << j << ")";
    }
  }
}

template<typename T>
void all_shapes() {
  // Square, then square with a leading dimension > m so the padding rows
  // [m, ldx) must be left untouched.
  check<T>(4, 4, 4, "square");
  check<T>(5, 5, 7, "square/pad");

  // Tall (m > n) and wide (m < n), each with and without padding.
  check<T>(6, 3, 6, "tall");
  check<T>(6, 3, 8, "tall/pad");
  check<T>(3, 6, 3, "wide");
  check<T>(3, 6, 5, "wide/pad");

  // 1x1: a single element scaled by d(0).
  check<T>(1, 1, 1, "one");
}

template<typename T>
void zero_dim_is_noop() {
  // m == 0 and n == 0: the wrapper enqueues nothing, so the whole sentinel
  // buffer survives -- which the host reference (an empty loop) also leaves
  // untouched. A non-zero ldx/n keeps the buffer (and the comparison) non-empty.
  check<T>(0, 4, 5, "zero_m");
  check<T>(4, 0, 5, "zero_n");
}

} // namespace

TEST(Lascl2OracleTests, AllShapes) {
  all_shapes<float>();
  all_shapes<double>();
}

TEST(Lascl2OracleTests, ZeroDimIsNoop) {
  zero_dim_is_noop<float>();
  zero_dim_is_noop<double>();
}

} // namespace calaman
