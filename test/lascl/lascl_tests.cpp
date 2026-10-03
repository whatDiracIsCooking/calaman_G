// Oracle test for calaman.lascl (built under src/lascl/) -- multiply a
// column-major matrix by cto/cfrom without over/underflow, LAPACK ?lascl for a
// full matrix (TYPE='G').
//
// The oracle is the reference LAPACKE_?lascl run on the host over the identical
// inputs. Both the device and the reference apply the SAME guarded multiplier
// chain, so they agree to rounding, not bit for bit: the comparison is a
// per-element relative-or-absolute tolerance scaled by eps, not `==`. Each case
// pre-fills the whole lda-by-n buffer with a distinct per-element value and
// compares the whole buffer, so the lda-m padding rows beyond the leading m are
// checked to stay untouched -- exactly as the reference leaves them.
//
// Cases cover square/tall/wide shapes, a leading dimension strictly greater than
// m (padding rows that must stay untouched), a 1x1 matrix, an ordinary ratio and
// two ratios chosen to drive the guarded loop through its smlnum/bignum steps
// (cto/cfrom far apart in both directions), and the m==0/n==0 early-return.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the matrix on the device
// and runs the scaling kernel, so `ctest -LE gpu` excludes it. It needs the
// reference LAPACK, so the suite is guarded on calaman::lapack_reference at
// configure time (CMakeLists.txt) -- a missing oracle is a missing tier, not a
// silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lascl;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;
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

// The reference, dispatched on T. LAPACKE_?lascl with TYPE='G' (KL/KU unused)
// over the same column-major lda-by-n buffer, in place.
template<typename T>
void reference_lascl(T cfrom, T cto, int m, int n, std::vector<T> &a, int lda);

template<>
void reference_lascl<float>(float cfrom, float cto, int m, int n, std::vector<float> &a, int lda) {
  LAPACKE_slascl(LAPACK_COL_MAJOR, 'G', 0, 0, cfrom, cto, m, n, a.data(), lda);
}

template<>
void reference_lascl<double>(double cfrom, double cto, int m, int n, std::vector<double> &a,
                             int lda) {
  LAPACKE_dlascl(LAPACK_COL_MAJOR, 'G', 0, 0, cfrom, cto, m, n, a.data(), lda);
}

// A distinct, moderate-magnitude value per matrix slot, so a mis-indexed write
// (or a stray write into the lda padding) lands on a value no correct run would
// produce. Kept O(1) so the scaled result stays well inside range.
template<typename T>
T fill(std::size_t k) {
  return static_cast<T>(1) + static_cast<T>(k) * static_cast<T>(0.5);
}

// Stage the SAME buffer on host and device, scale both by cto/cfrom, and assert
// the whole lda-by-n buffer agrees to a relative-or-absolute tolerance -- scaled
// region and untouched padding alike (the padding compares exactly, as neither
// path touches it).
template<typename T>
void check(T cfrom, T cto, int m, int n, int lda, const char *ctx) {
  auto handle = shared_device();
  const std::size_t size = static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);

  std::vector<T> ref(size);
  for (std::size_t k = 0; k < size; ++k) {
    ref[k] = fill<T>(k);
  }

  auto d_a = to_device(handle, ref); // device starts from the identical fill
  reference_lascl<T>(cfrom, cto, m, n, ref, lda);

  const auto status = calaman::lascl<T>(handle->stream().get(), cfrom, cto,
                                        static_cast<std::size_t>(m), static_cast<std::size_t>(n),
                                        d_a.data(), static_cast<std::size_t>(lda));
  EXPECT_TRUE(status.ok()) << ctx << ": lascl returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_a, size);

  // One multiply per element, so a relative error of a few eps is the bound; a
  // small absolute floor covers an element the reference drives to zero.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < lda; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * lda + i;
      const T tol = static_cast<T>(16) * eps<T>() * (std::abs(ref[k]) + static_cast<T>(1));
      EXPECT_NEAR(got[k], ref[k], tol) << ctx << ": mismatch at (" << i << "," << j << ")";
    }
  }
}

template<typename T>
void all_shapes() {
  // An ordinary ratio over square (with and without padding), tall and wide.
  check<T>(2, 3, 4, 4, 4, "square");
  check<T>(2, 3, 5, 5, 7, "square/pad");
  check<T>(4, 7, 6, 3, 6, "tall");
  check<T>(4, 7, 6, 3, 8, "tall/pad");
  check<T>(4, 7, 3, 6, 3, "wide");
  check<T>(4, 7, 3, 6, 5, "wide/pad");
  check<T>(3, 5, 1, 1, 1, "one");

  // cto/cfrom far apart in each direction, so the guarded loop takes its
  // smlnum/bignum branches rather than a single multiply.
  const T big = std::numeric_limits<T>::max();
  const T small = std::numeric_limits<T>::min();
  check<T>(small, big, 4, 4, 4, "up");   // ratio huge: repeated bignum steps
  check<T>(big, small, 4, 4, 4, "down"); // ratio tiny: repeated smlnum steps
}

template<typename T>
void zero_dim_is_noop() {
  // m == 0 and n == 0: the wrapper enqueues nothing, so the whole fill buffer
  // survives -- which the reference (a no-op for a zero extent) also leaves
  // untouched. A non-zero lda/n keeps the buffer (and the comparison) non-empty.
  check<T>(2, 3, 0, 4, 5, "zero_m");
  check<T>(2, 3, 4, 0, 5, "zero_n");
}

template<typename T>
void bad_cfrom_rejected() {
  // cfrom == 0 cannot form a ratio: ?lascl's INFO=-4, surfaced as an
  // InvalidValue runtime Status, with the matrix left untouched.
  auto handle = shared_device();
  std::vector<T> a{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)};
  auto d_a = to_device(handle, a);
  const auto status =
      calaman::lascl<T>(handle->stream().get(), static_cast<T>(0), static_cast<T>(5), 2, 2,
                        d_a.data(), 2);
  EXPECT_FALSE(status.ok()) << "cfrom==0 should be rejected";
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_a, a.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    EXPECT_EQ(got[k], a[k]) << "matrix must be untouched on a rejected cfrom";
  }
}

} // namespace

TEST(LasclOracleTests, AllShapes) {
  all_shapes<float>();
  all_shapes<double>();
}

TEST(LasclOracleTests, ZeroDimIsNoop) {
  zero_dim_is_noop<float>();
  zero_dim_is_noop<double>();
}

TEST(LasclOracleTests, BadCfromRejected) {
  bad_cfrom_rejected<float>();
  bad_cfrom_rejected<double>();
}

} // namespace calaman
