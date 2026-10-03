// Oracle test for calaman.lasrt (built under src/lasrt/) -- sort a 1-D array in
// place, into increasing or decreasing order.
//
// The oracle is the reference LAPACKE_?lasrt, run on the host over the identical
// input. A sort reorders values without computing any -- the result is a
// function of the input MULTISET and the direction alone -- so the device and
// the reference must agree BIT FOR BIT, and the comparison is `==`, not a
// tolerance (the same reasoning calaman.laset's oracle test uses: no arithmetic,
// so no rounding to tolerate).
//
// Cases cover both directions over a spread of sizes that straddle lasrt's
// insertion/quicksort threshold (<= 20 insertion, > 20 quicksort), including a
// large size that drives several partition levels and the explicit stack;
// duplicate-heavy, already-sorted and reverse-sorted inputs (the partition's
// worst and best cases); negatives and fractional values; and the n <= 1 early
// return (the wrapper enqueues nothing and succeeds). A host-only check confirms
// n < 0 returns an invalid-value Status without touching the device.
//
// REQUIRES_GPU (see CMakeLists.txt): every sorting case stages the array on the
// device and runs the sort kernel, so `ctest -LE gpu` excludes it. It needs the
// reference LAPACK, so the suite is guarded on calaman::lapack_reference at
// configure time (CMakeLists.txt) -- a missing oracle is a missing tier, not a
// silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lasrt;
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

// Reference ?lasrt, sorting d[0..n-1] in place.
lapack_int ref_lasrt(char id, lapack_int n, float *d) { return LAPACKE_slasrt(id, n, d); }
lapack_int ref_lasrt(char id, lapack_int n, double *d) { return LAPACKE_dlasrt(id, n, d); }

// SortDir -> the LAPACK id char the reference wants.
char id_char(SortDir id) { return id == SortDir::I ? 'I' : 'D'; }

// Sort `input` on the device and against the reference, and assert the two
// results agree bit for bit. The reference sorts a private copy; `input` itself
// is left untouched so a caller can reuse it for the other direction.
template<typename T>
void check(SortDir id, const std::vector<T> &input, const char *ctx) {
  auto handle = shared_device();
  const int n = static_cast<int>(input.size());

  auto d_d = to_device(handle, input);
  const auto status = calaman::lasrt<T>(handle->stream().get(), id, n, d_d.data());
  EXPECT_TRUE(status.ok()) << ctx << ": lasrt returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_d, input.size());

  std::vector<T> ref = input;
  const lapack_int info = ref_lasrt(id_char(id), n, ref.data());
  ASSERT_EQ(info, 0) << ctx << ": reference ?lasrt reported info=" << info;

  for (std::size_t i = 0; i < input.size(); ++i) {
    EXPECT_EQ(got[i], ref[i]) << ctx << ": mismatch at index " << i;
  }
}

// Run both directions over the same input.
template<typename T>
void check_both(const std::vector<T> &input, const char *ctx) {
  check<T>(SortDir::I, input, ctx);
  check<T>(SortDir::D, input, ctx);
}

// A deterministic finite fill in [-1000, 1000]; no NaN/Inf, whose ordering would
// be ill-defined and could legitimately differ between two sorts. mt19937 with a
// fixed seed keeps the suite reproducible.
template<typename T>
std::vector<T> make_random(std::size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    v[i] = static_cast<T>(dist(gen));
  }
  return v;
}

// ── cases ────────────────────────────────────────────────────────────────────

// Random inputs across sizes straddling the insertion/quicksort threshold (20),
// including a large one that drives several partition levels and the stack.
template<typename T>
void random_sizes() {
  unsigned seed = 12345;
  for (std::size_t n : {2u, 5u, 19u, 20u, 21u, 22u, 50u, 97u, 256u, 1000u}) {
    const auto v = make_random<T>(n, seed++);
    check_both<T>(v, "random_sizes");
  }
}

// A value threshold that must NOT revert to insertion: the smallest size the
// median-of-3 partition runs for, and one well past it.
template<typename T>
void duplicates() {
  // Many ties, so equal elements are indistinguishable -- the sorted output is
  // identical whatever order either sort places them in.
  std::mt19937 gen(777);
  std::uniform_int_distribution<int> dist(0, 4); // few distinct values => heavy ties
  std::vector<T> v(300);
  for (auto &x : v) {
    x = static_cast<T>(dist(gen));
  }
  check_both<T>(v, "duplicates");

  // All equal: the degenerate tie.
  std::vector<T> flat(64, static_cast<T>(3));
  check_both<T>(flat, "all_equal");
}

// Already-sorted and reverse-sorted inputs: the partition's structured extremes
// at a size past the threshold.
template<typename T>
void presorted() {
  const std::size_t n = 128;
  std::vector<T> inc(n), dec(n);
  for (std::size_t i = 0; i < n; ++i) {
    inc[i] = static_cast<T>(static_cast<double>(i) - 40.0);
    dec[i] = static_cast<T>(80.0 - static_cast<double>(i));
  }
  check_both<T>(inc, "already_increasing");
  check_both<T>(dec, "already_decreasing");
}

// n <= 1 is the early return: the wrapper enqueues nothing and succeeds, and the
// (empty or single-element) array is unchanged -- compared against the reference,
// which is likewise a no-op.
template<typename T>
void small_n() {
  check_both<T>(std::vector<T>{}, "empty");             // n == 0
  check_both<T>(std::vector<T>{static_cast<T>(42)}, "single"); // n == 1
}

// A negative length is rejected with an invalid-value Status, before any device
// work -- so this needs no device and passes a null array.
template<typename T>
void negative_n() {
  auto handle = shared_device();
  const auto status = calaman::lasrt<T>(handle->stream().get(), SortDir::I, -1,
                                        static_cast<T *>(nullptr));
  EXPECT_FALSE(status.ok()) << "negative_n: expected a failure Status";
  EXPECT_EQ(status, wwr::wwrErrorInvalidValue) << "negative_n: status=" << status.name();
}

} // namespace

TEST(LasrtOracleTests, RandomSizes) {
  random_sizes<float>();
  random_sizes<double>();
}

TEST(LasrtOracleTests, Duplicates) {
  duplicates<float>();
  duplicates<double>();
}

TEST(LasrtOracleTests, Presorted) {
  presorted<float>();
  presorted<double>();
}

TEST(LasrtOracleTests, SmallN) {
  small_n<float>();
  small_n<double>();
}

TEST(LasrtOracleTests, NegativeN) {
  negative_n<float>();
  negative_n<double>();
}

} // namespace calaman
