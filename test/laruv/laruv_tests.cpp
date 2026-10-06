// Oracle test for calaman.laruv -- LAPACK's uniform (0,1) generator ?laruv,
// and through it the laruv_block header. LAPACKE has no ?laruv binding, so the
// reference is the Fortran slaruv_ / dlaruv_ via calaman::lapack_reference.
//
// The port is integer limb arithmetic plus power-of-two scaling, so the device
// must agree with the reference BIT FOR BIT, both the draws and the updated
// ISEED, compared with std::bit_cast rather than a tolerance. Cases: a spread of
// seeds over n in {1, 2, 64, 127, 128}; a chain of calls threading the seed;
// float seeds built so a draw rounds to 1.0 (the reference's +2 retry), at the
// first index and mid-block; n == 0 (no-op) and n == 129 (invalid value).
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.laruv;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

// The reference ?laruv, Fortran-mangled; everything by reference.
extern "C" {
void slaruv_(int *iseed, const int *n, float *x);
void dlaruv_(int *iseed, const int *n, double *x);
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

using Seed = std::array<int, 4>;

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

void ref_laruv(int *iseed, int n, float *x) {
  slaruv_(iseed, &n, x);
}
void ref_laruv(int *iseed, int n, double *x) {
  dlaruv_(iseed, &n, x);
}

template<typename T>
auto bits(T v) {
  if constexpr (std::is_same_v<T, float>) {
    return std::bit_cast<std::uint32_t>(v);
  } else {
    return std::bit_cast<std::uint64_t>(v);
  }
}

// One device call against the reference from the same seed; returns the
// reference's updated seed so callers can chain.
template<typename T>
Seed check(const Seed &seed, int n, const std::string &ctx) {
  auto handle = shared_device();
  Seed ref_seed = seed;
  std::vector<T> ref(static_cast<std::size_t>(n));
  ref_laruv(ref_seed.data(), n, ref.data());

  auto d_iseed = to_device(handle, std::vector<int>(seed.begin(), seed.end()));
  DeviceBuffer<T> d_x(static_cast<std::size_t>(n), handle);
  const auto status = calaman::laruv<T>(handle->stream().get(), d_iseed.data(), n, d_x.data());
  EXPECT_TRUE(status.ok()) << ctx << ": laruv returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_x, static_cast<std::size_t>(n));
  const auto got_seed = from_device(handle, d_iseed, 4);

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bits(got[i]), bits(ref[i]))
        << ctx << ": x[" << i << "] device=" << got[i] << " reference=" << ref[i];
  }
  for (int k = 0; k < 4; ++k) {
    EXPECT_EQ(got_seed[k], ref_seed[k]) << ctx << ": iseed[" << k << "]";
  }
  return ref_seed;
}

const std::vector<Seed> &seeds() {
  static const std::vector<Seed> s = {
      {0, 0, 0, 1},       {1, 2, 3, 5},          {4095, 4095, 4095, 4095},
      {1234, 567, 89, 1011}, {3000, 17, 2048, 4093}, {0, 4095, 0, 2049},
  };
  return s;
}

template<typename T>
void seed_spread() {
  for (const Seed &seed : seeds()) {
    for (const int n : {1, 2, 64, 127, 128}) {
      check<T>(seed, n,
               std::format("seed=({},{},{},{}) n={}", seed[0], seed[1], seed[2], seed[3], n));
    }
  }
}

// Thread the updated seed through consecutive calls, as ?larnv does.
template<typename T>
void chained_calls() {
  Seed seed = {11, 22, 33, 45};
  for (int call = 0; call < 8; ++call) {
    seed = check<T>(seed, call % 2 == 0 ? 128 : 37, std::format("chain call {}", call));
  }
}

// The multipliers as 48-bit integers, read off the reference itself: seed
// (0,0,0,1) times row i, mod 2^48, is row i, and dlaruv_ returns it / 2^48
// exactly.
std::array<std::uint64_t, 128> multipliers() {
  Seed seed = {0, 0, 0, 1};
  std::array<double, 128> x{};
  const int n = 128;
  dlaruv_(seed.data(), &n, x.data());
  std::array<std::uint64_t, 128> mm{};
  for (std::size_t i = 0; i < 128; ++i) {
    mm[i] = static_cast<std::uint64_t>(std::ldexp(x[i], 48));
  }
  return mm;
}

constexpr std::uint64_t kMask48 = (std::uint64_t{1} << 48) - 1;

// The inverse of an odd m mod 2^48, by Newton's iteration.
std::uint64_t inverse48(std::uint64_t m) {
  std::uint64_t inv = m;
  for (int k = 0; k < 6; ++k) {
    inv *= 2 - m * inv;
  }
  return inv & kMask48;
}

Seed limbs(std::uint64_t v) {
  return {static_cast<int>((v >> 36) & 4095), static_cast<int>((v >> 24) & 4095),
          static_cast<int>((v >> 12) & 4095), static_cast<int>(v & 4095)};
}

// A seed whose draw at index `hit` has its top 36 bits set, which rounds to 1.0
// in float and makes the reference bump every seed limb by 2 and redraw.
Seed retry_seed(std::size_t hit) {
  const auto mm = multipliers();
  const std::uint64_t target = (std::uint64_t{0xFFFFFFFFF} << 12) | 0xABD;
  return limbs((target * inverse48(mm[hit])) & kMask48);
}

void float_retry() {
  const auto mm = multipliers();
  for (const std::size_t hit : {std::size_t{0}, std::size_t{5}, std::size_t{100}}) {
    const Seed seed = retry_seed(hit);
    const int n = 128;
    // Without the retry the updated seed would be seed * MM(n) mod 2^48; the
    // reference's differing proves this case does take the retry path.
    const std::uint64_t s = (std::uint64_t(seed[0]) << 36) | (std::uint64_t(seed[1]) << 24) |
                            (std::uint64_t(seed[2]) << 12) | std::uint64_t(seed[3]);
    const Seed no_retry = limbs((s * mm[n - 1]) & kMask48);
    const Seed after = check<float>(seed, n, std::format("float retry at {}", hit));
    EXPECT_NE(after, no_retry) << "retry seed for index " << hit << " did not retry";
    // A retry hit past the end of a short call changes nothing.
    check<float>(seed, static_cast<int>(hit) + 1, std::format("float retry, n={}", hit + 1));
  }
}

template<typename T>
void invalid_and_empty() {
  auto handle = shared_device();
  const std::vector<int> seed = {1, 2, 3, 5};
  auto d_iseed = to_device(handle, seed);
  DeviceBuffer<T> d_x(129, handle);
  EXPECT_FALSE(calaman::laruv<T>(handle->stream().get(), d_iseed.data(), 129, d_x.data()).ok());
  EXPECT_FALSE(calaman::laruv<T>(handle->stream().get(), d_iseed.data(), -1, d_x.data()).ok());
  EXPECT_TRUE(calaman::laruv<T>(handle->stream().get(), d_iseed.data(), 0, d_x.data()).ok());
  wwr::wwrStreamSynchronize(handle->stream().get());
  EXPECT_EQ(from_device(handle, d_iseed, 4), seed);
}

} // namespace

TEST(LaruvOracleTests, SeedSpreadFloat) {
  seed_spread<float>();
}

TEST(LaruvOracleTests, SeedSpreadDouble) {
  seed_spread<double>();
}

TEST(LaruvOracleTests, ChainedCalls) {
  chained_calls<float>();
  chained_calls<double>();
}

TEST(LaruvOracleTests, FloatRetryOnOne) {
  float_retry();
}

TEST(LaruvOracleTests, InvalidAndEmpty) {
  invalid_and_empty<float>();
  invalid_and_empty<double>();
}

} // namespace calaman
