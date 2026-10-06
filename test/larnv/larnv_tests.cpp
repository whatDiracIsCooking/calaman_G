// Oracle test for calaman.larnv -- a random vector from LAPACK's stream. The
// oracle is LAPACKE_slarnv / dlarnv / clarnv / zlarnv from the same seed.
//
// IDIST 1/2 are ?laruv draws plus an exact affine map, so the values must be
// BITWISE equal, as must the updated ISEED for every IDIST. IDIST 3/4/5 go
// through device log/sqrt/cos/sin, so they are compared to the shared
// tolerance (kTolFactor * eps * max(1, |ref|)). Lengths span one partial chunk,
// exact multiples of 64 and several chunks plus a remainder; seeds are chained
// across calls. Single-precision seeds are built so a draw deep in the vector
// rounds to 1 (?laruv's retry), which shifts every later chunk's seed -- the
// path the one-block finish kernel replays.
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.larnv;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// The reference ?laruv, to read the multiplier table off the reference itself.
extern "C" {
void dlaruv_(int *iseed, const int *n, double *x);
}

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;
using test::kTolFactor;
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

// Per-type: the components of an element, and the reference call.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using Real = float;
  static constexpr int kMaxIdist = 3;
  static std::array<float, 2> parts(float v) { return {v, 0.0f}; }
  static void ref(int idist, int *iseed, int n, float *x) { LAPACKE_slarnv(idist, iseed, n, x); }
};
template<>
struct elem<double> {
  using Real = double;
  static constexpr int kMaxIdist = 3;
  static std::array<double, 2> parts(double v) { return {v, 0.0}; }
  static void ref(int idist, int *iseed, int n, double *x) { LAPACKE_dlarnv(idist, iseed, n, x); }
};
template<>
struct elem<wwr::wwrFloatComplex> {
  using Real = float;
  static constexpr int kMaxIdist = 5;
  static std::array<float, 2> parts(wwr::wwrFloatComplex z) {
    return {wwr::wwrCrealf(z), wwr::wwrCimagf(z)};
  }
  static void ref(int idist, int *iseed, int n, wwr::wwrFloatComplex *x) {
    LAPACKE_clarnv(idist, iseed, n, reinterpret_cast<lapack_complex_float *>(x));
  }
};
template<>
struct elem<wwr::wwrDoubleComplex> {
  using Real = double;
  static constexpr int kMaxIdist = 5;
  static std::array<double, 2> parts(wwr::wwrDoubleComplex z) {
    return {wwr::wwrCreal(z), wwr::wwrCimag(z)};
  }
  static void ref(int idist, int *iseed, int n, wwr::wwrDoubleComplex *x) {
    LAPACKE_zlarnv(idist, iseed, n, reinterpret_cast<lapack_complex_double *>(x));
  }
};

// One device call against the reference from the same seed; returns the
// reference's updated seed so callers can chain.
template<typename T>
Seed check(int idist, const Seed &seed, int n, const std::string &ctx) {
  using R = typename elem<T>::Real;
  auto handle = shared_device();
  Seed ref_seed = seed;
  std::vector<T> ref(static_cast<std::size_t>(n));
  elem<T>::ref(idist, ref_seed.data(), n, ref.data());

  auto d_iseed = to_device(handle, std::vector<int>(seed.begin(), seed.end()));
  DeviceBuffer<T> d_x(static_cast<std::size_t>(n), handle);
  const auto status = calaman::larnv<T>(handle->stream().get(), idist, d_iseed.data(),
                                        static_cast<std::size_t>(n), d_x.data());
  EXPECT_TRUE(status.ok()) << ctx << ": larnv returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());
  const auto got = from_device(handle, d_x, static_cast<std::size_t>(n));
  const auto got_seed = from_device(handle, d_iseed, 4);

  for (int k = 0; k < 4; ++k) {
    EXPECT_EQ(got_seed[k], ref_seed[k]) << ctx << ": iseed[" << k << "]";
  }
  int reported = 0;
  for (int i = 0; i < n && reported < 5; ++i) {
    const auto g = elem<T>::parts(got[i]);
    const auto r = elem<T>::parts(ref[i]);
    for (int p = 0; p < 2; ++p) {
      bool ok = false;
      if (idist <= 2) {
        ok = std::bit_cast<std::conditional_t<sizeof(R) == 4, std::uint32_t, std::uint64_t>>(g[p]) ==
             std::bit_cast<std::conditional_t<sizeof(R) == 4, std::uint32_t, std::uint64_t>>(r[p]);
      } else {
        ok = std::abs(g[p] - r[p]) <= kTolFactor<R> * eps<R>() * std::max(R(1), std::abs(r[p]));
      }
      if (!ok) {
        ADD_FAILURE() << ctx << ": x[" << i << "] part " << p << " device=" << g[p]
                      << " reference=" << r[p];
        ++reported;
      }
    }
  }
  return ref_seed;
}

template<typename T>
void all_distributions() {
  const std::vector<Seed> seeds = {{0, 0, 0, 1}, {1234, 567, 89, 1011}, {4095, 4095, 4095, 4095}};
  for (int idist = 1; idist <= elem<T>::kMaxIdist; ++idist) {
    for (const Seed &seed : seeds) {
      for (const int n : {1, 2, 63, 64, 65, 128, 200, 64 * 40 + 17}) {
        check<T>(idist, seed, n,
                 std::format("idist={} seed=({},{},{},{}) n={}", idist, seed[0], seed[1], seed[2],
                             seed[3], n));
      }
    }
  }
}

template<typename T>
void chained_calls() {
  for (int idist = 1; idist <= elem<T>::kMaxIdist; ++idist) {
    Seed seed = {11, 22, 33, 45};
    for (int call = 0; call < 4; ++call) {
      seed = check<T>(idist, seed, 100 + 37 * call, std::format("idist={} call {}", idist, call));
    }
  }
}

// --- single-precision retry seeds -------------------------------------------

constexpr std::uint64_t kMask48 = (std::uint64_t{1} << 48) - 1;

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

std::uint64_t mulmod(std::uint64_t a, std::uint64_t b) {
  return (a * b) & kMask48;
}

std::uint64_t powmod(std::uint64_t b, std::uint64_t e) {
  std::uint64_t r = 1;
  for (; e != 0; e >>= 1, b = mulmod(b, b)) {
    if ((e & 1) != 0) {
      r = mulmod(r, b);
    }
  }
  return r;
}

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

std::uint64_t join(const Seed &s) {
  return (std::uint64_t(s[0]) << 36) | (std::uint64_t(s[1]) << 24) | (std::uint64_t(s[2]) << 12) |
         std::uint64_t(s[3]);
}

// A seed whose draw `i` of chunk `c` (draws_per_chunk per full chunk) has its
// top 36 bits set, so it rounds to 1.0 in float and the reference retries.
Seed retry_seed(int draws_per_chunk, std::uint64_t c, int i) {
  const auto mm = multipliers();
  const std::uint64_t target = (std::uint64_t{0xFFFFFFFFF} << 12) | 0xABD;
  const std::uint64_t jump_inv = inverse48(mm[draws_per_chunk - 1]);
  return limbs(mulmod(mulmod(target, inverse48(mm[i])), powmod(jump_inv, c)));
}

template<typename T>
void single_precision_retry(int idist, int draws_per_chunk) {
  const auto mm = multipliers();
  const int n = 64 * 30 + 9;
  for (const std::uint64_t c : {std::uint64_t{0}, std::uint64_t{3}, std::uint64_t{17}}) {
    const Seed seed = retry_seed(draws_per_chunk, c, 10);
    // With no retry the final seed is seed * J^(chunks-1) * MM(last il2 - 1).
    const int last_il2 = draws_per_chunk == 64 ? 9 : 18;
    const Seed no_retry = limbs(mulmod(mulmod(join(seed), powmod(mm[draws_per_chunk - 1], 30)),
                                       mm[last_il2 - 1]));
    const Seed after = check<T>(idist, seed, n, std::format("retry in chunk {}", c));
    EXPECT_NE(after, no_retry) << "seed for chunk " << c << " did not retry";
  }
}

template<typename T>
void invalid_and_empty() {
  auto handle = shared_device();
  const std::vector<int> seed = {1, 2, 3, 5};
  auto d_iseed = to_device(handle, seed);
  DeviceBuffer<T> d_x(4, handle);
  const auto stream = handle->stream().get();
  EXPECT_FALSE(calaman::larnv<T>(stream, 0, d_iseed.data(), 4, d_x.data()).ok());
  EXPECT_FALSE(
      calaman::larnv<T>(stream, elem<T>::kMaxIdist + 1, d_iseed.data(), 4, d_x.data()).ok());
  EXPECT_TRUE(calaman::larnv<T>(stream, 1, d_iseed.data(), 0, d_x.data()).ok());
  wwr::wwrStreamSynchronize(stream);
  EXPECT_EQ(from_device(handle, d_iseed, 4), seed);
}

} // namespace

TEST(LarnvOracleTests, Float) {
  all_distributions<float>();
}

TEST(LarnvOracleTests, Double) {
  all_distributions<double>();
}

TEST(LarnvOracleTests, ComplexFloat) {
  all_distributions<wwr::wwrFloatComplex>();
}

TEST(LarnvOracleTests, ComplexDouble) {
  all_distributions<wwr::wwrDoubleComplex>();
}

TEST(LarnvOracleTests, ChainedCalls) {
  chained_calls<float>();
  chained_calls<double>();
  chained_calls<wwr::wwrFloatComplex>();
  chained_calls<wwr::wwrDoubleComplex>();
}

TEST(LarnvOracleTests, SinglePrecisionRetry) {
  single_precision_retry<float>(1, 64);
  single_precision_retry<float>(3, 128);
  single_precision_retry<wwr::wwrFloatComplex>(1, 128);
}

TEST(LarnvOracleTests, InvalidAndEmpty) {
  invalid_and_empty<float>();
  invalid_and_empty<double>();
  invalid_and_empty<wwr::wwrFloatComplex>();
  invalid_and_empty<wwr::wwrDoubleComplex>();
}

} // namespace calaman
