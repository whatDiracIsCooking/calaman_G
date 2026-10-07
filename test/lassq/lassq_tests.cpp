// Oracle test for calaman.lassq -- the ?lassq scaled sum of squares of a
// strided vector, accumulated into a (scale, sumsq) pair. The oracle is the
// reference ?lassq (LAPACK_{s,d,c,z}lassq) in the SAME precision on the host,
// over the same buffers (the wwr complex types are layout-compatible with
// lapack_complex_*).
//
// WHAT IS COMPARED is the value the pair REPRESENTS, scale * sqrt(sumsq),
// computed in long double. The pair's own normalization is an implementation
// choice -- the reference is free to hand back any (s, q) with s^2*q equal to
// the sum -- while the represented value is the contract. long double is what
// lets a case deliberately placed near the overflow or underflow threshold
// (the whole point of ?lassq) still be compared on the host at all. The one
// exception is n == 0, which is exact bookkeeping with no arithmetic, so the
// raw pair is compared there.
//
// Sums are folded in a different order than the reference's single loop, so
// agreement is to rounding, not bit-for-bit.
//
// All suites stage data on the device, so they are REQUIRES_GPU (labeled
// `gpu`). Built only when calaman::lapack_reference exists; see this
// directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lassq;
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
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// ========================================================================
// Per-element-type traits: the builder, the oracle, and the two exponent
// shifts that put a generated vector outside ?lassq's mid-range band
// ========================================================================

// LAPACKE has no ?lassq wrapper, so each oracle calls the Fortran routine
// through lapack.h's LAPACK_?lassq (which lapacke.h includes).
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr int kBigShift = 60;    // above tbig = 2^52
  static constexpr int kSmallShift = -70; // below tsml = 2^-63
  static float make(double re, double) { return static_cast<float>(re); }
  static void ref(std::size_t n, const float *x, int incx, float *scale, float *sumsq) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    LAPACK_slassq(&ni, x, &ii, scale, sumsq);
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr int kBigShift = 500;    // above tbig = 2^486
  static constexpr int kSmallShift = -520; // below tsml = 2^-511
  static double make(double re, double) { return re; }
  static void ref(std::size_t n, const double *x, int incx, double *scale, double *sumsq) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    LAPACK_dlassq(&ni, x, &ii, scale, sumsq);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static constexpr int kBigShift = 60;
  static constexpr int kSmallShift = -70;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static void ref(std::size_t n, const wwr::wwrFloatComplex *x, int incx, float *scale,
                  float *sumsq) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    LAPACK_classq(&ni, reinterpret_cast<const lapack_complex_float *>(x), &ii, scale, sumsq);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static constexpr int kBigShift = 500;
  static constexpr int kSmallShift = -520;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static void ref(std::size_t n, const wwr::wwrDoubleComplex *x, int incx, double *scale,
                  double *sumsq) {
    const lapack_int ni = static_cast<lapack_int>(n);
    const lapack_int ii = static_cast<lapack_int>(incx);
    LAPACK_zlassq(&ni, reinterpret_cast<const lapack_complex_double *>(x), &ii, scale, sumsq);
  }
};

// ========================================================================
// Staging and comparison
// ========================================================================

template<typename T>
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
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
std::vector<T> from_device(const DeviceBuffer<T> &device, std::size_t n) {
  const auto handle = shared_device();
  HostBuffer<T> host(n == 0 ? 1 : n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

/// The value a (scale, sumsq) pair stands for, in a type wide enough to hold it
/// even when the pair was built precisely because the value does not fit in R.
template<typename R>
long double represented(R scale, R sumsq) {
  return static_cast<long double>(scale) * std::sqrt(static_cast<long double>(sumsq));
}

// The device folds its share of the sum per thread and then through a tree,
// while the reference runs one sequential loop, so the two round differently.
// The bound is kTolFactor * eps * sqrt(n): the growth of a sum of n terms of
// like magnitude, which both sides are, with kTolFactor absorbing the constant.
// Deliberately not eps * n -- that worst case needs adversarial magnitudes, and
// at n = 4099 in single precision it would be a 3% tolerance, loose enough to
// pass a wrong algorithm.
template<typename R>
void expect_close(long double got, long double want, std::size_t n, const std::string &what) {
  if (want == 0.0L) {
    EXPECT_EQ(got, 0.0L) << what;
    return;
  }
  ASSERT_TRUE(std::isfinite(want)) << what << ": oracle is not finite (" << want << ")";
  const long double rel = std::fabs(got - want) / std::fabs(want);
  const long double tol = static_cast<long double>(test::kTolFactor<R>) *
                          static_cast<long double>(test::eps<R>()) *
                          std::sqrt(static_cast<long double>(n == 0 ? 1 : n));
  EXPECT_LE(rel, tol) << what << ": got " << got << ", want " << want;
}

// ========================================================================
// Input generation
// ========================================================================

// Mixed-sign values over a few binades, every component shifted by 2^shift so
// the whole vector can be placed in any of ?lassq's three bands.
template<typename T>
std::vector<T> random_vector(std::size_t len, std::uint32_t seed, int shift = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  std::vector<T> v(len);
  for (auto &e : v) {
    e = elem<T>::make(std::ldexp(dist(gen), shift), std::ldexp(dist(gen), shift));
  }
  return v;
}

// One vector straddling all three bands, so the combine step has to merge them:
// every third element is huge, the next tiny, the next mid-range.
template<typename T>
std::vector<T> three_band_vector(std::size_t len, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  const int shifts[3] = {elem<T>::kBigShift, elem<T>::kSmallShift, 0};
  std::vector<T> v(len);
  for (std::size_t i = 0; i < len; ++i) {
    const int s = shifts[i % 3];
    v[i] = elem<T>::make(std::ldexp(dist(gen), s), std::ldexp(dist(gen), s));
  }
  return v;
}

// ========================================================================
// The one comparison every case runs
// ========================================================================

// Runs the device routine and the reference over the SAME host vector and seed
// pair, and compares the represented values. @p ref_incx is the stride the
// oracle is called with: for a negative incx the device visits x[0], x[|incx|],
// ..., x[(n-1)*|incx|] -- the same elements a positive |incx| visits -- so the
// oracle is called with |incx| rather than relying on the reference's own
// negative-stride handling, which older LAPACK does not have.
template<typename T>
void expect_matches_reference(const std::vector<T> &x, std::size_t n, int incx,
                              typename elem<T>::R scale_in, typename elem<T>::R sumsq_in,
                              const std::string &what) {
  using R = typename elem<T>::R;

  R ref_scale = scale_in;
  R ref_sumsq = sumsq_in;
  elem<T>::ref(n, x.data(), incx < 0 ? -incx : incx, &ref_scale, &ref_sumsq);

  auto d_x = to_device(x);
  auto d_pair = to_device(std::vector<R>{scale_in, sumsq_in});
  const Status s = lassq<T>(shared_device()->stream().get(), static_cast<int>(n), d_x.data(), incx,
                            d_pair.data(), d_pair.data() + 1);
  ASSERT_TRUE(s.ok()) << what << ": lassq returned " << s.name() << ": " << s.message();
  wwr::wwrStreamSynchronize(shared_device()->stream().get());
  const auto pair = from_device(d_pair, 2);

  expect_close<R>(represented(pair[0], pair[1]), represented(ref_scale, ref_sumsq), n, what);

  // ?lassq reads x and never writes it.
  EXPECT_EQ(std::memcmp(from_device(d_x, x.size()).data(), x.data(), x.size() * sizeof(T)), 0)
      << what << ": x was modified";
}

constexpr std::size_t kSizes[] = {1, 2, 3, 7, 64, 255, 256, 257, 1000, 4099};

// ========================================================================
// Well-scaled input, every size: the chunk-and-tree fold itself
// ========================================================================

template<typename T>
void run_sizes() {
  for (const std::size_t n : kSizes) {
    const auto x = random_vector<T>(n, static_cast<std::uint32_t>(17 * n + 1));
    expect_matches_reference<T>(x, n, 1, 1, 0, "n=" + std::to_string(n));
  }
}

TEST(LassqOracleTests, SizesFloat) {
  run_sizes<float>();
}
TEST(LassqOracleTests, SizesDouble) {
  run_sizes<double>();
}
TEST(LassqOracleTests, SizesComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LassqOracleTests, SizesComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// ========================================================================
// The three bands, which are the reason ?lassq is not a plain sum
// ========================================================================

// Every entry above tbig: squaring one directly would overflow, yet the norm of
// the whole vector is finite and must come out right.
template<typename T>
void run_big() {
  for (const std::size_t n : {std::size_t{1}, std::size_t{300}, std::size_t{4099}}) {
    const auto x = random_vector<T>(n, static_cast<std::uint32_t>(31 * n + 5), elem<T>::kBigShift);
    expect_matches_reference<T>(x, n, 1, 1, 0, "big n=" + std::to_string(n));
  }
}

// Every entry below tsml: squaring one directly would underflow to zero.
template<typename T>
void run_small() {
  for (const std::size_t n : {std::size_t{1}, std::size_t{300}, std::size_t{4099}}) {
    const auto x =
        random_vector<T>(n, static_cast<std::uint32_t>(37 * n + 7), elem<T>::kSmallShift);
    expect_matches_reference<T>(x, n, 1, 1, 0, "small n=" + std::to_string(n));
  }
}

// All three bands in one vector, so the combine step has to merge them -- and,
// because some entry is above tbig, discard asml exactly as `notbig` does.
template<typename T>
void run_three_band() {
  for (const std::size_t n : {std::size_t{3}, std::size_t{300}, std::size_t{4099}}) {
    const auto x = three_band_vector<T>(n, static_cast<std::uint32_t>(41 * n + 9));
    expect_matches_reference<T>(x, n, 1, 1, 0, "bands n=" + std::to_string(n));
  }
}

TEST(LassqOracleTests, AboveTbigFloat) {
  run_big<float>();
}
TEST(LassqOracleTests, AboveTbigDouble) {
  run_big<double>();
}
TEST(LassqOracleTests, AboveTbigComplexDouble) {
  run_big<wwr::wwrDoubleComplex>();
}
TEST(LassqOracleTests, BelowTsmlFloat) {
  run_small<float>();
}
TEST(LassqOracleTests, BelowTsmlDouble) {
  run_small<double>();
}
TEST(LassqOracleTests, BelowTsmlComplexDouble) {
  run_small<wwr::wwrDoubleComplex>();
}
TEST(LassqOracleTests, ThreeBandsFloat) {
  run_three_band<float>();
}
TEST(LassqOracleTests, ThreeBandsDouble) {
  run_three_band<double>();
}
TEST(LassqOracleTests, ThreeBandsComplexFloat) {
  run_three_band<wwr::wwrFloatComplex>();
}
TEST(LassqOracleTests, ThreeBandsComplexDouble) {
  run_three_band<wwr::wwrDoubleComplex>();
}

// ========================================================================
// The incoming pair: ?lassq's second stage, which places it in a band too
// ========================================================================

// A seed pair in each band, so the "put the existing sum into one of the
// accumulators" step is exercised in all three of its arms -- including the
// one guarded by `notbig`, reached here by a tiny seed beside a huge vector.
template<typename T>
void run_seeded() {
  using R = typename elem<T>::R;
  const R big = std::ldexp(R{1}, elem<T>::kBigShift);
  const R small = std::ldexp(R{1}, elem<T>::kSmallShift);
  const std::size_t n = 300;

  const struct {
    R scale;
    R sumsq;
    const char *name;
  } seeds[] = {{1, 0, "fresh"},        {1, 2.5, "mid"},         {big, 3, "huge"},
               {small, 3, "tiny"},     {0, 5, "zero-scale"},    {7, 0, "zero-sumsq"}};

  for (const auto &seed : seeds) {
    for (const int shift : {0, elem<T>::kBigShift, elem<T>::kSmallShift}) {
      const auto x = random_vector<T>(n, 4321, shift);
      expect_matches_reference<T>(x, n, 1, seed.scale, seed.sumsq,
                                  std::string("seed=") + seed.name + " shift=" +
                                      std::to_string(shift));
    }
  }
}

TEST(LassqOracleTests, SeededPairFloat) {
  run_seeded<float>();
}
TEST(LassqOracleTests, SeededPairDouble) {
  run_seeded<double>();
}
TEST(LassqOracleTests, SeededPairComplexDouble) {
  run_seeded<wwr::wwrDoubleComplex>();
}

// ========================================================================
// Strides
// ========================================================================

// incx != 1: the slots BETWEEN the visited elements hold NaN, so a kernel that
// ignored the stride would return NaN. A negative incx must visit exactly the
// elements its positive counterpart does (?lassq's ix starts at the far end).
template<typename T>
void run_strides() {
  const std::size_t n = 257;
  for (const int incx : {2, 3, -1, -2, -3}) {
    const std::size_t stride = static_cast<std::size_t>(incx < 0 ? -incx : incx);
    auto x = random_vector<T>((n - 1) * stride + 1, 77);
    const T poison = elem<T>::make(std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (i % stride != 0) {
        x[i] = poison;
      }
    }
    expect_matches_reference<T>(x, n, incx, 1, 0, "incx=" + std::to_string(incx));
  }
}

TEST(LassqOracleTests, StridesFloat) {
  run_strides<float>();
}
TEST(LassqOracleTests, StridesDouble) {
  run_strides<double>();
}
TEST(LassqOracleTests, StridesComplexDouble) {
  run_strides<wwr::wwrDoubleComplex>();
}

// incx == 0 is not a vector; both paths read x[0] n times, which is what the
// reference's `ix = ix + incx` stepping does for a zero increment.
TEST(LassqOracleTests, ZeroStrideRepeatsFirstElement) {
  const auto x = random_vector<double>(16, 909);
  expect_matches_reference<double>(x, 9, 0, 1, 0, "incx=0");
}

// ========================================================================
// Edges: n <= 0, and a NaN on either side
// ========================================================================

// n <= 0 runs only ?lassq's canonicalization of a pair representing zero, which
// is exact bookkeeping -- so the raw pair is compared, not its value.
TEST(LassqOracleTests, EmptyCanonicalizesPair) {
  const std::vector<double> dummy = {42.0};
  const struct {
    double scale;
    double sumsq;
  } seeds[] = {{1, 0}, {7, 0}, {0, 5}, {3, 4}};

  for (const auto &seed : seeds) {
    for (const int n : {0, -5}) {
      double ref_scale = seed.scale;
      double ref_sumsq = seed.sumsq;
      // The reference's own n <= 0 path, which still normalizes before returning.
      elem<double>::ref(n < 0 ? 0 : static_cast<std::size_t>(n), dummy.data(), 1, &ref_scale,
                        &ref_sumsq);

      auto d_x = to_device(dummy);
      auto d_pair = to_device(std::vector<double>{seed.scale, seed.sumsq});
      ASSERT_TRUE(lassq<double>(shared_device()->stream().get(), n, d_x.data(), 1, d_pair.data(),
                                d_pair.data() + 1)
                      .ok());
      wwr::wwrStreamSynchronize(shared_device()->stream().get());
      const auto pair = from_device(d_pair, 2);
      EXPECT_EQ(pair[0], ref_scale) << "n=" << n << " seed=(" << seed.scale << ", " << seed.sumsq
                                    << ")";
      EXPECT_EQ(pair[1], ref_sumsq) << "n=" << n << " seed=(" << seed.scale << ", " << seed.sumsq
                                    << ")";
    }
  }
}

// A NaN entry lands in amed, which propagates it -- the reference's own path.
TEST(LassqOracleTests, NaNEntryPropagates) {
  const std::size_t n = 600; // chunk 3, so the NaN is interior to one thread's share
  for (const std::size_t pos : {std::size_t{0}, std::size_t{7}, n - 1}) {
    auto x = random_vector<double>(n, 5);
    x[pos] = std::numeric_limits<double>::quiet_NaN();
    auto d_x = to_device(x);
    auto d_pair = to_device(std::vector<double>{1.0, 0.0});
    ASSERT_TRUE(lassq<double>(shared_device()->stream().get(), static_cast<int>(n), d_x.data(), 1,
                              d_pair.data(), d_pair.data() + 1)
                    .ok());
    wwr::wwrStreamSynchronize(shared_device()->stream().get());
    const auto pair = from_device(d_pair, 2);
    EXPECT_TRUE(std::isnan(pair[1])) << "pos=" << pos << " sumsq=" << pair[1];
  }
}

// A NaN in either incoming scalar leaves BOTH untouched (DLASSQ's quick return),
// which is the one case where the pair is not even canonicalized.
TEST(LassqOracleTests, NaNSeedLeavesPairUntouched) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto x = random_vector<double>(100, 13);
  for (const int which : {0, 1}) {
    std::vector<double> seed = {2.0, 3.0};
    seed[which] = nan;
    auto d_x = to_device(x);
    auto d_pair = to_device(seed);
    ASSERT_TRUE(lassq<double>(shared_device()->stream().get(), 100, d_x.data(), 1, d_pair.data(),
                              d_pair.data() + 1)
                    .ok());
    wwr::wwrStreamSynchronize(shared_device()->stream().get());
    const auto pair = from_device(d_pair, 2);
    EXPECT_EQ(std::isnan(pair[0]), std::isnan(seed[0])) << "which=" << which;
    EXPECT_EQ(std::isnan(pair[1]), std::isnan(seed[1])) << "which=" << which;
    if (!std::isnan(seed[which == 0 ? 1 : 0])) {
      EXPECT_EQ(pair[which == 0 ? 1 : 0], seed[which == 0 ? 1 : 0]) << "which=" << which;
    }
  }
}

// A zero vector leaves the pair representing exactly zero, through amed == 0.
TEST(LassqOracleTests, ZeroVector) {
  const std::vector<double> x(500, 0.0);
  expect_matches_reference<double>(x, x.size(), 1, 1, 0, "zeros");
}

// ========================================================================
// Accumulating across calls -- what the pair exists for
// ========================================================================

// Three chained calls over three chunks of one vector must land where one call
// over the whole vector does: the Frobenius-norm-by-columns pattern.
TEST(LassqOracleTests, ChainedCallsMatchSingleCall) {
  const std::size_t n = 900;
  const auto x = random_vector<double>(n, 2024);

  auto d_x = to_device(x);
  auto d_chained = to_device(std::vector<double>{1.0, 0.0});
  std::size_t at = 0;
  for (const std::size_t len : {std::size_t{100}, std::size_t{317}, n - 417}) {
    ASSERT_TRUE(lassq<double>(shared_device()->stream().get(), static_cast<int>(len),
                              d_x.data() + at, 1, d_chained.data(), d_chained.data() + 1)
                    .ok());
    at += len;
  }
  ASSERT_EQ(at, n);
  wwr::wwrStreamSynchronize(shared_device()->stream().get());
  const auto chained = from_device(d_chained, 2);

  double ref_scale = 1.0;
  double ref_sumsq = 0.0;
  elem<double>::ref(n, x.data(), 1, &ref_scale, &ref_sumsq);
  expect_close<double>(represented(chained[0], chained[1]), represented(ref_scale, ref_sumsq), n,
                       "chained");
}

} // namespace
} // namespace calaman
