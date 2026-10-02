// Oracle test for calaman.random_unitary -- fill n-by-n with standard-normal
// entries, then expand the Q of that Gaussian matrix (geqrf, then orgqr/ungqr).
// Like the orthogonalize suite, the oracle is NOT the reference LAPACK: the
// input is random and generated on the device, so there is no fixed matrix to
// compare against. The defining, convention-free property of the output is that
// its columns are orthonormal -- orthogonal for a real T, unitary for a complex
// one -- which is the single invariant:
//
//   Orthonormality: Q^H Q = I_n.
//
// evaluated on the host in std::complex after Q is copied back. A second check
// pins that the fill is actually random and seed-driven: two different seeds
// must produce substantially different matrices (||Q_a - Q_b||_F not tiny), so a
// routine that silently skipped the fill -- leaving a zero or stale buffer --
// would fail even though a zero buffer would also fail the orthonormality check.
//
// All four instantiated types are exercised, so BOTH the orgqr (real) and ungqr
// (complex) branches inside calaman.orthogonalize, and both the real and
// complex random_normal draws, are covered.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages work on the device,
// creates a solver handle, initializes device RNG states and runs geqrf +
// orgqr/ungqr, so `ctest -LE gpu` excludes it.

#include <gtest/gtest.h>

import std;

import wwr.solver;
import wwr.runtime_api;
import wwr.rand;
import wwr.complex;
import wwr.extension.memory_buffer;
import wwr.extension.init_state;
import calaman.random_unitary;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;
import calaman.test.shared.tolerance;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::shared_device;
using test::factorization_tol;

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

// --- Element-type plumbing: map each device element type T to a host scalar and
// convert at the buffer boundary, so the verification math is one std::complex
// code path for all four types. For a real T the imaginary part is simply zero.

template<typename T>
inline constexpr bool is_complex_v =
    std::is_same_v<T, wwr::wwrFloatComplex> || std::is_same_v<T, wwr::wwrDoubleComplex>;

// The real scalar underlying T: float for float/wwrFloatComplex, double otherwise.
template<typename T>
using real_t =
    std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, wwr::wwrFloatComplex>, float,
                       double>;

template<typename T>
std::complex<real_t<T>> to_std(T v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return {v, real_t<T>{0}};
  } else if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return {wwr::wwrCrealf(v), wwr::wwrCimagf(v)};
  } else {
    return {wwr::wwrCreal(v), wwr::wwrCimag(v)};
  }
}

// Frobenius norm of a host complex matrix: sqrt(sum |z|^2). std::norm(z) is |z|^2.
template<typename Real>
Real frobenius(const std::vector<std::complex<Real>> &v) {
  Real sum{};
  for (const auto z : v) {
    sum += std::norm(z);
  }
  return std::sqrt(sum);
}

// Run random_unitary once and return Q read back to the host as device elements.
// Also asserts the two solver infos are clean.
template<typename T>
std::vector<T> run_random_unitary(std::shared_ptr<DeviceHandle> handle, int n,
                                  unsigned long long seed) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

  wwr::wwrsolverDnHandle_t solver{};
  EXPECT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  // The caller owns the generator states: n*n of them, one per matrix element,
  // seeded here before the fill advances them inside random_unitary.
  DeviceBuffer<wwr::wwrrandState> states(nn, handle);
  wwr::extension::init_state(handle->stream().get(), nn, states.data(), seed);
  wwr::wwrStreamSynchronize(handle->stream().get());

  DeviceBuffer<T> d_q(nn, handle);

  int lwork = 0;
  EXPECT_EQ(random_unitary_bufferSize<T>(solver, n, &lwork), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_GT(lwork, 0);

  std::vector<T> work_init(static_cast<std::size_t>(lwork)); // value-initialised scratch
  auto d_work = to_device(handle, work_init);
  std::vector<int> info_init(1, 0);
  auto d_info_geqrf = to_device(handle, info_init);
  auto d_info_gqr = to_device(handle, info_init);

  const auto status = random_unitary<T>(solver, n, d_q.data(), d_work.data(), states.data(), lwork,
                                        d_info_geqrf.data(), d_info_gqr.data());
  EXPECT_TRUE(status.ok()) << "random_unitary status n=" << n << " seed=" << seed
                           << " status=" << status.name();

  const auto got = from_device(handle, d_q, nn);
  const auto info_geqrf = from_device(handle, d_info_geqrf, 1);
  const auto info_gqr = from_device(handle, d_info_gqr, 1);
  wwr::wwrsolverDnDestroy(solver);

  EXPECT_EQ(info_geqrf[0], 0) << "geqrf info n=" << n << " seed=" << seed;
  EXPECT_EQ(info_gqr[0], 0) << "orgqr/ungqr info n=" << n << " seed=" << seed;
  return got;
}

/// @brief random_unitary's output must have orthonormal columns: Q^H Q = I.
///
/// Generates an n-by-n random Q, copies it back, and checks the single
/// convention-free invariant ||Q^H Q - I||_F on the host. The bound drops the
/// ||A|| factor the orthogonalize suite keeps on its reprojection check: Q's
/// columns are unit-norm regardless of the (random) input's scale.
template<typename T>
void expect_unitary(int n, unsigned long long seed) {
  using Real = real_t<T>;
  using Cplx = std::complex<Real>;
  const int lda = n;
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

  auto handle = shared_device();
  const auto got = run_random_unitary<T>(handle, n, seed);

  std::vector<Cplx> q(nn);
  for (std::size_t i = 0; i < nn; ++i) {
    q[i] = to_std<T>(got[i]);
  }

  const Real ortho_tol =
      factorization_tol<Real>(Real{1}, static_cast<std::size_t>(n), static_cast<std::size_t>(n));

  // G = Q^H Q should be I_n. Accumulate ||G - I||_F.
  std::vector<Cplx> gram_minus_i(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      Cplx g{};
      for (int i = 0; i < n; ++i) {
        g += std::conj(q[static_cast<std::size_t>(j) * lda + i]) *
             q[static_cast<std::size_t>(k) * lda + i];
      }
      gram_minus_i[static_cast<std::size_t>(k) * n + j] = g - (j == k ? Cplx{1} : Cplx{0});
    }
  }
  EXPECT_LE(frobenius(gram_minus_i), ortho_tol)
      << "||Q^H Q - I|| n=" << n << " seed=" << seed;
}

/// @brief Two different seeds must produce substantially different matrices.
///
/// Guards against a routine that skips the fill: a zero or stale buffer is
/// seed-independent, so ||Q_a - Q_b||_F would be ~0. A genuine pair of random
/// unitaries differs by O(sqrt(n)); the threshold 1.0 is far above float noise
/// and far below that, so it catches a no-op fill without being flaky.
template<typename T>
void expect_seed_sensitive(int n, unsigned long long seed_a, unsigned long long seed_b) {
  using Real = real_t<T>;
  using Cplx = std::complex<Real>;
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

  auto handle = shared_device();
  const auto a = run_random_unitary<T>(handle, n, seed_a);
  const auto b = run_random_unitary<T>(handle, n, seed_b);

  std::vector<Cplx> diff(nn);
  for (std::size_t i = 0; i < nn; ++i) {
    diff[i] = to_std<T>(a[i]) - to_std<T>(b[i]);
  }
  EXPECT_GT(frobenius(diff), Real{1})
      << "two seeds gave near-identical matrices n=" << n
      << " seeds=" << seed_a << "," << seed_b;
}

// Orthonormality across all four types and a range of sizes.
TEST(RandomUnitaryOracleTests, UnitaryFloat) {
  expect_unitary<float>(4, 1);
  expect_unitary<float>(13, 2);
}
TEST(RandomUnitaryOracleTests, UnitaryDouble) {
  expect_unitary<double>(4, 3);
  expect_unitary<double>(13, 4);
}
TEST(RandomUnitaryOracleTests, UnitaryComplexFloat) {
  expect_unitary<wwr::wwrFloatComplex>(4, 5);
  expect_unitary<wwr::wwrFloatComplex>(13, 6);
}
TEST(RandomUnitaryOracleTests, UnitaryComplexDouble) {
  expect_unitary<wwr::wwrDoubleComplex>(4, 7);
  expect_unitary<wwr::wwrDoubleComplex>(13, 8);
}

// n == 1: the 1x1 "unitary" is a single unit-magnitude scalar; tau_size =
// align_up(1, 256) dominates the workspace layout.
TEST(RandomUnitaryOracleTests, SingleElement) {
  expect_unitary<double>(1, 9);
  expect_unitary<wwr::wwrDoubleComplex>(1, 10);
}

// Larger, past a single warp of columns, real and complex.
TEST(RandomUnitaryOracleTests, Larger) {
  expect_unitary<double>(64, 11);
  expect_unitary<wwr::wwrDoubleComplex>(48, 12);
}

// The fill is genuinely seed-driven, real and complex.
TEST(RandomUnitaryOracleTests, SeedSensitiveDouble) {
  expect_seed_sensitive<double>(16, 100, 101);
}
TEST(RandomUnitaryOracleTests, SeedSensitiveComplexDouble) {
  expect_seed_sensitive<wwr::wwrDoubleComplex>(16, 200, 201);
}

} // namespace
} // namespace calaman
