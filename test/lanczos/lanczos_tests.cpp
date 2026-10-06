// Suite for calaman.lanczos's host-checkable contract and its workspace:
//
//   * the argument-checking contract of lanczos_bufferSize / make_lanczos_slices
//     / lanczos_solve -- nev < 1, ncv < 2*nev + 1, ncv > n, a null out-pointer,
//     a null result or an empty matvec is rejected before any handle use
//     (host-only; the solve itself is lanczos_solve_tests.cpp);
//   * the :ritz selection -- positions per LanczosWhich, the nested-selection
//     property, residual estimates and convergence flags on a hand-built
//     snapshot, and each Ritz stage's argument checks (host-only);
//   * the workspace sizing and carving -- the size query equals the extent the
//     carve actually spans, and every region is non-null, 256-aligned, inside the
//     buffer and disjoint from the others (REQUIRES_GPU: syevd_bufferSize needs a
//     solver handle).

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.rand;
import wwr.extension.memory_buffer;
import calaman.lanczos;
import calaman.error_handling;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

constexpr int kInvalidValue = static_cast<int>(wwr::WWRBLAS_STATUS_INVALID_VALUE);

LanczosMatvecFn<double> identity_matvec() {
  return [](wwr::wwrStream_t, const double *, double *) -> Status {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  };
}

// ── argument checking (host-only; the handles are never dereferenced) ────────

TEST(LanczosArgCheckTests, ShapePredicate) {
  EXPECT_TRUE(lanczos_shape_ok(16, 1, 3));
  EXPECT_TRUE(lanczos_shape_ok(16, 4, 9));
  EXPECT_TRUE(lanczos_shape_ok(9, 4, 9));
  EXPECT_FALSE(lanczos_shape_ok(16, 0, 3));  // nev < 1
  EXPECT_FALSE(lanczos_shape_ok(16, 4, 8));  // ncv < 2*nev + 1
  EXPECT_FALSE(lanczos_shape_ok(8, 4, 9));   // ncv > n
  EXPECT_FALSE(lanczos_shape_ok(0, 1, 3));   // ncv > n, n empty
}

TEST(LanczosArgCheckTests, RestartKeepsBetweenNevAndNcvMinusOne) {
  EXPECT_EQ(lanczos_restart_keep(1, 3), 2);
  EXPECT_EQ(lanczos_restart_keep(3, 40), 21);
  for (int nev = 1; nev <= 20; ++nev) {
    for (int ncv = 2 * nev + 1; ncv <= 60; ++ncv) {
      const int k = lanczos_restart_keep(nev, ncv);
      EXPECT_GE(k, nev) << nev << ' ' << ncv;
      EXPECT_LE(k, ncv - 1) << nev << ' ' << ncv;
    }
  }
}

TEST(LanczosArgCheckTests, BufferSizeRejectsBadShape) {
  const wwr::wwrsolverDnHandle_t no_handle{}; // never touched on the rejection path
  std::size_t lwork = 0;

  EXPECT_EQ(lanczos_bufferSize<double>(no_handle, 16, 0, 3, &lwork).code, kInvalidValue);
  EXPECT_EQ(lanczos_bufferSize<double>(no_handle, 16, -1, 3, &lwork).code, kInvalidValue);
  EXPECT_EQ(lanczos_bufferSize<double>(no_handle, 16, 4, 8, &lwork).code, kInvalidValue);
  EXPECT_EQ(lanczos_bufferSize<double>(no_handle, 8, 4, 9, &lwork).code, kInvalidValue);
  EXPECT_EQ(lanczos_bufferSize<float>(no_handle, 16, 4, 17, &lwork).code, kInvalidValue);
}

TEST(LanczosArgCheckTests, MakeSlicesRejectsBadShape) {
  const wwr::wwrsolverDnHandle_t no_handle{};
  LanczosSlices<double> s;
  std::size_t lwork = 0;
  EXPECT_EQ(make_lanczos_slices<double>(no_handle, 16, 0, 3, nullptr, &s, &lwork).code,
            kInvalidValue);
  EXPECT_EQ(make_lanczos_slices<double>(no_handle, 16, 2, 4, nullptr, &s, &lwork).code,
            kInvalidValue);
  EXPECT_EQ(make_lanczos_slices<double>(no_handle, 4, 2, 5, nullptr, &s, &lwork).code,
            kInvalidValue);
}

TEST(LanczosArgCheckTests, BufferSizeRejectsNullOutPointer) {
  const wwr::wwrsolverDnHandle_t no_handle{};
  EXPECT_EQ(lanczos_bufferSize<double>(no_handle, 16, 2, 5, nullptr).code, kInvalidValue);
}

Status solve_with(int n, int nev, int ncv, const LanczosMatvecFn<double> &matvec,
                  LanczosResult<double> *result) {
  const LanczosSlices<double> s;
  return lanczos_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                               wwr::wwrStream_t{}, n, nev, ncv, LanczosWhich::smallest, s, matvec,
                               nullptr, result);
}

TEST(LanczosArgCheckTests, SolveRejectsBadArguments) {
  LanczosResult<double> result;
  EXPECT_EQ(solve_with(16, 0, 3, identity_matvec(), &result).code, kInvalidValue);
  EXPECT_EQ(solve_with(16, 4, 8, identity_matvec(), &result).code, kInvalidValue);
  EXPECT_EQ(solve_with(8, 4, 9, identity_matvec(), &result).code, kInvalidValue);
  EXPECT_EQ(solve_with(16, 2, 5, identity_matvec(), nullptr).code, kInvalidValue);
  EXPECT_EQ(solve_with(16, 2, 5, LanczosMatvecFn<double>{}, &result).code, kInvalidValue);
}

TEST(LanczosArgCheckTests, OptionDefaults) {
  const LanczosOptions<float> options;
  EXPECT_GT(options.tolerance, 0.0f);
  EXPECT_GT(options.max_restarts, 0);
  EXPECT_TRUE(options.fail_on_non_convergence);
  EXPECT_FALSE(options.verify_residuals);
  EXPECT_EQ(options.start_vector, nullptr);
}

// ── Ritz selection (host-only) ───────────────────────────────────────────────

TEST(LanczosSelectTests, PositionsPerWhich) {
  using V = std::vector<int>;
  EXPECT_EQ(lanczos_select(LanczosWhich::smallest, 10, 3), (V{0, 1, 2}));
  EXPECT_EQ(lanczos_select(LanczosWhich::largest, 10, 3), (V{7, 8, 9}));
  EXPECT_EQ(lanczos_select(LanczosWhich::both_ends, 10, 3), (V{0, 8, 9})); // 2 top, 1 bottom
  EXPECT_EQ(lanczos_select(LanczosWhich::both_ends, 10, 4), (V{0, 1, 8, 9}));
  EXPECT_EQ(lanczos_select(LanczosWhich::both_ends, 10, 1), (V{9}));
  EXPECT_EQ(lanczos_select(LanczosWhich::largest, 5, 5), (V{0, 1, 2, 3, 4}));
  EXPECT_TRUE(lanczos_select(LanczosWhich::smallest, 10, 0).empty());
  EXPECT_TRUE(lanczos_select(LanczosWhich::largest, 10, 11).empty());
}

TEST(LanczosSelectTests, LargerSelectionContainsSmaller) {
  for (const LanczosWhich which :
       {LanczosWhich::smallest, LanczosWhich::largest, LanczosWhich::both_ends}) {
    for (int nev = 1; nev <= 12; ++nev) {
      const std::vector<int> wanted = lanczos_select(which, 12, nev);
      for (int k = nev; k <= 12; ++k) {
        const std::vector<int> kept = lanczos_select(which, 12, k);
        EXPECT_TRUE(std::ranges::includes(kept, wanted)) << nev << " in " << k;
      }
    }
  }
}

TEST(LanczosSelectTests, ResidualEstimatesAndConvergence) {
  LanczosRitz<double> ritz;
  ritz.theta = {-3.0, -1.0, 0.0, 2.0, 5.0};
  ritz.s_last_row = {1e-9, 0.5, -0.2, -1e-12, -0.3};
  ritz.beta_m = -2.0;
  ritz.t_norm = 5.0;

  const auto sel = lanczos_ritz_select(ritz, LanczosWhich::both_ends, 4, 1e-6);
  EXPECT_EQ(sel.index, (std::vector<int>{0, 1, 3, 4}));
  EXPECT_EQ(sel.values, (std::vector<double>{-3.0, -1.0, 2.0, 5.0}));
  EXPECT_EQ(sel.residuals, (std::vector<double>{2e-9, 1.0, 2e-12, 0.6}));
  // threshold 1e-6 * max(|theta|, 5) = 5e-6 for every pair here.
  EXPECT_EQ(sel.converged, (std::vector<bool>{true, false, true, false}));
  EXPECT_EQ(sel.converged_count, 2);
  EXPECT_FALSE(sel.all_converged());

  const auto none = lanczos_ritz_select(ritz, LanczosWhich::smallest, 0, 1e-6);
  EXPECT_TRUE(none.index.empty());
  EXPECT_TRUE(none.all_converged()); // vacuously
}

TEST(LanczosSelectTests, RitzStagesRejectBadArguments) {
  const LanczosSlices<double> s; // never dereferenced on the rejection path
  EXPECT_EQ(lanczos_ritz_extract<double>(wwr::wwrsolverDnHandle_t{}, wwr::wwrStream_t{}, 5, s,
                                         nullptr)
                .code,
            kInvalidValue);
  LanczosRitz<double> ritz;
  EXPECT_EQ(
      lanczos_ritz_extract<double>(wwr::wwrsolverDnHandle_t{}, wwr::wwrStream_t{}, 0, s, &ritz)
          .code,
      kInvalidValue);
  const wwr::wwrStream_t no_stream{};
  EXPECT_EQ(lanczos_ritz_compact<double>(no_stream, 5, {1, 1}, s).code, kInvalidValue);
  EXPECT_EQ(lanczos_ritz_compact<double>(no_stream, 5, {2, 1}, s).code, kInvalidValue);
  EXPECT_EQ(lanczos_ritz_compact<double>(no_stream, 5, {-1}, s).code, kInvalidValue);
  EXPECT_EQ(lanczos_ritz_compact<double>(no_stream, 5, {5}, s).code, kInvalidValue);
  EXPECT_TRUE(lanczos_ritz_compact<double>(no_stream, 5, {0, 1, 2}, s).ok()); // already leading
  double x = 0.0;
  const wwr::wwrblasHandle_t no_blas{};
  EXPECT_EQ(lanczos_ritz_vectors<double>(no_blas, 8, 5, 2, s, nullptr, 8).code, kInvalidValue);
  EXPECT_EQ(lanczos_ritz_vectors<double>(no_blas, 8, 5, 2, s, &x, 7).code, kInvalidValue);
  EXPECT_EQ(lanczos_ritz_vectors<double>(no_blas, 8, 5, 0, s, &x, 8).code, kInvalidValue);
  EXPECT_EQ(lanczos_ritz_vectors<double>(no_blas, 8, 5, 6, s, &x, 8).code, kInvalidValue);
}

// ── sizing and carving (REQUIRES_GPU) ────────────────────────────────────────

struct SolverHandle {
  wwr::wwrsolverDnHandle_t h{};
  SolverHandle() { EXPECT_EQ(wwr::wwrsolverDnCreate(&h), wwr::WWRSOLVER_STATUS_SUCCESS); }
  ~SolverHandle() { wwr::wwrsolverDnDestroy(h); }
  SolverHandle(const SolverHandle &) = delete;
  SolverHandle &operator=(const SolverHandle &) = delete;
};

template<typename T>
std::size_t size_for(wwr::wwrsolverDnHandle_t solver, int n, int nev, int ncv) {
  std::size_t lwork = 0;
  const Status st = lanczos_bufferSize<T>(solver, n, nev, ncv, &lwork);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  return lwork;
}

template<typename T>
void check_sizing(wwr::wwrsolverDnHandle_t solver) {
  constexpr int n = 64;
  constexpr int nev = 3;
  const std::size_t small = size_for<T>(solver, n, nev, 2 * nev + 1);
  EXPECT_GT(small, 0u);
  const std::size_t large = size_for<T>(solver, n, nev, 4 * nev);
  EXPECT_GT(large, small);
  const std::size_t taller = size_for<T>(solver, 2 * n, nev, 4 * nev);
  EXPECT_GT(taller, large);
}

TEST(LanczosBufferSizeTests, SizingIsPositiveAndMonotone) {
  SolverHandle solver;
  check_sizing<float>(solver.h);
  check_sizing<double>(solver.h);
}

constexpr std::size_t aligned(std::size_t bytes) { return (bytes + 255u) / 256u * 256u; }

template<typename T>
void check_carve(int n, int nev, int ncv) {
  SCOPED_TRACE(::testing::Message() << "n=" << n << " nev=" << nev << " ncv=" << ncv
                                    << " sizeof(T)=" << sizeof(T));
  auto handle = shared_device();
  SolverHandle solver;

  const std::size_t queried = size_for<T>(solver.h, n, nev, ncv);
  ASSERT_GT(queried, 0u);

  DeviceBuffer<std::byte> work(queried, handle);
  LanczosSlices<T> s;
  std::size_t carved = 0;
  ASSERT_TRUE(make_lanczos_slices<T>(solver.h, n, nev, ncv, work.data(), &s, &carved).ok());
  EXPECT_EQ(carved, queried);

  const auto base = reinterpret_cast<std::uintptr_t>(work.data());
  const auto nz = static_cast<std::size_t>(n);
  const auto mz = static_cast<std::size_t>(ncv);
  // device::LanczosStatus is a bridge-header type, not exported by name.
  using StatusBlock = std::remove_pointer_t<decltype(s.status)>;

  struct Region {
    const void *p;
    std::size_t bytes;
  };
  std::vector<Region> regions = {
      {s.v, sizeof(T) * nz * (mz + 1)},
      {s.keep, sizeof(T) * nz * static_cast<std::size_t>(lanczos_restart_keep(nev, ncv))},
      {s.t, sizeof(T) * mz * mz},
      {s.s, sizeof(T) * mz * mz},
      {s.theta, sizeof(T) * mz},
      {s.coeffs, sizeof(T) * mz},
      {s.alpha, sizeof(T) * mz},
      {s.beta, sizeof(T) * mz},
      {s.rng, sizeof(wwr::wwrrandState) * nz},
      {s.status, sizeof(StatusBlock)},
      {s.eig_scratch, sizeof(T) * static_cast<std::size_t>(s.lwork_eig)},
  };

  for (const Region &r : regions) {
    const auto a = reinterpret_cast<std::uintptr_t>(r.p);
    EXPECT_NE(r.p, nullptr);
    EXPECT_EQ(a % 256u, 0u);
    EXPECT_GE(a, base);
    EXPECT_LE(a + r.bytes, base + queried);
  }
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(s.eig_info),
            reinterpret_cast<std::uintptr_t>(s.status) + offsetof(StatusBlock, eig_info));

  // The size query is the carved extent: the scratch region is last, and its
  // aligned end is exactly the end of the buffer the query asked for.
  const auto scratch_end = reinterpret_cast<std::uintptr_t>(s.eig_scratch) +
                           aligned(sizeof(T) * static_cast<std::size_t>(s.lwork_eig));
  EXPECT_EQ(scratch_end, base + queried);

  // Disjoint, in carve order.
  for (std::size_t i = 0; i + 1 < regions.size(); ++i) {
    const auto end = reinterpret_cast<std::uintptr_t>(regions[i].p) + regions[i].bytes;
    EXPECT_LE(end, reinterpret_cast<std::uintptr_t>(regions[i + 1].p)) << "region " << i;
  }
}

TEST(LanczosBufferSizeTests, CarveMatchesQueryFloat) {
  check_carve<float>(48, 3, 7);
  check_carve<float>(100, 4, 20);
}

TEST(LanczosBufferSizeTests, CarveMatchesQueryDouble) {
  check_carve<double>(48, 3, 7);
  check_carve<double>(9, 4, 9); // ncv == n
}

} // namespace
} // namespace calaman
