// Suite for calaman.lanczos's host-checkable contract and its workspace:
//
//   * the argument-checking contract of lanczos_bufferSize / make_lanczos_slices
//     / lanczos_solve -- nev < 1, ncv < 2*nev + 1, ncv > n, a null out-pointer,
//     a null result or a null matvec is rejected before any handle use
//     (host-only; the solve itself is lanczos_solve_tests.cpp);
//   * the :ritz selection -- residual estimates and convergence flags on a
//     hand-built snapshot, and each Ritz stage's argument checks (host-only;
//     the positions and nesting are calaman.ritz's, test/ritz/);
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

auto identity_matvec() {
  return [](wwr::wwrStream_t, const double *, double *) -> Status {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  };
}

using MatvecPtr = Status (*)(wwr::wwrStream_t, const double *, double *);
using MatvecFunction = std::function<Status(wwr::wwrStream_t, const double *, double *)>;

// The concept accepts lambdas, function pointers and std::function of the right
// shape, and rejects the wrong arity or element type.
static_assert(lanczos_matvec<decltype(identity_matvec()), double>);
static_assert(lanczos_matvec<MatvecPtr, double>);
static_assert(lanczos_matvec<MatvecFunction, double>);
static_assert(!lanczos_matvec<decltype(identity_matvec()), float>);
static_assert(!lanczos_matvec<Status (*)(wwr::wwrStream_t, const double *), double>);

/// A linear_operator model with apply only, and no call operator.
struct IdentityOperator {
  Status apply(wwr::wwrStream_t, int, const double *, double *) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

// The adapter makes any lanczos_matvec a linear_operator; the two concepts are
// disjoint for these models, so overload resolution picks exactly one path.
static_assert(linear_operator<LanczosMatvecOperator<double, MatvecPtr>, double>);
static_assert(linear_operator<IdentityOperator, double>);
static_assert(!lanczos_matvec<IdentityOperator, double>);
static_assert(!linear_operator<decltype(identity_matvec()), double>);

/// lanczos_solve accepts @p Op as an lvalue (a const one for a matvec).
template<class Op>
concept solve_accepts = requires(Op &op, const LanczosSlices<double> &s, double *out,
                                 LanczosInfo *info) {
  lanczos_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{}, wwr::wwrStream_t{}, 16,
                        2, 5, LanczosWhich::smallest, s, op, out, out, info);
};
static_assert(solve_accepts<IdentityOperator>);
static_assert(solve_accepts<const decltype(identity_matvec())>);
static_assert(solve_accepts<decltype(identity_matvec())>);
static_assert(solve_accepts<const MatvecFunction>);
static_assert(!solve_accepts<const IdentityOperator>); // apply is non-const
static_assert(!solve_accepts<int>);

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

double g_fake_values[4]; // a never-dereferenced stand-in for the device output

Status solve_with(int n, int nev, int ncv, const lanczos_matvec<double> auto &matvec,
                  LanczosInfo *info, double *values = g_fake_values) {
  const LanczosSlices<double> s;
  return lanczos_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                               wwr::wwrStream_t{}, n, nev, ncv, LanczosWhich::smallest, s, matvec,
                               values, nullptr, info);
}

TEST(LanczosArgCheckTests, SolveRejectsBadArguments) {
  LanczosInfo info;
  EXPECT_EQ(solve_with(16, 0, 3, identity_matvec(), &info).code, kInvalidValue);
  EXPECT_EQ(solve_with(16, 4, 8, identity_matvec(), &info).code, kInvalidValue);
  EXPECT_EQ(solve_with(8, 4, 9, identity_matvec(), &info).code, kInvalidValue);
  EXPECT_EQ(solve_with(16, 2, 5, identity_matvec(), nullptr).code, kInvalidValue);
  EXPECT_EQ(solve_with(16, 2, 5, identity_matvec(), &info, nullptr).code, kInvalidValue);
}

TEST(LanczosArgCheckTests, OperatorSolveRejectsBadArguments) {
  const LanczosSlices<double> s;
  IdentityOperator op;
  LanczosInfo info;
  const auto solve = [&](int n, int nev, int ncv, LanczosInfo *out) {
    return lanczos_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                 wwr::wwrStream_t{}, n, nev, ncv, LanczosWhich::smallest, s, op,
                                 g_fake_values, nullptr, out);
  };
  EXPECT_EQ(solve(16, 0, 3, &info).code, kInvalidValue);
  EXPECT_EQ(solve(16, 4, 8, &info).code, kInvalidValue);
  EXPECT_EQ(solve(8, 4, 9, &info).code, kInvalidValue);
  EXPECT_EQ(solve(16, 2, 5, nullptr).code, kInvalidValue);
}

TEST(LanczosArgCheckTests, OptionDefaults) {
  const LanczosOptions<float> options;
  EXPECT_GT(options.tolerance, 0.0f);
  EXPECT_GT(options.max_iterations, 0);
  EXPECT_FALSE(options.verify_residuals);
  EXPECT_EQ(options.start_vector, nullptr);
}

// ── Ritz selection (host-only) ───────────────────────────────────────────────

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
