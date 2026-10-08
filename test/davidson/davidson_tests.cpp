// Suite for calaman.davidson. Three things are checked:
//
//   * the argument-checking contract of davidson_bufferSize /
//     make_davidson_slices / davidson_solve -- a bad shape, a null out-pointer, a
//     bad guess_count, or a (not-yet-implemented) metric callback is rejected
//     before any handle use (host-only);
//   * the workspace sizing and carving -- a valid shape sizes to a positive byte
//     count that grows with the subspace and with the metric path, and carving a
//     real buffer hands back non-null, 256-aligned, in-range pointers, with the
//     metric regions present only when sized with_metric (REQUIRES_GPU);
//   * the Euclidean solve against the reference LAPACK -- on a diagonally
//     dominant symmetric operator driven through a gemv callback with a diagonal
//     preconditioner, davidson_solve converges the lowest n_roots eigenvalues to
//     LAPACKE_?syevd's, over float and double, and a linear_operator model takes
//     the sigma lambda's exact path (REQUIRES_GPU);
//   * which operator shapes davidson_solve accepts (static_asserts);
//   * the non-converged stops -- a one-iteration budget (MaxIterations) and a
//     zero preconditioner (Stagnated) both return success with the first
//     iteration's Ritz values written (REQUIRES_GPU).
//
// Guarded on calaman::lapack_reference (see CMakeLists.txt): the reference suite
// is the solver's natural oracle, and bundling the rest in the same binary keeps
// one target. The host arithmetic is done in double regardless of T.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

#include "shared/expect_converged.h"

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.wrappers.common;
import wwr.wrappers.blas;
import wwr.extension.memory_buffer;
import calaman.davidson;
import calaman.error_handling;
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

constexpr int kInvalidValue = static_cast<int>(wwr::WWRBLAS_STATUS_INVALID_VALUE);

// Do-nothing callbacks for the rejection paths, which never invoke them.
constexpr auto kNoopBlock = [](wwr::wwrStream_t, int, const double *, double *) -> Status {
  return wwr::WWRBLAS_STATUS_SUCCESS;
};
constexpr auto kNoopPrecondition = [](wwr::wwrStream_t, int, const double *, const double *,
                                      double *) -> Status { return wwr::WWRBLAS_STATUS_SUCCESS; };

// The callback concepts accept the shapes davidson_solve calls and reject others.
static_assert(davidson_sigma<decltype(kNoopBlock), double>);
static_assert(davidson_metric<decltype(kNoopBlock), double>);
static_assert(davidson_metric<DavidsonNoMetric, double>);
static_assert(davidson_preconditioner<decltype(kNoopPrecondition), double>);
static_assert(!davidson_sigma<decltype(kNoopPrecondition), double>);
static_assert(!davidson_sigma<decltype(kNoopBlock), float>);

using SigmaPtr = Status (*)(wwr::wwrStream_t, int, const double *, double *);
using SigmaFunction = std::function<Status(wwr::wwrStream_t, int, const double *, double *)>;

/// A linear_operator model with apply only, and no call operator.
struct NoopOperator {
  Status apply(wwr::wwrStream_t, int, const double *, double *) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

/// Models both concepts: overload resolution must pick the operator path.
struct NoopBoth : NoopOperator {
  Status operator()(wwr::wwrStream_t, int, const double *, double *) const {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }
};

// The adapter makes any davidson_sigma a linear_operator; the two concepts are
// disjoint for the plain models, so overload resolution picks exactly one path.
static_assert(linear_operator<DavidsonSigmaOperator<double, decltype(kNoopBlock)>, double>);
static_assert(linear_operator<DavidsonSigmaOperator<double, SigmaPtr>, double>);
static_assert(linear_operator<NoopOperator, double>);
static_assert(!davidson_sigma<NoopOperator, double>);
static_assert(!linear_operator<decltype(kNoopBlock), double>);
static_assert(linear_operator<NoopBoth, double> && davidson_sigma<NoopBoth, double>);

/// davidson_solve accepts @p Op as an lvalue (a const one for a sigma callable).
template<class Op>
concept solve_accepts =
    requires(Op &op, const DavidsonSlices<double> &s, double *out, DavidsonInfo<double> *info) {
      davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{}, wwr::wwrStream_t{},
                             16, 4, 8, out, 4, s, op, kNoopPrecondition, out, out, info);
    };
static_assert(solve_accepts<NoopOperator>);
static_assert(solve_accepts<NoopBoth>);
static_assert(solve_accepts<decltype(kNoopBlock)>); // constexpr: already const
static_assert(solve_accepts<std::remove_const_t<decltype(kNoopBlock)>>);
static_assert(solve_accepts<SigmaPtr>);
static_assert(solve_accepts<const SigmaFunction>);
static_assert(!solve_accepts<const NoopOperator>); // apply is non-const
static_assert(!solve_accepts<int>);

// ── argument checking (host-only; the handle is never dereferenced) ──────────

TEST(DavidsonArgCheckTests, RejectsBadShape) {
  const wwr::wwrsolverDnHandle_t no_handle{}; // never touched on the rejection path
  std::size_t lwork = 0;

  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 0, 1, 2, false, &lwork).code, kInvalidValue);
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 4, 0, 2, false, &lwork).code, kInvalidValue);
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 4, 5, 8, false, &lwork).code, kInvalidValue);
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 16, 4, 7, false, &lwork).code, kInvalidValue);
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 8, 2, 16, false, &lwork).code, kInvalidValue);
}

TEST(DavidsonArgCheckTests, RejectsNullOutPointer) {
  const wwr::wwrsolverDnHandle_t no_handle{};
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 16, 4, 8, false, nullptr).code, kInvalidValue);
}

// Never-dereferenced stand-ins for the solve's device out-pointers.
double g_fake_values[4];
double g_fake_vectors[64];

TEST(DavidsonArgCheckTests, RejectsBadGuessCount) {
  // guess_count outside [n_roots, max_subspace] is rejected before any handle use.
  DavidsonSlices<double> s;
  DavidsonInfo<double> info;
  const auto solve = [&](int guess_count) {
    return davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                  wwr::wwrStream_t{}, 16, 4, 8, nullptr, guess_count, s, kNoopBlock,
                                  kNoopPrecondition, g_fake_values, g_fake_vectors, &info);
  };
  EXPECT_EQ(solve(3).code, kInvalidValue); // below n_roots
  EXPECT_EQ(solve(9).code, kInvalidValue); // above max_subspace
}

TEST(DavidsonArgCheckTests, RejectsNullSolveOutPointer) {
  DavidsonSlices<double> s;
  DavidsonInfo<double> info;
  const auto solve = [&](double *values, double *vectors, DavidsonInfo<double> *out) {
    return davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                  wwr::wwrStream_t{}, 16, 4, 8, nullptr, 4, s, kNoopBlock,
                                  kNoopPrecondition, values, vectors, out);
  };
  EXPECT_EQ(solve(nullptr, g_fake_vectors, &info).code, kInvalidValue);
  EXPECT_EQ(solve(g_fake_values, nullptr, &info).code, kInvalidValue);
  EXPECT_EQ(solve(g_fake_values, g_fake_vectors, nullptr).code, kInvalidValue);
}

TEST(DavidsonArgCheckTests, RejectsMetricWithoutMetricWorkspace) {
  // A metric selects the generalized path, which needs a workspace
  // sized with_metric; a default (Euclidean) DavidsonSlices has null metric
  // regions, so it is rejected before any handle use rather than silently
  // solving the wrong (Euclidean) problem.
  DavidsonSlices<double> s; // all-null: not sized with_metric
  DavidsonInfo<double> info;
  const Status st = davidson_solve<double>(
      wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{}, wwr::wwrStream_t{}, 16, 4, 8, nullptr, 4,
      s, kNoopBlock, kNoopPrecondition, g_fake_values, g_fake_vectors, &info, {}, kNoopBlock);
  EXPECT_EQ(st.code, kInvalidValue);
}

TEST(DavidsonArgCheckTests, OperatorSolveRejectsBadArguments) {
  // The linear_operator path runs the same checks before any handle use.
  DavidsonSlices<double> s;
  NoopOperator op;
  DavidsonInfo<double> info;
  const auto solve = [&](int guess_count, DavidsonInfo<double> *out) {
    return davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                  wwr::wwrStream_t{}, 16, 4, 8, nullptr, guess_count, s, op,
                                  kNoopPrecondition, g_fake_values, g_fake_vectors, out);
  };
  EXPECT_EQ(solve(3, &info).code, kInvalidValue);
  EXPECT_EQ(solve(9, &info).code, kInvalidValue);
  EXPECT_EQ(solve(4, nullptr).code, kInvalidValue);
  EXPECT_EQ(davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                   wwr::wwrStream_t{}, 16, 4, 8, nullptr, 4, s, op,
                                   kNoopPrecondition, g_fake_values, g_fake_vectors, &info, {},
                                   kNoopBlock)
                .code,
            kInvalidValue); // a metric against a workspace not sized with_metric
}

// ── shared device plumbing ───────────────────────────────────────────────────

struct Handles {
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
};

Handles make_handles(std::shared_ptr<DeviceHandle> handle) {
  Handles h;
  EXPECT_EQ(wwr::wwrblasCreate(&h.blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(h.blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnCreate(&h.solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnSetStream(h.solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  return h;
}

void destroy_handles(Handles &h) {
  wwr::wwrblasDestroy(h.blas);
  wwr::wwrsolverDnDestroy(h.solver);
}

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

// ── sizing and carving (REQUIRES_GPU) ────────────────────────────────────────

template<typename T>
std::size_t size_for(wwr::wwrsolverDnHandle_t solver, int n, int n_roots, int max_subspace,
                     bool with_metric) {
  std::size_t lwork = 0;
  const Status st = davidson_bufferSize<T>(solver, n, n_roots, max_subspace, with_metric, &lwork);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  return lwork;
}

template<typename T>
void check_sizing(wwr::wwrsolverDnHandle_t solver) {
  constexpr int n = 64;
  constexpr int n_roots = 4;

  const std::size_t small = size_for<T>(solver, n, n_roots, 2 * n_roots, false);
  EXPECT_GT(small, 0u);
  const std::size_t large = size_for<T>(solver, n, n_roots, 4 * n_roots, false);
  EXPECT_GE(large, small);
  const std::size_t euclid = size_for<T>(solver, n, n_roots, 4 * n_roots, false);
  const std::size_t metric = size_for<T>(solver, n, n_roots, 4 * n_roots, true);
  EXPECT_GT(metric, euclid);
}

TEST(DavidsonBufferSizeTests, SizingIsPositiveAndMonotone) {
  auto handle = shared_device();
  Handles h = make_handles(handle);
  check_sizing<float>(h.solver);
  check_sizing<double>(h.solver);
  destroy_handles(h);
}

template<typename T>
void check_carve(std::shared_ptr<DeviceHandle> handle, wwr::wwrsolverDnHandle_t solver,
                 bool with_metric) {
  constexpr int n = 48;
  constexpr int n_roots = 3;
  constexpr int max_subspace = 4 * n_roots;

  std::size_t lwork = 0;
  ASSERT_TRUE(davidson_bufferSize<T>(solver, n, n_roots, max_subspace, with_metric, &lwork).ok());
  ASSERT_GT(lwork, 0u);

  DeviceBuffer<std::byte> work(lwork, handle);
  DavidsonSlices<T> s;
  ASSERT_TRUE(make_davidson_slices<T>(solver, n, n_roots, max_subspace, with_metric, work.data(),
                                      &s, &lwork)
                  .ok());

  const auto base = reinterpret_cast<std::uintptr_t>(work.data());
  const auto in_range = [&](const void *p) {
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    return a >= base && a < base + lwork && (a % 256u == 0u);
  };

  for (const void *p : {static_cast<const void *>(s.v), static_cast<const void *>(s.av),
                        static_cast<const void *>(s.h), static_cast<const void *>(s.ritz),
                        static_cast<const void *>(s.ritz_av),
                        static_cast<const void *>(s.residual),
                        static_cast<const void *>(s.correction), static_cast<const void *>(s.proj),
                        static_cast<const void *>(s.info),
                        static_cast<const void *>(s.eig_scratch)}) {
    EXPECT_NE(p, nullptr);
    EXPECT_TRUE(in_range(p));
  }
  EXPECT_GT(s.lwork_eig, 0);

  if (with_metric) {
    for (const void *p : {static_cast<const void *>(s.mv), static_cast<const void *>(s.s_sub),
                          static_cast<const void *>(s.metric_scratch)}) {
      EXPECT_NE(p, nullptr);
      EXPECT_TRUE(in_range(p));
    }
  } else {
    EXPECT_EQ(s.mv, nullptr);
    EXPECT_EQ(s.s_sub, nullptr);
    EXPECT_EQ(s.metric_scratch, nullptr);
  }
}

TEST(DavidsonBufferSizeTests, CarveEuclidean) {
  auto handle = shared_device();
  Handles h = make_handles(handle);
  check_carve<float>(handle, h.solver, false);
  check_carve<double>(handle, h.solver, false);
  destroy_handles(h);
}

TEST(DavidsonBufferSizeTests, CarveMetric) {
  auto handle = shared_device();
  Handles h = make_handles(handle);
  check_carve<float>(handle, h.solver, true);
  check_carve<double>(handle, h.solver, true);
  destroy_handles(h);
}

// ── Euclidean solve vs the reference LAPACK (REQUIRES_GPU) ────────────────────

// A = diag(1..n) + eps*(R + R^T): symmetric, diagonally dominant so its lowest
// eigenvalues are well separated (near 1, 2, 3, ...), which the diagonal
// preconditioner resolves quickly. Full (both triangles), column-major.
template<typename T>
std::vector<T> diag_plus_perturbation(int n, double eps, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      const double v = (i == j) ? static_cast<double>(j + 1) : eps * dist(rng);
      a[static_cast<std::size_t>(j) * n + i] = static_cast<T>(v);
      a[static_cast<std::size_t>(i) * n + j] = static_cast<T>(v);
    }
  }
  return a;
}

lapack_int call_syevd(int n, float *a, float *w) {
  return LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a, n, w);
}
lapack_int call_syevd(int n, double *a, double *w) {
  return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', n, a, n, w);
}

template<typename T>
std::vector<T> reference_eigenvalues(int n, std::vector<T> a) {
  std::vector<T> w(n);
  EXPECT_EQ(call_syevd(n, a.data(), w.data()), 0);
  return w; // ascending
}

/// ||A||_2 of a symmetric matrix from its ascending spectrum: the larger end.
template<typename T>
double spectral_norm(const std::vector<T> &ascending) {
  return std::max(std::abs(static_cast<double>(ascending.front())),
                  std::abs(static_cast<double>(ascending.back())));
}

/// Host copy of a device buffer of @p count elements.
template<typename T>
std::vector<T> to_host(std::shared_ptr<DeviceHandle> handle, const T *device, std::size_t count) {
  std::vector<T> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, sizeof(T) * count, wwr::wwrMemcpyDeviceToHost,
                                handle->stream().get()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);
  return host;
}

/// A (n x n, device) as a linear_operator model: one gemm per apply, counting
/// its applies and noting any made outside HOST pointer mode.
template<typename T>
struct GemmOperator {
  wwr::wwrblasHandle_t blas;
  int n;
  const T *d_a;
  int applies = 0;
  bool host_mode = true;

  Status apply(wwr::wwrStream_t, int k, const T *X, T *Y) {
    ++applies;
    wwr::wwrblasPointerMode_t mode{};
    const Status got = wwr::wwrblasGetPointerMode(blas, &mode);
    host_mode = host_mode && got.ok() && mode == wwr::WWRBLAS_POINTER_MODE_HOST;
    const T one{1};
    const T zero{0};
    return wwr::gemm<T, int>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, k, n, &one, d_a, n, X,
                             n, &zero, Y, n);
  }
};
static_assert(linear_operator<GemmOperator<float>, float>);
static_assert(linear_operator<GemmOperator<double>, double>);

/// The Euclidean problem every solve below runs: A = diag_plus_perturbation,
/// unit-vector guesses, a gemm sigma and the diagonal preconditioner (or, with
/// zero_correction, one that returns a zero block, which forces stagnation).
template<typename T>
struct EuclideanProblem {
  static constexpr int n = 48;
  static constexpr int n_roots = 4;
  static constexpr int max_subspace = 20;

  std::shared_ptr<DeviceHandle> handle = shared_device();
  Handles h = make_handles(handle);
  std::vector<T> a = diag_plus_perturbation<T>(n, 0.05, 20261004u);
  DeviceBuffer<T> d_a = to_device(handle, a);
  DeviceBuffer<T> d_guess = to_device(handle, unit_guess());
  DeviceBuffer<T> d_vals{static_cast<std::size_t>(n_roots), handle};
  DeviceBuffer<T> d_vecs{static_cast<std::size_t>(n) * n_roots, handle};
  DeviceBuffer<std::byte> work{workspace_bytes(), handle};
  DavidsonSlices<T> s;

  EuclideanProblem() {
    std::size_t lwork = 0;
    EXPECT_TRUE(
        make_davidson_slices<T>(h.solver, n, n_roots, max_subspace, false, work.data(), &s, &lwork)
            .ok());
  }
  ~EuclideanProblem() { destroy_handles(h); }
  EuclideanProblem(const EuclideanProblem &) = delete;
  EuclideanProblem &operator=(const EuclideanProblem &) = delete;

  static std::vector<T> unit_guess() {
    std::vector<T> g(static_cast<std::size_t>(n) * n_roots, T{0});
    for (int c = 0; c < n_roots; ++c) {
      g[static_cast<std::size_t>(c) * n + c] = T{1}; // orthonormal unit vectors
    }
    return g;
  }

  std::size_t workspace_bytes() const {
    std::size_t lwork = 0;
    EXPECT_TRUE(davidson_bufferSize<T>(h.solver, n, n_roots, max_subspace, false, &lwork).ok());
    return lwork;
  }

  Status solve(const DavidsonOptions<T> &options, DavidsonInfo<T> *info,
               bool zero_correction = false) {
    // sigma(B) = A B via a single gemm; the solver holds the handle in host
    // pointer mode for the whole call, so host scalar pointers are correct here.
    const auto sigma = [&](wwr::wwrStream_t, int block, const T *b, T *out) -> Status {
      const T one{1};
      const T zero{0};
      return wwr::gemm<T, int>(h.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, block, n, &one,
                               d_a.data(), n, b, n, &zero, out, n);
    };
    return davidson_solve<T>(h.blas, h.solver, handle->stream().get(), n, n_roots, max_subspace,
                             d_guess.data(), n_roots, s, sigma, precondition(zero_correction),
                             d_vals.data(), d_vecs.data(), info, options);
  }

  /// The same solve with @p op, a linear_operator model, in place of the lambda.
  Status solve(const DavidsonOptions<T> &options, DavidsonInfo<T> *info, GemmOperator<T> &op) {
    return davidson_solve<T>(h.blas, h.solver, handle->stream().get(), n, n_roots, max_subspace,
                             d_guess.data(), n_roots, s, op, precondition(false), d_vals.data(),
                             d_vecs.data(), info, options);
  }

  /// Diagonal Davidson preconditioner correction = residual / (theta - diag),
  /// computed on the host (no kernel needed for a test), guarding the near-zero
  /// denominator when a Ritz value approaches its own diagonal entry.
  auto precondition(bool zero_correction) {
    return [this, zero_correction](wwr::wwrStream_t stream, int roots, const T *theta,
                                   const T *residual, T *correction) -> Status {
      const std::size_t cnt = static_cast<std::size_t>(n) * roots;
      if (zero_correction) {
        return wwr::wwrMemsetAsync(correction, 0, sizeof(T) * cnt, stream);
      }
      std::vector<T> rbuf(cnt);
      std::vector<T> cbuf(cnt);
      if (const Status st{wwr::wwrMemcpyAsync(rbuf.data(), residual, sizeof(T) * cnt,
                                              wwr::wwrMemcpyDeviceToHost, stream)};
          !st.ok()) {
        return st;
      }
      wwr::wwrStreamSynchronize(stream);
      for (int i = 0; i < roots; ++i) {
        for (int j = 0; j < n; ++j) {
          T denom = theta[i] - a[static_cast<std::size_t>(j) * n + j];
          if (std::abs(denom) < T{1e-3}) {
            denom = denom < T{0} ? T{-1e-3} : T{1e-3};
          }
          cbuf[static_cast<std::size_t>(i) * n + j] =
              rbuf[static_cast<std::size_t>(i) * n + j] / denom;
        }
      }
      const Status st{wwr::wwrMemcpyAsync(correction, cbuf.data(), sizeof(T) * cnt,
                                          wwr::wwrMemcpyHostToDevice, stream)};
      wwr::wwrStreamSynchronize(stream);
      return st;
    };
  }

  std::vector<T> eigenvalues() { return to_host(handle, d_vals.data(), n_roots); }

  /// Ritz values of the first iteration: the unit-vector guess spans the first
  /// n_roots coordinates, so they are the eigenvalues of A's leading block.
  std::vector<T> leading_block_eigenvalues() const {
    std::vector<T> block(static_cast<std::size_t>(n_roots) * n_roots);
    for (int j = 0; j < n_roots; ++j) {
      for (int i = 0; i < n_roots; ++i) {
        block[static_cast<std::size_t>(j) * n_roots + i] = a[static_cast<std::size_t>(j) * n + i];
      }
    }
    return reference_eigenvalues<T>(n_roots, block);
  }
};

template<typename T>
void check_reference() {
  const bool is_float = std::is_same_v<T, float>;
  EuclideanProblem<T> p;
  const auto ref = reference_eigenvalues<T>(p.n, p.a);
  const double eig_tol = is_float ? 5e-3 : 1e-6;

  DavidsonOptions<T> options;
  options.residual_tolerance = is_float ? T{1e-4} : T{1e-8};
  options.max_iterations = 300;
  DavidsonInfo<T> info;
  const Status st = p.solve(options, &info);

  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(info);
  // The relative predicate's bound tol * max(|theta|, ||H||_2) is at most
  // tol * ||A||_2: Ritz values lie inside A's spectrum.
  EXPECT_LE(static_cast<double>(info.max_residual_norm),
            static_cast<double>(options.residual_tolerance) * spectral_norm(ref));
  const auto vals = p.eigenvalues();
  for (int i = 0; i < p.n_roots; ++i) {
    // ascending, and matching the reference's lowest n_roots.
    EXPECT_NEAR(static_cast<double>(vals[static_cast<std::size_t>(i)]),
                static_cast<double>(ref[static_cast<std::size_t>(i)]), eig_tol);
    if (i > 0) {
      EXPECT_LE(vals[static_cast<std::size_t>(i - 1)],
                vals[static_cast<std::size_t>(i)] + static_cast<T>(eig_tol));
    }
  }
}

TEST(DavidsonReferenceTests, LowestEigenpairsDouble) { check_reference<double>(); }
TEST(DavidsonReferenceTests, LowestEigenpairsFloat) { check_reference<float>(); }

// A linear_operator model and the sigma lambda over the same A take the same
// path: the same iterations, one apply per iteration (a block of new columns
// each), every one in HOST pointer mode, and LAPACKE_?syevd's eigenvalues.
template<typename T>
void check_operator_model() {
  const bool is_float = std::is_same_v<T, float>;
  EuclideanProblem<T> p;
  const auto ref = reference_eigenvalues<T>(p.n, p.a);
  const double eig_tol = is_float ? 5e-3 : 1e-6;

  DavidsonOptions<T> options;
  options.residual_tolerance = is_float ? T{1e-4} : T{1e-8};
  options.max_iterations = 300;
  DavidsonInfo<T> by_sigma;
  ASSERT_TRUE(p.solve(options, &by_sigma).ok());

  GemmOperator<T> op{p.h.blas, p.n, p.d_a.data()};
  DavidsonInfo<T> by_operator;
  const Status st = p.solve(options, &by_operator, op);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(by_operator);
  EXPECT_EQ(by_operator.iterations, by_sigma.iterations);
  EXPECT_EQ(op.applies, by_operator.iterations);
  EXPECT_TRUE(op.host_mode);

  const auto vals = p.eigenvalues();
  for (int i = 0; i < p.n_roots; ++i) {
    EXPECT_NEAR(static_cast<double>(vals[static_cast<std::size_t>(i)]),
                static_cast<double>(ref[static_cast<std::size_t>(i)]), eig_tol)
        << i;
  }
}

TEST(DavidsonReferenceTests, OperatorModelMatchesSigmaDouble) { check_operator_model<double>(); }
TEST(DavidsonReferenceTests, OperatorModelMatchesSigmaFloat) { check_operator_model<float>(); }

/// The two non-converged stops: each is success, with the reason set and the
/// first iteration's Ritz values (A's leading block) still written.
template<typename T>
void check_first_iteration_stop(int max_iterations, bool zero_correction,
                                DavidsonStopReason expected) {
  EuclideanProblem<T> p;
  DavidsonOptions<T> options;
  options.max_iterations = max_iterations;
  DavidsonInfo<T> info;
  const Status st = p.solve(options, &info, zero_correction);

  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_FALSE(converged(info));
  EXPECT_EQ(static_cast<int>(info.reason), static_cast<int>(expected));
  EXPECT_EQ(info.iterations, 1);
  // Above tol * ||A||_2, so above every root's relative bound: not converged.
  EXPECT_GT(static_cast<double>(info.max_residual_norm),
            static_cast<double>(options.residual_tolerance) *
                spectral_norm(reference_eigenvalues<T>(p.n, p.a)));
  const auto vals = p.eigenvalues();
  const auto ref = p.leading_block_eigenvalues();
  const double tol = std::is_same_v<T, float> ? 1e-5 : 1e-12;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(vals[i]), static_cast<double>(ref[i]), tol) << i;
  }
}

TEST(DavidsonReferenceTests, MaxIterationsIsSuccess) {
  check_first_iteration_stop<double>(1, false, DavidsonStopReason::MaxIterations);
  check_first_iteration_stop<float>(1, false, DavidsonStopReason::MaxIterations);
}

TEST(DavidsonReferenceTests, ZeroCorrectionStagnates) {
  check_first_iteration_stop<double>(100, true, DavidsonStopReason::Stagnated);
  check_first_iteration_stop<float>(100, true, DavidsonStopReason::Stagnated);
}

TEST(DavidsonReferenceTests, ZeroBudgetZeroesEigenvalues) {
  EuclideanProblem<double> p;
  DavidsonOptions<double> options;
  options.max_iterations = 0;
  DavidsonInfo<double> info;
  ASSERT_TRUE(p.solve(options, &info).ok());
  EXPECT_EQ(static_cast<int>(info.reason), static_cast<int>(DavidsonStopReason::MaxIterations));
  EXPECT_EQ(info.iterations, 0);
  EXPECT_EQ(p.eigenvalues(), std::vector<double>(p.n_roots, 0.0));
}

// ── generalized (metric) solve vs the reference LAPACK (REQUIRES_GPU) ─────────
//
// The metric path solves the generalized problem A x = lambda M x (A symmetric,
// M SPD) by presenting the operator Sigma = M^{-1} A, which is self-adjoint in
// the M-inner-product: davidson's Ritz values are the generalized eigenvalues.
// The oracle is LAPACKE_?sygvd on (A, M); Sigma is precomputed on the host
// (LAPACKE_?posv solving M Sigma = A) so the sigma/metric callbacks are plain
// gemms. All oracle arithmetic is in double.

template<typename T>
std::vector<T> spd_matrix(int n, double diag, double eps, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<T> m(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = j; i < n; ++i) {
      const double v = (i == j) ? diag : eps * dist(rng); // diag-dominant => SPD
      m[static_cast<std::size_t>(j) * n + i] = static_cast<T>(v);
      m[static_cast<std::size_t>(i) * n + j] = static_cast<T>(v);
    }
  }
  return m;
}

// Sigma = M^{-1} A via LAPACKE_?posv (solves M X = A, X overwrites the RHS).
lapack_int call_posv(int n, double *m, double *a) {
  return LAPACKE_dposv(LAPACK_COL_MAJOR, 'L', n, n, m, n, a, n);
}
// Generalized eigenvalues of A x = lambda M x (itype 1), ascending.
lapack_int call_sygvd(int n, double *a, double *m, double *w) {
  return LAPACKE_dsygvd(LAPACK_COL_MAJOR, 1, 'N', 'L', n, a, n, m, n, w);
}

template<typename T>
std::vector<T> cast_vec(const std::vector<double> &v) {
  std::vector<T> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<T>(v[i]);
  }
  return out;
}

template<typename T>
void check_reference_metric() {
  constexpr int n = 40;
  constexpr int n_roots = 3;
  constexpr int max_subspace = 18;
  const bool is_float = std::is_same_v<T, float>;
  const T res_tol = is_float ? T{1e-3} : T{1e-8};
  const double eig_tol = is_float ? 2e-2 : 1e-5;

  auto handle = shared_device();
  Handles h = make_handles(handle);

  const auto a_d = diag_plus_perturbation<double>(n, 0.05, 20261004u);
  const auto m_d = spd_matrix<double>(n, 4.0, 0.03, 77u);

  // Reference generalized eigenvalues (copies: sygvd overwrites both operands).
  std::vector<double> ref(static_cast<std::size_t>(n));
  {
    auto a_copy = a_d;
    auto m_copy = m_d;
    ASSERT_EQ(call_sygvd(n, a_copy.data(), m_copy.data(), ref.data()), 0);
  }

  // Sigma = M^{-1} A (copies: posv overwrites both operands; Sigma lands in a_copy).
  std::vector<double> sigma_d = a_d;
  {
    auto m_copy = m_d;
    ASSERT_EQ(call_posv(n, m_copy.data(), sigma_d.data()), 0);
  }
  std::vector<double> diag_sigma(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    diag_sigma[static_cast<std::size_t>(j)] = sigma_d[static_cast<std::size_t>(j) * n + j];
  }

  DeviceBuffer<T> d_sigma = to_device(handle, cast_vec<T>(sigma_d));
  DeviceBuffer<T> d_m = to_device(handle, cast_vec<T>(m_d));

  std::vector<T> guess_host(static_cast<std::size_t>(n) * n_roots, T{0});
  for (int c = 0; c < n_roots; ++c) {
    guess_host[static_cast<std::size_t>(c) * n + c] = T{1};
  }
  DeviceBuffer<T> d_guess = to_device(handle, guess_host);
  DeviceBuffer<T> d_vals(static_cast<std::size_t>(n_roots), handle);
  DeviceBuffer<T> d_vecs(static_cast<std::size_t>(n) * n_roots, handle);

  std::size_t lwork = 0;
  ASSERT_TRUE(davidson_bufferSize<T>(h.solver, n, n_roots, max_subspace, true, &lwork).ok());
  DeviceBuffer<std::byte> work(lwork, handle);
  DavidsonSlices<T> s;
  ASSERT_TRUE(
      make_davidson_slices<T>(h.solver, n, n_roots, max_subspace, true, work.data(), &s, &lwork)
          .ok());

  const auto sigma = [&](wwr::wwrStream_t, int block, const T *b, T *out) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::gemm<T, int>(h.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, block, n, &one,
                             d_sigma.data(), n, b, n, &zero, out, n);
  };
  const auto metric = [&](wwr::wwrStream_t, int block, const T *b, T *out) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::gemm<T, int>(h.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, block, n, &one,
                             d_m.data(), n, b, n, &zero, out, n);
  };
  const auto precondition = [&](wwr::wwrStream_t stream, int roots, const T *theta,
                                const T *residual, T *correction) -> Status {
    const std::size_t cnt = static_cast<std::size_t>(n) * roots;
    std::vector<T> rbuf(cnt);
    std::vector<T> cbuf(cnt);
    if (const Status st{wwr::wwrMemcpyAsync(rbuf.data(), residual, sizeof(T) * cnt,
                                            wwr::wwrMemcpyDeviceToHost, stream)};
        !st.ok()) {
      return st;
    }
    wwr::wwrStreamSynchronize(stream);
    for (int i = 0; i < roots; ++i) {
      for (int j = 0; j < n; ++j) {
        T denom = theta[i] - static_cast<T>(diag_sigma[static_cast<std::size_t>(j)]);
        if (std::abs(denom) < T{1e-3}) {
          denom = denom < T{0} ? T{-1e-3} : T{1e-3};
        }
        cbuf[static_cast<std::size_t>(i) * n + j] =
            rbuf[static_cast<std::size_t>(i) * n + j] / denom;
      }
    }
    const Status st{wwr::wwrMemcpyAsync(correction, cbuf.data(), sizeof(T) * cnt,
                                        wwr::wwrMemcpyHostToDevice, stream)};
    wwr::wwrStreamSynchronize(stream);
    return st;
  };

  DavidsonOptions<T> options;
  options.residual_tolerance = res_tol;
  options.max_iterations = 400;

  DavidsonInfo<T> info;
  const Status st =
      davidson_solve<T>(h.blas, h.solver, handle->stream().get(), n, n_roots, max_subspace,
                        d_guess.data(), n_roots, s, sigma, precondition, d_vals.data(),
                        d_vecs.data(), &info, options, metric);

  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_CONVERGED(info);
  const auto vals = to_host(handle, d_vals.data(), n_roots);
  for (int i = 0; i < n_roots; ++i) {
    EXPECT_NEAR(static_cast<double>(vals[static_cast<std::size_t>(i)]),
                ref[static_cast<std::size_t>(i)], eig_tol);
  }

  destroy_handles(h);
}

TEST(DavidsonMetricReferenceTests, GeneralizedLowestEigenpairsDouble) {
  check_reference_metric<double>();
}
TEST(DavidsonMetricReferenceTests, GeneralizedLowestEigenpairsFloat) {
  check_reference_metric<float>();
}

} // namespace
} // namespace calaman
