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
//     LAPACKE_?syevd's, over float and double (REQUIRES_GPU).
//
// Guarded on calaman::lapack_reference (see CMakeLists.txt): the reference suite
// is the solver's natural oracle, and bundling the rest in the same binary keeps
// one target. The host arithmetic is done in double regardless of T.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

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

TEST(DavidsonArgCheckTests, RejectsBadGuessCount) {
  // guess_count outside [n_roots, max_subspace] is rejected before any handle use.
  DavidsonSlices<double> s;
  DavidsonResult<double> result;
  const auto solve = [&](int guess_count) {
    return davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                  wwr::wwrStream_t{}, 16, 4, 8, nullptr, guess_count, s, {}, {},
                                  nullptr, &result);
  };
  EXPECT_EQ(solve(3).code, kInvalidValue); // below n_roots
  EXPECT_EQ(solve(9).code, kInvalidValue); // above max_subspace
}

TEST(DavidsonArgCheckTests, RejectsMetricWithoutMetricWorkspace) {
  // A non-empty metric selects the generalized path, which needs a workspace
  // sized with_metric; a default (Euclidean) DavidsonSlices has null metric
  // regions, so it is rejected before any handle use rather than silently
  // solving the wrong (Euclidean) problem.
  DavidsonSlices<double> s; // all-null: not sized with_metric
  DavidsonResult<double> result;
  DavidsonMetricFn<double> metric = [](wwr::wwrStream_t, int, const double *, double *) -> Status {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  };
  const Status st = davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                           wwr::wwrStream_t{}, 16, 4, 8, nullptr, 4, s, {}, {},
                                           nullptr, &result, {}, metric);
  EXPECT_EQ(st.code, kInvalidValue);
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

template<typename T>
void check_reference() {
  constexpr int n = 48;
  constexpr int n_roots = 4;
  constexpr int max_subspace = 20;
  const bool is_float = std::is_same_v<T, float>;
  const T res_tol = is_float ? T{1e-4} : T{1e-8};
  const double eig_tol = is_float ? 5e-3 : 1e-6;

  auto handle = shared_device();
  Handles h = make_handles(handle);

  const auto a = diag_plus_perturbation<T>(n, 0.05, 20261004u);
  const auto ref = reference_eigenvalues<T>(n, a);

  std::vector<double> diagd(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    diagd[static_cast<std::size_t>(i)] = static_cast<double>(a[static_cast<std::size_t>(i) * n + i]);
  }

  std::vector<T> guess_host(static_cast<std::size_t>(n) * n_roots, T{0});
  for (int c = 0; c < n_roots; ++c) {
    guess_host[static_cast<std::size_t>(c) * n + c] = T{1}; // orthonormal unit vectors
  }

  DeviceBuffer<T> d_a = to_device(handle, a);
  DeviceBuffer<T> d_guess = to_device(handle, guess_host);
  DeviceBuffer<T> d_vecs(static_cast<std::size_t>(n) * n_roots, handle);

  std::size_t lwork = 0;
  ASSERT_TRUE(davidson_bufferSize<T>(h.solver, n, n_roots, max_subspace, false, &lwork).ok());
  DeviceBuffer<std::byte> work(lwork, handle);
  DavidsonSlices<T> s;
  ASSERT_TRUE(
      make_davidson_slices<T>(h.solver, n, n_roots, max_subspace, false, work.data(), &s, &lwork)
          .ok());

  // sigma(B) = A B via a single gemm; the solver holds the handle in host
  // pointer mode for the whole call, so host scalar pointers are correct here.
  DavidsonSigmaFn<T> sigma = [&](wwr::wwrStream_t, int block, const T *b, T *out) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::gemm<T, int>(h.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, block, n, &one,
                             d_a.data(), n, b, n, &zero, out, n);
  };

  // Diagonal Davidson preconditioner correction = residual / (theta - diag),
  // computed on the host (no kernel needed for a test), guarding the near-zero
  // denominator when a Ritz value approaches its own diagonal entry.
  DavidsonPreconditionFn<T> precondition = [&](wwr::wwrStream_t stream, int roots, const T *theta,
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
        T denom = theta[i] - static_cast<T>(diagd[static_cast<std::size_t>(j)]);
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
  options.max_iterations = 300;

  DavidsonResult<T> result;
  const Status st = davidson_solve<T>(h.blas, h.solver, handle->stream().get(), n, n_roots,
                                      max_subspace, d_guess.data(), n_roots, s, sigma, precondition,
                                      d_vecs.data(), &result, options);

  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_TRUE(result.converged);
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(n_roots));
  for (int i = 0; i < n_roots; ++i) {
    // ascending, and matching the reference's lowest n_roots.
    EXPECT_NEAR(static_cast<double>(result.eigenvalues[static_cast<std::size_t>(i)]),
                static_cast<double>(ref[static_cast<std::size_t>(i)]), eig_tol);
    if (i > 0) {
      EXPECT_LE(result.eigenvalues[static_cast<std::size_t>(i - 1)],
                result.eigenvalues[static_cast<std::size_t>(i)] + static_cast<T>(eig_tol));
    }
  }

  destroy_handles(h);
}

TEST(DavidsonReferenceTests, LowestEigenpairsDouble) { check_reference<double>(); }
TEST(DavidsonReferenceTests, LowestEigenpairsFloat) { check_reference<float>(); }

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
  DeviceBuffer<T> d_vecs(static_cast<std::size_t>(n) * n_roots, handle);

  std::size_t lwork = 0;
  ASSERT_TRUE(davidson_bufferSize<T>(h.solver, n, n_roots, max_subspace, true, &lwork).ok());
  DeviceBuffer<std::byte> work(lwork, handle);
  DavidsonSlices<T> s;
  ASSERT_TRUE(
      make_davidson_slices<T>(h.solver, n, n_roots, max_subspace, true, work.data(), &s, &lwork)
          .ok());

  DavidsonSigmaFn<T> sigma = [&](wwr::wwrStream_t, int block, const T *b, T *out) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::gemm<T, int>(h.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, block, n, &one,
                             d_sigma.data(), n, b, n, &zero, out, n);
  };
  DavidsonMetricFn<T> metric = [&](wwr::wwrStream_t, int block, const T *b, T *out) -> Status {
    const T one{1};
    const T zero{0};
    return wwr::gemm<T, int>(h.blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, block, n, &one,
                             d_m.data(), n, b, n, &zero, out, n);
  };
  DavidsonPreconditionFn<T> precondition = [&](wwr::wwrStream_t stream, int roots, const T *theta,
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

  DavidsonResult<T> result;
  const Status st =
      davidson_solve<T>(h.blas, h.solver, handle->stream().get(), n, n_roots, max_subspace,
                        d_guess.data(), n_roots, s, sigma, precondition, d_vecs.data(), &result,
                        options, metric);

  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  EXPECT_TRUE(result.converged);
  ASSERT_EQ(result.eigenvalues.size(), static_cast<std::size_t>(n_roots));
  for (int i = 0; i < n_roots; ++i) {
    EXPECT_NEAR(static_cast<double>(result.eigenvalues[static_cast<std::size_t>(i)]),
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
