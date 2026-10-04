// Suite for calaman.davidson (skeleton). Two things are testable before the
// iteration lands:
//
//   * the argument-checking contract of davidson_bufferSize /
//     make_davidson_slices -- a bad shape or a null out-pointer is rejected with
//     WWRBLAS_STATUS_INVALID_VALUE before any handle use (host-only);
//   * the workspace sizing and carving -- a valid shape sizes to a positive byte
//     count that grows with the subspace and with the metric path, and carving a
//     real buffer hands back non-null, 256-aligned, in-range pointers, with the
//     metric regions present only when sized with_metric (REQUIRES_GPU: it needs
//     a cuSOLVER handle for syevd_bufferSize and a device buffer to carve).
//
// davidson_solve is a NOT_SUPPORTED stub and is not exercised here.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.davidson;
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

// ── argument checking (host-only; the handle is never dereferenced) ──────────

TEST(DavidsonArgCheckTests, RejectsBadShape) {
  const wwr::wwrsolverDnHandle_t no_handle{}; // never touched on the rejection path
  std::size_t lwork = 0;

  // n and n_roots must be positive.
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 0, 1, 2, false, &lwork).code, kInvalidValue);
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 4, 0, 2, false, &lwork).code, kInvalidValue);
  // n_roots must not exceed n.
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 4, 5, 8, false, &lwork).code, kInvalidValue);
  // max_subspace must be at least 2 * n_roots ...
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 16, 4, 7, false, &lwork).code, kInvalidValue);
  // ... and at most n.
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 8, 2, 16, false, &lwork).code, kInvalidValue);
}

TEST(DavidsonArgCheckTests, RejectsNullOutPointer) {
  const wwr::wwrsolverDnHandle_t no_handle{};
  EXPECT_EQ(davidson_bufferSize<double>(no_handle, 16, 4, 8, false, nullptr).code, kInvalidValue);
}

TEST(DavidsonArgCheckTests, SolveStubReportsNotSupported) {
  // The skeleton's solve is explicitly unimplemented; pin that it says so.
  DavidsonSlices<double> s;
  DavidsonResult<double> result;
  const Status st = davidson_solve<double>(wwr::wwrblasHandle_t{}, wwr::wwrsolverDnHandle_t{},
                                           wwr::wwrStream_t{}, 16, 4, 8, nullptr, 4, s, {}, {},
                                           nullptr, &result);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code, static_cast<int>(wwr::WWRBLAS_STATUS_NOT_SUPPORTED));
}

// ── sizing and carving (REQUIRES_GPU) ────────────────────────────────────────

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
std::size_t size_for(wwr::wwrsolverDnHandle_t solver, int n, int n_roots, int max_subspace,
                     bool with_metric) {
  std::size_t lwork = 0;
  const Status st =
      davidson_bufferSize<T>(solver, n, n_roots, max_subspace, with_metric, &lwork);
  EXPECT_TRUE(st.ok()) << "domain " << static_cast<int>(st.domain) << " code " << st.code;
  return lwork;
}

template<typename T>
void check_sizing(wwr::wwrsolverDnHandle_t solver) {
  constexpr int n = 64;
  constexpr int n_roots = 4;

  // A valid shape sizes to a positive byte count.
  const std::size_t small = size_for<T>(solver, n, n_roots, 2 * n_roots, false);
  EXPECT_GT(small, 0u);

  // Growing the subspace does not shrink the workspace (V/Sigma_V scale with it).
  const std::size_t large = size_for<T>(solver, n, n_roots, 4 * n_roots, false);
  EXPECT_GE(large, small);

  // The metric path carves M V, V^T M V and a metric scratch block on top, so it
  // is strictly larger than the Euclidean path at the same shape.
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

// Every slice pointer that must be live is non-null, 256-aligned, and inside the
// reported workspace; the metric slices follow with_metric.
template<typename T>
void check_carve(std::shared_ptr<DeviceHandle> handle, wwr::wwrsolverDnHandle_t solver,
                 bool with_metric) {
  constexpr int n = 48;
  constexpr int n_roots = 3;
  constexpr int max_subspace = 4 * n_roots;

  std::size_t lwork = 0;
  ASSERT_TRUE(
      davidson_bufferSize<T>(solver, n, n_roots, max_subspace, with_metric, &lwork).ok());
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

} // namespace
} // namespace calaman
