// Oracle test for calaman.lahr2: the Hessenberg-panel reduction -- the reduced
// matrix A, the reflector scalars tau, the triangular factor T and the auxiliary
// Y -- must all agree with the reference LAPACK, computed in the SAME precision
// on the host.
//
// The oracle is netlib's own ?lahr2, called through its Fortran ABI (dlahr2_ /
// slahr2_), not LAPACKE: this LAPACK build ships no LAPACKE_?lahr2 C binding
// (?lahr2 is an auxiliary routine), but the Fortran symbol is in LAPACK::LAPACK,
// which calaman::lapack_reference links. ?lahr2 takes only integer/real
// arguments -- no CHARACTER -- so there is no hidden gfortran length. The
// reference matrix is column-major, matching the device storage.
//
// Each case picks n, k, nb with 1 <= nb <= n - k and k >= 1 so the Y(1:k, :)
// block is non-trivial; both the n > k + nb branch (the trailing gemm runs) and
// the n == k + nb branch (it is skipped) are covered, plus the nb == 1 edge.
// A is a deterministic pseudo-random fill, generic enough that every column's
// reflector is non-degenerate.
//
// Comparisons: A is checked in full (the reduced entries, the reflector tails,
// AND the columns past nb the routine must leave unchanged); tau and Y in full;
// T only on and above the diagonal -- its strict lower triangle is scratch in
// both paths (the reference never writes it, lahr2 reuses its last column as the
// apply workspace), so comparing it would read undefined memory.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A on the device and runs
// the gemv/gemm/trmv/trmm kernels, so `ctest -LE gpu` excludes it. Built only
// when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass
// (docs/architecture.md section 3).

#include <gtest/gtest.h>

#include <cstddef> // std::size_t for the Fortran prototypes below

// Netlib ?lahr2 through its Fortran ABI. gfortran passes scalars/arrays by
// reference; ?lahr2 has no CHARACTER argument, so there is no trailing hidden
// length. The symbols live in LAPACK::LAPACK, linked by calaman::lapack_reference.
extern "C" {
void slahr2_(const int *n, const int *k, const int *nb, float *a, const int *lda, float *tau,
             float *t, const int *ldt, float *y, const int *ldy);
void dlahr2_(const int *n, const int *k, const int *nb, double *a, const int *lda, double *tau,
             double *t, const int *ldt, double *y, const int *ldy);
}

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lahr2;
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
using test::factorization_tol;
using test::frobenius_norm;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

/// @brief Upload `host` to a fresh device buffer on `handle`'s stream
template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const HostBuffer<T> &host,
                          std::size_t n) {
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, host, handle->stream().get());
  return device;
}

/// @brief Download `device` (length n) back to a host vector
template<typename T>
std::vector<T> from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<T> &device,
                           std::size_t n) {
  HostBuffer<T> host(n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  std::vector<T> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = host.data()[i];
  }
  return out;
}

// The reference oracle: netlib ?lahr2 (Fortran ABI), same precision as the
// device path, column-major to match the device storage.
void ref_lahr2(int n, int k, int nb, float *a, int lda, float *tau, float *t, int ldt, float *y,
               int ldy) {
  slahr2_(&n, &k, &nb, a, &lda, tau, t, &ldt, y, &ldy);
}
void ref_lahr2(int n, int k, int nb, double *a, int lda, double *tau, double *t, int ldt, double *y,
               int ldy) {
  dlahr2_(&n, &k, &nb, a, &lda, tau, t, &ldt, y, &ldy);
}

/// @brief A deterministic, well-spread pseudo-random value in [-10, 10]
template<typename T>
T gen(std::size_t i, unsigned seed) {
  unsigned x = static_cast<unsigned>(i) * 2654435761u + seed * 40503u + 1u;
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return static_cast<T>(static_cast<int>(x % 2001) - 1000) / static_cast<T>(100);
}

/// @brief calaman::lahr2 on the device must match the reference for one case
///
/// A is n x (n-k+1) column-major with lda == n; T is nb x nb (ldt == nb); Y is
/// n x nb (ldy == n). The reference reduces a host copy; lahr2 reduces a device
/// copy of the same A. A, tau and Y are compared element-by-element, T only on
/// and above its diagonal, all to the shared factorization tolerance scaled by
/// the relevant output's norm.
template<typename T>
void expect_matches_reference(int n, int k, int nb, unsigned seed) {
  ASSERT_GE(k, 0);
  ASSERT_LT(k, n);
  ASSERT_GE(nb, 1);
  ASSERT_LE(nb, n - k);

  const int acols = n - k + 1;
  const std::size_t an = static_cast<std::size_t>(n) * static_cast<std::size_t>(acols);
  const std::size_t tn = static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb);
  const std::size_t yn = static_cast<std::size_t>(n) * static_cast<std::size_t>(nb);
  const std::size_t taun = static_cast<std::size_t>(nb);

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  // Fill A once; ref_a is the host oracle's workspace, host_a the device upload.
  HostBuffer<T> host_a(an);
  std::vector<T> ref_a(an);
  for (std::size_t i = 0; i < an; ++i) {
    host_a.data()[i] = gen<T>(i, seed);
    ref_a[i] = host_a.data()[i];
  }

  // Reference: reduce ref_a in host memory, filling ref_tau / ref_t / ref_y.
  std::vector<T> ref_tau(taun, T{0});
  std::vector<T> ref_t(tn, T{0});
  std::vector<T> ref_y(yn, T{0});
  ref_lahr2(n, k, nb, ref_a.data(), n, ref_tau.data(), ref_t.data(), nb, ref_y.data(), n);

  // Device: reduce a copy of the same A.
  auto d_a = to_device(handle, host_a, an);
  DeviceBuffer<T> d_tau(taun, handle);
  DeviceBuffer<T> d_t(tn, handle);
  DeviceBuffer<T> d_y(yn, handle);

  const auto status =
      lahr2<T>(blas, n, k, nb, d_a.data(), n, d_tau.data(), d_t.data(), nb, d_y.data(), n);
  const auto got_a = from_device(handle, d_a, an);
  const auto got_tau = from_device(handle, d_tau, taun);
  const auto got_t = from_device(handle, d_t, tn);
  const auto got_y = from_device(handle, d_y, yn);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " k=" << k << " nb=" << nb;

  const std::string tag =
      "n=" + std::to_string(n) + " k=" + std::to_string(k) + " nb=" + std::to_string(nb);
  const T norm_a = frobenius_norm<T>(ref_a);
  const T tol_a = factorization_tol<T>(norm_a, static_cast<std::size_t>(n),
                                       static_cast<std::size_t>(acols));
  for (std::size_t i = 0; i < an; ++i) {
    EXPECT_NEAR(got_a[i], ref_a[i], tol_a) << tag << " A i=" << i;
  }

  const T tol_tau = factorization_tol<T>(norm_a, static_cast<std::size_t>(nb), 1);
  for (std::size_t i = 0; i < taun; ++i) {
    EXPECT_NEAR(got_tau[i], ref_tau[i], tol_tau) << tag << " tau i=" << i;
  }

  // T only on and above the diagonal (column-major, ldt == nb): r <= c.
  const T tol_t = factorization_tol<T>(frobenius_norm<T>(ref_t), static_cast<std::size_t>(nb),
                                       static_cast<std::size_t>(nb));
  for (int c = 0; c < nb; ++c) {
    for (int r = 0; r <= c; ++r) {
      const std::size_t idx = static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * nb;
      EXPECT_NEAR(got_t[idx], ref_t[idx], tol_t) << tag << " T r=" << r << " c=" << c;
    }
  }

  const T tol_y = factorization_tol<T>(frobenius_norm<T>(ref_y), static_cast<std::size_t>(n),
                                       static_cast<std::size_t>(nb));
  for (std::size_t i = 0; i < yn; ++i) {
    EXPECT_NEAR(got_y[i], ref_y[i], tol_y) << tag << " Y i=" << i;
  }
}

TEST(Lahr2OracleTests, MatchesReferenceFloat) {
  expect_matches_reference<float>(8, 2, 3, 1);  // n > k + nb: trailing gemm runs
  expect_matches_reference<float>(7, 3, 2, 2);  // n > k + nb
  expect_matches_reference<float>(6, 1, 4, 3);  // n > k + nb, K dim 1
  expect_matches_reference<float>(6, 2, 4, 4);  // n == k + nb: gemm skipped
  expect_matches_reference<float>(9, 4, 5, 5);  // n == k + nb: gemm skipped
  expect_matches_reference<float>(5, 1, 1, 6);  // nb == 1 edge
}

TEST(Lahr2OracleTests, MatchesReferenceDouble) {
  expect_matches_reference<double>(8, 2, 3, 11);
  expect_matches_reference<double>(7, 3, 2, 12);
  expect_matches_reference<double>(6, 1, 4, 13);
  expect_matches_reference<double>(6, 2, 4, 14);
  expect_matches_reference<double>(9, 4, 5, 15);
  expect_matches_reference<double>(5, 1, 1, 16);
  expect_matches_reference<double>(12, 3, 5, 17); // larger panel, gemm runs
}

} // namespace
} // namespace calaman
