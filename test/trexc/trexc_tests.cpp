// Oracle test for calaman.trexc: the device reorder of a real Schur form --
// moving a selected diagonal block from IFST to ILST by a sequence of adjacent
// swaps -- must agree with reference LAPACK ?trexc computed in the SAME precision
// on the host. The device path must produce the same T, the same Q (when
// wantq), the same final IFST / ILST, and the same INFO.
//
// ?trexc HAS a LAPACKE C binding (unlike ?laexc), so the oracle calls
// LAPACKE_strexc / LAPACKE_dtrexc with LAPACK_COL_MAJOR -- no Fortran-symbol
// dance. COMPQ is 'V' (update Q) or 'N'. IFST / ILST are 1-based and in/out:
// LAPACKE passes them by pointer and updates them to the blocks' final first-row
// positions; trexc takes host int* with the same in/out contract.
//
// The Schur forms are built valid by construction: upper quasi-triangular with
// distinct real eigenvalues on the 1x1 blocks and standardised complex-conjugate
// pairs (a == d, b*c < 0) on the 2x2 blocks, with a strict-upper fill elsewhere.
// Cases move a block DOWN and UP, and across every 1x1 / 2x2 neighbour mix, so
// the forward and backward bubbling paths and the 2x2-split handling are all
// exercised. Both paths run the identical algorithm, so for an accepted move
// only rounding (device FMA contraction) separates the numbers -- a tight
// relative tolerance.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages T / Q on the device and
// runs the kernel loop, so the suite is excluded by `ctest -LE gpu`. Built only
// when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.trexc;
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
using test::eps;
using test::shared_device;

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

// Reference ?trexc via LAPACKE (column-major). compq is 'V' or 'N'; ifst/ilst
// are 1-based and updated in place; returns INFO.
int ref_trexc(char compq, int n, float *t, int ldt, float *q, int ldq, int *ifst, int *ilst) {
  return LAPACKE_strexc(LAPACK_COL_MAJOR, compq, n, t, ldt, q, ldq, ifst, ilst);
}
int ref_trexc(char compq, int n, double *t, int ldt, double *q, int ldq, int *ifst, int *ilst) {
  return LAPACKE_dtrexc(LAPACK_COL_MAJOR, compq, n, t, ldt, q, ldq, ifst, ilst);
}

// A relative tolerance tight enough to catch a wrong entry yet generous for the
// chain of swaps each path runs (device FMA contraction is the only divergence
// for an accepted move).
template<typename T>
T tol(T scale) {
  const T a = scale < T{0} ? -scale : scale;
  return T{512} * eps<T>() * (a + T{1});
}

// Column-major (i,j) index into an n-row matrix, 0-based i, j.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// One case: trexc on the device must match the reference. t0 is an n-by-n
// column-major real Schur form; Q starts as the identity when wantq. ifst/ilst
// are 1-based on entry.
template<typename T>
void run_case(bool wantq, int n, const std::vector<T> &t0, int ifst, int ilst, const char *ctx) {
  const int ld = n;
  const char compq = wantq ? 'V' : 'N';

  // Reference overwrites T (and Q) and updates ifst/ilst in place.
  std::vector<T> r_t = t0;
  std::vector<T> r_q(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_q[idx(i, i, ld)] = T{1};
  }
  int r_ifst = ifst;
  int r_ilst = ilst;
  const int r_info = ref_trexc(compq, n, r_t.data(), ld, r_q.data(), ld, &r_ifst, &r_ilst);

  auto handle = shared_device();
  auto d_t = to_device(handle, t0);
  std::vector<T> q0(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    q0[idx(i, i, ld)] = T{1};
  }
  auto d_q = to_device(handle, q0);
  auto d_laexc_info = to_device(handle, std::vector<int>{-1});

  int g_ifst = ifst;
  int g_ilst = ilst;
  int g_info = -1;
  const auto status = trexc<T>(handle->stream().get(), wantq, n, d_t.data(), ld, d_q.data(), ld,
                               &g_ifst, &g_ilst, &g_info, d_laexc_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  EXPECT_EQ(g_info, r_info) << ctx << " info";
  EXPECT_EQ(g_ifst, r_ifst) << ctx << " ifst";
  EXPECT_EQ(g_ilst, r_ilst) << ctx << " ilst";

  const auto g_t = from_device(handle, d_t, static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t k = idx(i, j, ld);
      EXPECT_NEAR(g_t[k], r_t[k], tol(r_t[k])) << ctx << " T(" << i << "," << j << ")";
    }
  }
  if (wantq) {
    const auto g_q = from_device(handle, d_q, static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t k = idx(i, j, ld);
        EXPECT_NEAR(g_q[k], r_q[k], tol(r_q[k])) << ctx << " Q(" << i << "," << j << ")";
      }
    }
  }
}

// Set entry (i,j), 0-based, in an n-row column-major matrix.
template<typename T>
void put(std::vector<T> &a, int i, int j, int n, T v) {
  a[idx(i, j, n)] = v;
}

// A single standardised 2x2 complex-conjugate block at 0-based position p:
// [center gamma; -beta center] with gamma, beta > 0 (b*c < 0, a == d).
template<typename T>
void put_2x2(std::vector<T> &t, int p, int n, T center, T gamma, T beta) {
  put(t, p, p, n, center);
  put(t, p, p + 1, n, gamma);
  put(t, p + 1, p, n, -beta);
  put(t, p + 1, p + 1, n, center);
}

// A 7x7 Schur form with block layout [1x1, 2x2, 1x1, 2x2, 1x1] starting at
// 0-based rows 0, 1, 3, 4, 6 (1-based first rows 1, 2, 4, 5, 7). Distinct real
// eigenvalues; standardised complex pairs; a strict-upper fill that keeps it
// quasi-triangular. The blocks' 1-based first rows are the valid IFST / ILST.
template<typename T>
std::vector<T> form_mixed() {
  const int n = 7;
  std::vector<T> t(static_cast<std::size_t>(n) * n, T{0});
  // Real eigenvalues on the 1x1 blocks.
  put(t, 0, 0, n, T{10});
  put(t, 3, 3, n, T{7});
  put(t, 6, 6, n, T{1});
  // Standardised 2x2 complex pairs.
  put_2x2<T>(t, 1, n, T{5}, T{3}, T{2}); // 5 +/- i*sqrt(6)
  put_2x2<T>(t, 4, n, T{3}, T{2}, T{3}); // 3 +/- i*sqrt(6)
  // Strict-upper fill, leaving the standardised 2x2 subdiagonals alone.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      put(t, i, j, n, static_cast<T>(0.3 * (j + 1) - 0.15 * (i + 1)));
    }
  }
  // Restore the two 2x2 subdiagonals the fill overwrote (c < 0).
  put(t, 2, 1, n, T{-2});
  put(t, 5, 4, n, T{-3});
  return t;
}

// Block first rows (1-based) in form_mixed: 1(1x1) 2(2x2) 4(1x1) 5(2x2) 7(1x1).
template<typename T>
void run_all(bool wantq) {
  const char *q = wantq ? " (wantq)" : " (noq)";

  // --- Move DOWN (ifst < ilst). ---
  // 1x1 down past a 2x2 and more: block at row 1 -> row 7.
  run_case<T>(wantq, 7, form_mixed<T>(), 1, 7, (std::string("down 1x1->end") + q).c_str());
  // 2x2 down past a 1x1 then a 2x2: block at row 2 -> row 7.
  run_case<T>(wantq, 7, form_mixed<T>(), 2, 7, (std::string("down 2x2") + q).c_str());
  // 1x1 down one neighbour (a 2x2): row 4 -> row 6.
  run_case<T>(wantq, 7, form_mixed<T>(), 4, 6, (std::string("down 1x1 short") + q).c_str());
  // 2x2 down one neighbour (a 1x1): row 2 -> row 4.
  run_case<T>(wantq, 7, form_mixed<T>(), 2, 4, (std::string("down 2x2 short") + q).c_str());

  // --- Move UP (ifst > ilst). ---
  // 1x1 up to the top: row 7 -> row 1.
  run_case<T>(wantq, 7, form_mixed<T>(), 7, 1, (std::string("up 1x1->top") + q).c_str());
  // 2x2 up past a 1x1 then a 2x2: row 5 -> row 1.
  run_case<T>(wantq, 7, form_mixed<T>(), 5, 1, (std::string("up 2x2") + q).c_str());
  // 1x1 up one neighbour (a 2x2): row 4 -> row 2.
  run_case<T>(wantq, 7, form_mixed<T>(), 4, 2, (std::string("up 1x1 short") + q).c_str());
  // 2x2 up one neighbour (a 1x1): row 5 -> row 4.
  run_case<T>(wantq, 7, form_mixed<T>(), 5, 4, (std::string("up 2x2 short") + q).c_str());

  // --- IFST inside a 2x2 block: snapped to the first row, then no-op move. ---
  // ifst=3 is the 2nd row of the 2x2 at row 2; ilst=2 is its first row -> no move.
  run_case<T>(wantq, 7, form_mixed<T>(), 3, 2, (std::string("snap no-op") + q).c_str());
}

} // namespace

TEST(TrexcOracleTests, MatchesReferenceFloat) {
  run_all<float>(/*wantq=*/false);
  run_all<float>(/*wantq=*/true);
}

TEST(TrexcOracleTests, MatchesReferenceDouble) {
  run_all<double>(/*wantq=*/false);
  run_all<double>(/*wantq=*/true);
}

// Quick return: n <= 1 reorders nothing and reports info 0. trexc must agree
// with the reference, which leaves a 1x1 matrix untouched.
TEST(TrexcOracleTests, QuickReturnTiny) {
  const int n = 1;
  std::vector<double> t{double{3}};
  run_case<double>(/*wantq=*/true, n, t, /*ifst=*/1, /*ilst=*/1, "n=1 quick return");
}

} // namespace calaman
