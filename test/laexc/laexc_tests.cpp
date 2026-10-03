// Oracle test for calaman.laexc: the device swap of two adjacent diagonal blocks
// of a real Schur form must agree with reference LAPACK -- ?laexc -- computed in
// the SAME precision on the host. The device path must produce the same T, the
// same Q (when WANTQ), and the same INFO.
//
// ?laexc is a computational routine with no LAPACKE C binding, so the oracle
// calls the Fortran symbol slaexc_ / dlaexc_ directly, exactly as the lanv2 /
// lasy2 suites call their auxiliaries. Fortran passes every scalar by reference;
// WANTQ is a LOGICAL, which gfortran represents as a 4-byte int (0 / 1). J1 is
// 1-based in the Fortran ABI, and laexc takes the same 1-based J1.
//
// The four (N1, N2) combinations {(1,1),(1,2),(2,1),(2,2)} each get a valid real
// Schur form T: upper quasi-triangular with real eigenvalues on the 1x1 blocks
// and standardised complex-conjugate pairs (a == d, b*c < 0) on the 2x2 blocks,
// embedded in a larger N so the trailing DROT / reflector applies touch real
// rows and columns. Both paths run the identical algorithm, so for a swap that
// is accepted INFO is 0 on both and only rounding (device FMA contraction)
// separates the numbers -- a tight relative tolerance. A rejected swap (INFO==1)
// is compared as INFO alone, since the reference leaves T / Q unchanged.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages T / Q on the device and
// runs the kernel, so the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.laexc;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?laexc: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; WANTQ (LOGICAL) is a 4-byte int (0 / 1).
extern "C" {
void slaexc_(const int *wantq, const int *n, float *t, const int *ldt, float *q, const int *ldq,
             const int *j1, const int *n1, const int *n2, float *work, int *info);
void dlaexc_(const int *wantq, const int *n, double *t, const int *ldt, double *q, const int *ldq,
             const int *j1, const int *n1, const int *n2, double *work, int *info);
}

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

// Reference dispatch, same precision as the device path.
void ref_laexc(int wantq, int n, float *t, int ldt, float *q, int ldq, int j1, int n1, int n2,
               int *info) {
  std::vector<float> work(static_cast<std::size_t>(n) + 1);
  slaexc_(&wantq, &n, t, &ldt, q, &ldq, &j1, &n1, &n2, work.data(), info);
}
void ref_laexc(int wantq, int n, double *t, int ldt, double *q, int ldq, int j1, int n1, int n2,
               int *info) {
  std::vector<double> work(static_cast<std::size_t>(n) + 1);
  dlaexc_(&wantq, &n, t, &ldt, q, &ldq, &j1, &n1, &n2, work.data(), info);
}

// A relative tolerance tight enough to catch a wrong entry yet generous for the
// Sylvester solve, the length-3 reflectors and the ?lanv2 standardisation each
// path runs (device FMA contraction is the only divergence for an accepted swap).
template<typename T>
T tol(T scale) {
  const T a = scale < T{0} ? -scale : scale;
  return T{512} * eps<T>() * (a + T{1});
}

// Column-major (i,j) index into an n-row matrix, 0-based i, j.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// One case: laexc on the device must match the reference. T is an n-by-n column-
// major real Schur form; Q starts as the identity when wantq. j1 is 1-based.
template<typename T>
void run_case(bool wantq, int n, const std::vector<T> &t0, int j1, int n1, int n2,
              const char *ctx) {
  const int ld = n;

  // Reference overwrites T (and Q) in place.
  std::vector<T> r_t = t0;
  std::vector<T> r_q(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_q[idx(i, i, ld)] = T{1};
  }
  int r_info = -1;
  ref_laexc(wantq ? 1 : 0, n, r_t.data(), ld, r_q.data(), ld, j1, n1, n2, &r_info);

  auto handle = shared_device();
  auto d_t = to_device(handle, t0);
  std::vector<T> q0(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    q0[idx(i, i, ld)] = T{1};
  }
  auto d_q = to_device(handle, q0);
  auto d_info = to_device(handle, std::vector<int>{-1});

  const auto status = laexc<T>(handle->stream().get(), wantq, n, d_t.data(), ld, d_q.data(), ld, j1,
                               n1, n2, d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const int g_info = from_device(handle, d_info, 1)[0];
  EXPECT_EQ(g_info, r_info) << ctx << " info";

  // A rejected swap leaves T / Q untouched on both paths; info is the only claim.
  if (r_info != 0 || g_info != 0) {
    return;
  }

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

// --- (1,1): two real eigenvalues adjacent, inside a larger triangle. ---
template<typename T>
std::vector<T> form_11(int n, int j1) {
  std::vector<T> t(static_cast<std::size_t>(n) * n, T{0});
  // Distinct real eigenvalues on the diagonal.
  const T diag[] = {T{6}, T{5}, T{4}, T{3}, T{2}};
  for (int i = 0; i < n; ++i) {
    put(t, i, i, n, diag[i % 5]);
  }
  // Upper triangle fill (strictly above diagonal) -- keeps it quasi-triangular.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      put(t, i, j, n, static_cast<T>(0.5 * (i + 1) - 0.3 * (j + 1)));
    }
  }
  (void)j1;
  return t;
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

// --- (1,2): a 1x1 block at j1 followed by a 2x2 complex pair. ---
template<typename T>
std::vector<T> form_12(int n, int j1) {
  std::vector<T> t(static_cast<std::size_t>(n) * n, T{0});
  const int p = j1 - 1; // 0-based start of the 1x1 block
  // Fill leading/trailing 1x1 real eigenvalues.
  for (int i = 0; i < n; ++i) {
    put(t, i, i, n, static_cast<T>(7 - i));
  }
  // The 1x1 block eigenvalue, then the 2x2 pair after it.
  put(t, p, p, n, T{8});
  put_2x2<T>(t, p + 1, n, T{2}, T{3}, T{2}); // eigenvalues 2 +/- i*sqrt(6)
  // Strict upper-triangle fill everywhere else (leave the 2x2 subdiagonal alone).
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      if (!(i == p + 1 && j == p + 1)) {
        put(t, i, j, n, static_cast<T>(0.4 * (j + 1) - 0.2 * (i + 1)));
      }
    }
  }
  return t;
}

// --- (2,1): a 2x2 complex pair at j1 followed by a 1x1 block. ---
template<typename T>
std::vector<T> form_21(int n, int j1) {
  std::vector<T> t(static_cast<std::size_t>(n) * n, T{0});
  const int p = j1 - 1; // 0-based start of the 2x2 block
  for (int i = 0; i < n; ++i) {
    put(t, i, i, n, static_cast<T>(7 - i));
  }
  put_2x2<T>(t, p, n, T{3}, T{4}, T{1}); // eigenvalues 3 +/- i*2
  put(t, p + 2, p + 2, n, T{9});         // the trailing 1x1 block
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      put(t, i, j, n, static_cast<T>(0.3 * (j + 1) - 0.15 * (i + 1)));
    }
  }
  // Restore the standardised 2x2 subdiagonal the fill left untouched (c < 0).
  put(t, p + 1, p, n, T{-1});
  return t;
}

// --- (2,2): two adjacent 2x2 complex pairs. ---
template<typename T>
std::vector<T> form_22(int n, int j1) {
  std::vector<T> t(static_cast<std::size_t>(n) * n, T{0});
  const int p = j1 - 1; // 0-based start of the first 2x2 block
  for (int i = 0; i < n; ++i) {
    put(t, i, i, n, static_cast<T>(7 - i));
  }
  put_2x2<T>(t, p, n, T{2}, T{5}, T{1});     // first pair 2 +/- i*sqrt(5)
  put_2x2<T>(t, p + 2, n, T{6}, T{2}, T{3}); // second pair 6 +/- i*sqrt(6)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      put(t, i, j, n, static_cast<T>(0.25 * (j + 1) - 0.1 * (i + 1)));
    }
  }
  put(t, p + 1, p, n, T{-1});
  put(t, p + 3, p + 2, n, T{-3});
  return t;
}

template<typename T>
void run_all(bool wantq) {
  const char *q = wantq ? " (wantq)" : " (noq)";
  {
    const int n = 5, j1 = 2;
    std::string c = std::string("1x1") + q;
    run_case<T>(wantq, n, form_11<T>(n, j1), j1, 1, 1, c.c_str());
  }
  {
    const int n = 6, j1 = 2;
    std::string c = std::string("1x2") + q;
    run_case<T>(wantq, n, form_12<T>(n, j1), j1, 1, 2, c.c_str());
  }
  {
    const int n = 6, j1 = 2;
    std::string c = std::string("2x1") + q;
    run_case<T>(wantq, n, form_21<T>(n, j1), j1, 2, 1, c.c_str());
  }
  {
    const int n = 7, j1 = 2;
    std::string c = std::string("2x2") + q;
    run_case<T>(wantq, n, form_22<T>(n, j1), j1, 2, 2, c.c_str());
  }
}

} // namespace

TEST(LaexcOracleTests, MatchesReferenceFloat) {
  run_all<float>(/*wantq=*/false);
  run_all<float>(/*wantq=*/true);
}

TEST(LaexcOracleTests, MatchesReferenceDouble) {
  run_all<double>(/*wantq=*/false);
  run_all<double>(/*wantq=*/true);
}

// A quick return: J1 + N1 > N means the first block runs off the matrix, so the
// reference returns INFO = 0 and touches nothing. laexc must agree.
TEST(LaexcOracleTests, QuickReturnOffEnd) {
  const int n = 3;
  auto t0 = form_11<double>(n, 3);
  run_case<double>(/*wantq=*/true, n, t0, /*j1=*/3, /*n1=*/1, /*n2=*/1, "off-end quick return");
}

} // namespace calaman
