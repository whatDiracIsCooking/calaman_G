// Oracle test for calaman.lahqr: the device double-shift Francis QR reduction of
// an upper Hessenberg matrix to real Schur form must agree with reference LAPACK
// -- ?lahqr -- computed in the SAME precision on the host. The device path must
// produce the same Schur form H (when WANTT), the same accumulated Z (when
// WANTZ), the same eigenvalues WR / WI, and the same INFO.
//
// ?lahqr is an auxiliary routine with no LAPACKE C binding, so the oracle calls
// the Fortran symbol slahqr_ / dlahqr_ directly, exactly as the laexc / lanv2
// suites call their auxiliaries. Fortran passes every scalar by reference; WANTT
// / WANTZ are LOGICALs, which gfortran represents as a 4-byte int (0 / 1). ILO /
// IHI / ILOZ / IHIZ are 1-based in the Fortran ABI, and lahqr takes the same.
//
// Several real Hessenberg matrices of varying order are reduced: small ones with
// real eigenvalues, ones that deflate complex-conjugate pairs through the 2x2
// ?lanv2 path, and a larger one that exercises repeated bulge chasing. Both paths
// run the identical algorithm, so for a converged reduction INFO is 0 on both and
// only rounding (device FMA contraction) separates the numbers -- a tolerance
// that scales with the entry magnitude, because a Schur form grows no larger than
// the input norm but the QR sweep is many fused steps.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages H / Z on the device and
// runs the kernel, so the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lahqr;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?lahqr: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; WANTT / WANTZ (LOGICAL) are 4-byte ints (0 / 1).
extern "C" {
void slahqr_(const int *wantt, const int *wantz, const int *n, const int *ilo, const int *ihi,
             float *h, const int *ldh, float *wr, float *wi, const int *iloz, const int *ihiz,
             float *z, const int *ldz, int *info);
void dlahqr_(const int *wantt, const int *wantz, const int *n, const int *ilo, const int *ihi,
             double *h, const int *ldh, double *wr, double *wi, const int *iloz, const int *ihiz,
             double *z, const int *ldz, int *info);
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
void ref_lahqr(int wantt, int wantz, int n, int ilo, int ihi, float *h, int ldh, float *wr,
               float *wi, int iloz, int ihiz, float *z, int ldz, int *info) {
  slahqr_(&wantt, &wantz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z, &ldz, info);
}
void ref_lahqr(int wantt, int wantz, int n, int ilo, int ihi, double *h, int ldh, double *wr,
               double *wi, int iloz, int ihiz, double *z, int ldz, int *info) {
  dlahqr_(&wantt, &wantz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z, &ldz, info);
}

// A relative tolerance tight enough to catch a wrong entry yet generous for the
// many fused QR sweeps each path runs (device FMA contraction is the only
// divergence for a converged reduction).
template<typename T>
T tol(T scale) {
  const T a = scale < T{0} ? -scale : scale;
  return T{4096} * eps<T>() * (a + T{1});
}

// Column-major (i,j) index into an n-row matrix, 0-based i, j.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// One case: lahqr on the device must match the reference. H is an n-by-n column-
// major upper Hessenberg matrix; Z starts as the identity when wantz. The window
// is the whole matrix (ilo=1, ihi=n, iloz=1, ihiz=n).
template<typename T>
void run_case(bool wantt, bool wantz, int n, const std::vector<T> &h0, const char *ctx) {
  const int ld = n;

  // Reference overwrites H (and Z) in place.
  std::vector<T> r_h = h0;
  std::vector<T> r_z(static_cast<std::size_t>(n) * n, T{0});
  std::vector<T> r_wr(n, T{0}), r_wi(n, T{0});
  for (int i = 0; i < n; ++i) {
    r_z[idx(i, i, ld)] = T{1};
  }
  int r_info = -1;
  ref_lahqr(wantt ? 1 : 0, wantz ? 1 : 0, n, 1, n, r_h.data(), ld, r_wr.data(), r_wi.data(), 1, n,
            r_z.data(), ld, &r_info);

  auto handle = shared_device();
  auto d_h = to_device(handle, h0);
  std::vector<T> z0(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    z0[idx(i, i, ld)] = T{1};
  }
  auto d_z = to_device(handle, z0);
  auto d_wr = to_device(handle, std::vector<T>(n, T{0}));
  auto d_wi = to_device(handle, std::vector<T>(n, T{0}));
  auto d_info = to_device(handle, std::vector<int>{-1});

  const auto status = lahqr<T>(handle->stream().get(), wantt, wantz, n, 1, n, d_h.data(), ld,
                               d_wr.data(), d_wi.data(), 1, n, d_z.data(), ld, d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const int g_info = from_device(handle, d_info, 1)[0];
  EXPECT_EQ(g_info, r_info) << ctx << " info";
  if (r_info != 0 || g_info != 0) {
    return; // non-convergence: compare INFO alone
  }

  // Eigenvalues: same order on both paths (both run the identical deflation).
  const auto g_wr = from_device(handle, d_wr, n);
  const auto g_wi = from_device(handle, d_wi, n);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(g_wr[i], r_wr[i], tol(r_wr[i])) << ctx << " wr(" << i << ")";
    EXPECT_NEAR(g_wi[i], r_wi[i], tol(r_wi[i])) << ctx << " wi(" << i << ")";
  }

  if (wantt) {
    const auto g_h = from_device(handle, d_h, static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t k = idx(i, j, ld);
        EXPECT_NEAR(g_h[k], r_h[k], tol(r_h[k])) << ctx << " H(" << i << "," << j << ")";
      }
    }
  }
  if (wantz) {
    const auto g_z = from_device(handle, d_z, static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t k = idx(i, j, ld);
        EXPECT_NEAR(g_z[k], r_z[k], tol(r_z[k])) << ctx << " Z(" << i << "," << j << ")";
      }
    }
  }
}

// Set entry (i,j), 0-based, in an n-row column-major matrix.
template<typename T>
void put(std::vector<T> &a, int i, int j, int n, T v) {
  a[idx(i, j, n)] = v;
}

// A dense upper Hessenberg matrix (zero below the first subdiagonal) built from a
// small integer seed so float and double see the same entries. coef shapes the
// spectrum: a larger subdiagonal tends to force complex pairs.
template<typename T>
std::vector<T> hessenberg(int n, T coef) {
  std::vector<T> h(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= j + 1 && i < n; ++i) {
      // Upper triangle and the first subdiagonal (i == j+1).
      const T v = static_cast<T>((((i * 7 + j * 3) % 11) - 5)) + coef * static_cast<T>(i == j + 1);
      put(h, i, j, n, v);
    }
  }
  return h;
}

template<typename T>
void run_all(bool wantt, bool wantz) {
  const char *t = wantt ? " T" : " e";
  const char *z = wantz ? "Z" : "-";
  auto label = [&](const char *base) {
    return std::string(base) + t + z;
  };
  // Order 2: a single 2x2 that deflates in one step (real or complex).
  {
    std::vector<T> h = hessenberg<T>(2, T{3});
    run_case<T>(wantt, wantz, 2, h, label("n2").c_str());
  }
  // Order 4 and 5: a mix of 1x1 and 2x2 deflations.
  {
    std::vector<T> h = hessenberg<T>(4, T{2});
    run_case<T>(wantt, wantz, 4, h, label("n4").c_str());
  }
  {
    std::vector<T> h = hessenberg<T>(5, T{4});
    run_case<T>(wantt, wantz, 5, h, label("n5").c_str());
  }
  // Order 8: repeated bulge chasing down a longer window.
  {
    std::vector<T> h = hessenberg<T>(8, T{3});
    run_case<T>(wantt, wantz, 8, h, label("n8").c_str());
  }
}

} // namespace

TEST(LahqrOracleTests, MatchesReferenceFloat) {
  run_all<float>(/*wantt=*/true, /*wantz=*/true);
  run_all<float>(/*wantt=*/true, /*wantz=*/false);
  run_all<float>(/*wantt=*/false, /*wantz=*/false);
}

TEST(LahqrOracleTests, MatchesReferenceDouble) {
  run_all<double>(/*wantt=*/true, /*wantz=*/true);
  run_all<double>(/*wantt=*/true, /*wantz=*/false);
  run_all<double>(/*wantt=*/false, /*wantz=*/false);
}

// An odd order that deflates a 2x2 at the bottom then a 1x1 above it, so both
// the ?lanv2 standardisation and the trailing single-eigenvalue store run.
TEST(LahqrOracleTests, OrderThree) {
  const int n = 3;
  auto h0 = hessenberg<double>(n, 2.0);
  run_case<double>(/*wantt=*/true, /*wantz=*/true, n, h0, "n3 TZ");
}

} // namespace calaman
