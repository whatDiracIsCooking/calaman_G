// Oracle test for calaman.laqr2: one aggressive-early-deflation (AED) pass on the
// device must agree with reference LAPACK -- ?laqr2 in the SAME precision on the
// host. The device path must reproduce the deflated Hessenberg H, the accumulated
// Z (when WANTZ), the window eigenvalues SR / SI (at the positions ?laqr2
// documents), and the shift/deflation counts NS / ND.
//
// ?laqr2 is a computational auxiliary with no LAPACKE C binding, so the oracle
// calls the Fortran symbol slaqr2_ / dlaqr2_ directly, exactly as the laqr5 /
// laexc suites call their auxiliaries. Fortran passes every scalar by reference;
// WANTT / WANTZ are LOGICALs, which gfortran represents as a 4-byte int (0 / 1).
// KTOP / KBOT / ILOZ / IHIZ are 1-based in the ABI, and laqr2 takes the same.
// The reference overwrites its V / T / WV / WORK scratch (dimensioned to its
// declared bounds); the device takes V / T scratch and ignores NH / NV / WV.
//
// The device is the small-window ?laqr2 variant (?lahqr for the window Schur
// form), so the oracle MUST be ?laqr2, not ?laqr3 (which recurses through
// ?laqr4) -- they deflate identically but reduce the window by different code.
// Both run the identical algorithm, so only rounding (device FMA contraction)
// separates an accepted result -- a relative tolerance scaled by ||H||.
//
// The cases walk a real upper-Hessenberg H whose trailing window carries both
// deflatable eigenvalues (tiny spike coupling, so they peel off) and
// undeflatable ones (reordered up and reflected back): WANTT with and without
// WANTZ, windows that cover the whole block and a KTOP-isolated sub-block, and a
// mix of real and complex-conjugate eigenvalues. A small spike H(KWTOP,KWTOP-1)
// makes several eigenvalues deflate; the rest are re-Hessenberged.
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
import calaman.laqr2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?laqr2: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; WANTT / WANTZ (LOGICAL) are 4-byte ints (0 / 1).
extern "C" {
void slaqr2_(const int *wantt, const int *wantz, const int *n, const int *ktop, const int *kbot,
             const int *nw, float *h, const int *ldh, const int *iloz, const int *ihiz, float *z,
             const int *ldz, int *ns, int *nd, float *sr, float *si, float *v, const int *ldv,
             const int *nh, float *t, const int *ldt, const int *nv, float *wv, const int *ldwv,
             float *work, const int *lwork);
void dlaqr2_(const int *wantt, const int *wantz, const int *n, const int *ktop, const int *kbot,
             const int *nw, double *h, const int *ldh, const int *iloz, const int *ihiz, double *z,
             const int *ldz, int *ns, int *nd, double *sr, double *si, double *v, const int *ldv,
             const int *nh, double *t, const int *ldt, const int *nv, double *wv, const int *ldwv,
             double *work, const int *lwork);
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
void ref_laqr2(int wantt, int wantz, int n, int ktop, int kbot, int nw, float *h, int ldh, int iloz,
               int ihiz, float *z, int ldz, int *ns, int *nd, float *sr, float *si) {
  const int ldv = nw, ldt = nw, ldwv = nw, nh = nw, nv = nw, lwork = 4 * nw + 8;
  std::vector<float> v(static_cast<std::size_t>(ldv) * nw);
  std::vector<float> t(static_cast<std::size_t>(ldt) * nw);
  std::vector<float> wv(static_cast<std::size_t>(ldwv) * nw);
  std::vector<float> work(static_cast<std::size_t>(lwork));
  slaqr2_(&wantt, &wantz, &n, &ktop, &kbot, &nw, h, &ldh, &iloz, &ihiz, z, &ldz, ns, nd, sr, si,
          v.data(), &ldv, &nh, t.data(), &ldt, &nv, wv.data(), &ldwv, work.data(), &lwork);
}
void ref_laqr2(int wantt, int wantz, int n, int ktop, int kbot, int nw, double *h, int ldh,
               int iloz, int ihiz, double *z, int ldz, int *ns, int *nd, double *sr, double *si) {
  const int ldv = nw, ldt = nw, ldwv = nw, nh = nw, nv = nw, lwork = 4 * nw + 8;
  std::vector<double> v(static_cast<std::size_t>(ldv) * nw);
  std::vector<double> t(static_cast<std::size_t>(ldt) * nw);
  std::vector<double> wv(static_cast<std::size_t>(ldwv) * nw);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dlaqr2_(&wantt, &wantz, &n, &ktop, &kbot, &nw, h, &ldh, &iloz, &ihiz, z, &ldz, ns, nd, sr, si,
          v.data(), &ldv, &nh, t.data(), &ldt, &nv, wv.data(), &ldwv, work.data(), &lwork);
}

// Column-major (i,j) index into a matrix with leading dimension ld, 0-based.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// A relative tolerance scaled by ||H||. The AED pass is a LONG chain -- the
// iterative ?lahqr, the ?trexc reorders, the ?gehrd re-Hessenberg, the ?ormhr
// accumulate and three slab gemms -- so the reflector products accumulate far
// more rounding than a single-routine oracle (laqr5 uses 256, trexc 512); device
// FMA contraction is the only divergence given the identical algorithm, but it
// compounds across the chain, hence the larger constant.
template<typename T>
T tol(T hnorm) {
  return T{16384} * eps<T>() * (hnorm + T{1});
}

// One case: laqr2 on the device must match the reference. h0 is an n-by-n
// column-major real upper-Hessenberg matrix; ktop/kbot/iloz/ihiz are 1-based.
template<typename T>
void run_case(bool wantt, bool wantz, int n, const std::vector<T> &h0, int ktop, int kbot, int nw,
              const char *ctx) {
  const int ld = n;
  const int iloz = 1, ihiz = n;
  const int ldv = nw, ldt = nw, ldwv = nw;

  T hnorm = T{1};
  for (const T e : h0) {
    hnorm = std::max(hnorm, std::abs(e));
  }

  // Reference overwrites H, Z, SR, SI, NS, ND in place.
  std::vector<T> r_h = h0;
  std::vector<T> r_z(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_z[idx(i, i, ld)] = T{1};
  }
  std::vector<T> r_sr(static_cast<std::size_t>(kbot), T{0}), r_si(static_cast<std::size_t>(kbot),
                                                                  T{0});
  int r_ns = -1, r_nd = -1;
  ref_laqr2(wantt ? 1 : 0, wantz ? 1 : 0, n, ktop, kbot, nw, r_h.data(), ld, iloz, ihiz,
            r_z.data(), ld, &r_ns, &r_nd, r_sr.data(), r_si.data());

  auto handle = shared_device();
  auto d_h = to_device(handle, h0);
  std::vector<T> z0(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    z0[idx(i, i, ld)] = T{1};
  }
  auto d_z = to_device(handle, z0);
  auto d_sr = to_device(handle, std::vector<T>(static_cast<std::size_t>(kbot), T{0}));
  auto d_si = to_device(handle, std::vector<T>(static_cast<std::size_t>(kbot), T{0}));
  auto d_v = to_device(handle, std::vector<T>(static_cast<std::size_t>(ldv) * nw, T{0}));
  auto d_t = to_device(handle, std::vector<T>(static_cast<std::size_t>(ldt) * nw, T{0}));
  auto d_wv = to_device(handle, std::vector<T>(static_cast<std::size_t>(ldwv) * nw, T{0}));
  auto d_ns = to_device(handle, std::vector<int>{-1});
  auto d_nd = to_device(handle, std::vector<int>{-1});

  const auto status =
      laqr2<T>(handle->stream().get(), wantt, wantz, n, ktop, kbot, nw, d_h.data(), ld, iloz, ihiz,
               d_z.data(), ld, d_ns.data(), d_nd.data(), d_sr.data(), d_si.data(), d_v.data(), ldv,
               nw, d_t.data(), ldt, nw, d_wv.data(), ldwv);
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const auto g_ns = from_device(handle, d_ns, 1)[0];
  const auto g_nd = from_device(handle, d_nd, 1)[0];
  EXPECT_EQ(g_ns, r_ns) << ctx << " ns";
  EXPECT_EQ(g_nd, r_nd) << ctx << " nd";

  const auto g_h = from_device(handle, d_h, static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t k = idx(i, j, ld);
      EXPECT_NEAR(g_h[k], r_h[k], tol(hnorm)) << ctx << " H(" << i << "," << j << ")";
    }
  }
  if (wantz) {
    const auto g_z = from_device(handle, d_z, static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t k = idx(i, j, ld);
        EXPECT_NEAR(g_z[k], r_z[k], tol(hnorm)) << ctx << " Z(" << i << "," << j << ")";
      }
    }
  }
  // The window eigenvalues land in SR/SI positions KWTOP..KBOT (1-based); compare
  // the whole prefix the reference wrote.
  const auto g_sr = from_device(handle, d_sr, static_cast<std::size_t>(kbot));
  const auto g_si = from_device(handle, d_si, static_cast<std::size_t>(kbot));
  for (int i = 0; i < kbot; ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_NEAR(g_sr[u], r_sr[u], tol(hnorm)) << ctx << " SR(" << i << ")";
    EXPECT_NEAR(g_si[u], r_si[u], tol(hnorm)) << ctx << " SI(" << i << ")";
  }
}

// Set entry (i,j), 0-based, in an ld-leading-dimension column-major matrix.
template<typename T>
void put(std::vector<T> &a, int i, int j, int ld, T v) {
  a[idx(i, j, ld)] = v;
}

// A real upper-Hessenberg H of order n. The diagonal is WELL SEPARATED
// (geometrically spaced, ratio ~1.8) so the graded-matrix bubble sort ?laqr2
// runs orders its blocks by an unambiguous margin in both float and double --
// clustered near-equal eigenvalues leave that EVI >= EVK comparison on a float
// knife-edge, where device and reference pick different (both valid) Schur bases
// that agree on eigenvalues but not on H / Z. The first subdiagonal is modest,
// and smaller still near the window top so the spike H(kwtop, kwtop-1) is weak
// and several window eigenvalues deflate while others survive.
template<typename T>
std::vector<T> make_hess(int n, int kwtop) {
  std::vector<T> h(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= std::min(j + 1, n - 1); ++i) {
      T v;
      if (i == j) {
        // Geometric diagonal 2, 3.6, 6.48, ... : well separated in magnitude.
        v = static_cast<T>(2.0 * std::pow(1.8, i));
      } else if (i == j + 1) {
        // Small subdiagonal, smaller still around the window top (0-based kwtop-1).
        const T base = static_cast<T>(0.15 + 0.02 * j);
        v = (j + 1 >= kwtop - 1) ? static_cast<T>(base * 0.02) : base;
      } else {
        v = static_cast<T>(0.1 * (j + 1) - 0.05 * (i + 1)); // strict upper
      }
      put(h, i, j, n, v);
    }
  }
  return h;
}

// A KTOP-isolated active block: as make_hess but with H(ktop, ktop-1) == 0 so
// rows/cols ktop..kbot form an isolated diagonal block, as ?laqr2 assumes.
template<typename T>
std::vector<T> make_hess_isolated(int n, int ktop, int kwtop) {
  std::vector<T> h = make_hess<T>(n, kwtop);
  if (ktop > 1) {
    put(h, ktop - 1, ktop - 2, n, T{0}); // H(ktop, ktop-1) = 0, 0-based
  }
  return h;
}

template<typename T>
void run_all() {
  // Whole-block window on an order-8 matrix, Schur factor only (no Z).
  run_case<T>(/*wantt=*/true, /*wantz=*/false, 8, make_hess<T>(8, 4), 1, 8, 5, "whole noz");
  // Same, accumulating Z.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 8, make_hess<T>(8, 4), 1, 8, 5, "whole z");
  // Eigenvalues-only update (wantt=false): only enough of H is touched.
  run_case<T>(/*wantt=*/false, /*wantz=*/true, 8, make_hess<T>(8, 4), 1, 8, 5, "whole noT z");
  // Smaller window (fewer rows in the deflation block) on a larger matrix.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 10, make_hess<T>(10, 7), 1, 10, 4, "win4 z");
  // KTOP > 1 isolated sub-block (ktop=3, kbot=9) with a window inside it.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 10, make_hess_isolated<T>(10, 3, 5), 3, 9, 5,
              "ktop3 z");
  // Minimal non-trivial window (nw=2 exercises the 1x1 / tiny-window paths).
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 6, make_hess<T>(6, 5), 1, 6, 2, "win2 z");
}

} // namespace

TEST(Laqr2OracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(Laqr2OracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

} // namespace calaman
