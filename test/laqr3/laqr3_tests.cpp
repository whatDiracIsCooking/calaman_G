// Oracle test for calaman.laqr3: one aggressive-early-deflation (AED) pass on the
// device must agree with reference LAPACK -- ?laqr3 in the SAME precision on the
// host. ?laqr3 is ?laqr2 byte-for-byte EXCEPT the window Schur form: ?laqr2
// always uses ?lahqr, ?laqr3 uses the recursive multishift ?laqr4 when the window
// JW exceeds the ILAENV(12) crossover NMIN = 75. So this suite has two regimes:
//
//   SMALL WINDOW (JW <= 75): ?laqr3 == ?laqr2, and the device delegates the whole
//   pass to the shipped calaman.laqr2 kernel. Both run the identical ?lahqr-based
//   algorithm, so -- as in the laqr2 suite -- the result is reproduced
//   ELEMENT-WISE (the deflated H, the accumulated Z, SR / SI, and NS / ND), with
//   only device FMA contraction separating an accepted result.
//
//   LARGE WINDOW (JW > 75): the window Schur form is the recursive ?laqr4, whose
//   REAL SCHUR FACTORIZATION IS NOT UNIQUE -- device and reference legitimately
//   pick different (equally valid) Schur bases (vector signs, close-eigenvalue
//   order), exactly as the laqr4 suite documents. So element-wise H / Z vs the
//   reference is meaningless here; instead the oracle checks backend-stable
//   invariants that still pin the device to the reference's answer: the window
//   EIGENVALUES vs the reference as a sorted set, and the device's OWN pass is a
//   valid orthogonal similarity -- Z orthogonal and Z * H_out * Z^T == H_in.
//
// ?laqr3 is a computational auxiliary with no LAPACKE C binding, so the oracle
// calls the Fortran symbol slaqr3_ / dlaqr3_ directly, exactly as the laqr2 /
// laqr4 suites call theirs. Fortran passes every scalar by reference; WANTT /
// WANTZ (LOGICAL) are 4-byte ints (0 / 1); KTOP / KBOT / ILOZ / IHIZ are 1-based.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages H / Z on the device and
// runs the kernels (the laqr2 delegate, or laqr3_setup / laqr4 / laqr3_finish), so
// the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise, so
// its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.laqr3;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?laqr3: no LAPACKE binding, so the Fortran symbols directly. Argument
// list is ?laqr2's plus WORK / LWORK (the ?laqr4 workspace). All by reference;
// WANTT / WANTZ (LOGICAL) are 4-byte ints (0 / 1).
extern "C" {
void slaqr3_(const int *wantt, const int *wantz, const int *n, const int *ktop, const int *kbot,
             const int *nw, float *h, const int *ldh, const int *iloz, const int *ihiz, float *z,
             const int *ldz, int *ns, int *nd, float *sr, float *si, float *v, const int *ldv,
             const int *nh, float *t, const int *ldt, const int *nv, float *wv, const int *ldwv,
             float *work, const int *lwork);
void dlaqr3_(const int *wantt, const int *wantz, const int *n, const int *ktop, const int *kbot,
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

// Column-major (i,j) index into a matrix with leading dimension ld, 0-based.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// Reference dispatch, same precision as the device path. A generous LWORK (the
// WORK holds the ?laqr4 scratch); the invariant comparison is schedule-
// independent, so an over-large LWORK is harmless.
void ref_laqr3(int wantt, int wantz, int n, int ktop, int kbot, int nw, float *h, int ldh, int iloz,
               int ihiz, float *z, int ldz, int *ns, int *nd, float *sr, float *si) {
  const int ldv = nw, ldt = nw, ldwv = nw, nh = nw, nv = nw;
  const int lwork = 8 * nw * nw + 16;
  std::vector<float> v(static_cast<std::size_t>(ldv) * nw);
  std::vector<float> t(static_cast<std::size_t>(ldt) * nw);
  std::vector<float> wv(static_cast<std::size_t>(ldwv) * nw);
  std::vector<float> work(static_cast<std::size_t>(lwork));
  slaqr3_(&wantt, &wantz, &n, &ktop, &kbot, &nw, h, &ldh, &iloz, &ihiz, z, &ldz, ns, nd, sr, si,
          v.data(), &ldv, &nh, t.data(), &ldt, &nv, wv.data(), &ldwv, work.data(), &lwork);
}
void ref_laqr3(int wantt, int wantz, int n, int ktop, int kbot, int nw, double *h, int ldh,
               int iloz, int ihiz, double *z, int ldz, int *ns, int *nd, double *sr, double *si) {
  const int ldv = nw, ldt = nw, ldwv = nw, nh = nw, nv = nw;
  const int lwork = 8 * nw * nw + 16;
  std::vector<double> v(static_cast<std::size_t>(ldv) * nw);
  std::vector<double> t(static_cast<std::size_t>(ldt) * nw);
  std::vector<double> wv(static_cast<std::size_t>(ldwv) * nw);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dlaqr3_(&wantt, &wantz, &n, &ktop, &kbot, &nw, h, &ldh, &iloz, &ihiz, z, &ldz, ns, nd, sr, si,
          v.data(), &ldv, &nh, t.data(), &ldt, &nv, wv.data(), &ldwv, work.data(), &lwork);
}

// Device LWORK (governs the port's calaman.laqr4) and the matching WORK length
// for a window of order nw -- NWMAX / NSMAX exactly as calaman.laqr4 carves them,
// then 2 V/T windows + the NS-by-NS shift copy + a 1x1 Z dummy + 3 counters. nw
// is an upper bound on the actual JW, and the length is monotone in n, so sizing
// for nw covers every window the pass forms.
int dev_lwork(int nw) {
  return 2 * nw + 16;
}
std::size_t dev_workbuf_len(int nw) {
  const int lwork = dev_lwork(nw);
  const int nwmax = std::min((nw - 1) / 3, lwork / 2);
  int nsmax = std::min((nw - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 4;
}

// Stage the common device buffers and run calaman.laqr3; returns handle + the
// device buffers so each case can read back what it checks. ld == n throughout.
template<typename T>
struct DeviceRun {
  std::shared_ptr<DeviceHandle> handle;
  DeviceBuffer<T> h, z, sr, si;
  int ns, nd;
};

template<typename T>
DeviceRun<T> run_device(bool wantt, bool wantz, int n, int ktop, int kbot, int nw,
                        const std::vector<T> &h0, const char *ctx) {
  const int ld = n;
  const int iloz = 1, ihiz = n;
  const int ldv = nw, ldt = nw, ldwv = nw;

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
  auto d_work = to_device(handle, std::vector<T>(dev_workbuf_len(nw), T{0}));
  auto d_ns = to_device(handle, std::vector<int>{-1});
  auto d_nd = to_device(handle, std::vector<int>{-1});

  const auto status = laqr3<T>(handle->stream().get(), wantt, wantz, n, ktop, kbot, nw, d_h.data(),
                               ld, iloz, ihiz, d_z.data(), ld, d_ns.data(), d_nd.data(),
                               d_sr.data(), d_si.data(), d_v.data(), ldv, nw, d_t.data(), ldt, nw,
                               d_wv.data(), ldwv, d_work.data(), dev_lwork(nw));
  EXPECT_EQ(status, wwr::wwrSuccess) << ctx;

  const int g_ns = from_device(handle, d_ns, 1)[0];
  const int g_nd = from_device(handle, d_nd, 1)[0];
  return {handle, std::move(d_h), std::move(d_z), std::move(d_sr), std::move(d_si), g_ns, g_nd};
}

// A relative tolerance scaled by ||H||. The AED pass is a LONG reflector chain
// (the window Schur form, the ?trexc reorders, the ?gehrd re-Hessenberg, the
// ?ormhr accumulate and the slab products) -- for the large-window path it runs
// the whole multishift ?laqr4 inside that -- so rounding compounds; device FMA
// contraction is the only algorithmic divergence, hence the large constant (the
// laqr2 suite uses 16384, laqr4 65536).
template<typename T>
T tol(T hnorm) {
  return T{65536} * eps<T>() * (hnorm + T{1});
}

// ---- Small window (JW <= 75): element-wise, exactly the laqr2 oracle ----
template<typename T>
void run_case_small(bool wantt, bool wantz, int n, const std::vector<T> &h0, int ktop, int kbot,
                    int nw, const char *ctx) {
  const int ld = n;
  T hnorm = T{1};
  for (const T e : h0) {
    hnorm = std::max(hnorm, std::abs(e));
  }

  std::vector<T> r_h = h0;
  std::vector<T> r_z(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_z[idx(i, i, ld)] = T{1};
  }
  std::vector<T> r_sr(static_cast<std::size_t>(kbot), T{0}), r_si(static_cast<std::size_t>(kbot),
                                                                  T{0});
  int r_ns = -1, r_nd = -1;
  ref_laqr3(wantt ? 1 : 0, wantz ? 1 : 0, n, ktop, kbot, nw, r_h.data(), ld, 1, n, r_z.data(), ld,
            &r_ns, &r_nd, r_sr.data(), r_si.data());

  auto run = run_device<T>(wantt, wantz, n, ktop, kbot, nw, h0, ctx);
  EXPECT_EQ(run.ns, r_ns) << ctx << " ns";
  EXPECT_EQ(run.nd, r_nd) << ctx << " nd";

  const auto g_h = from_device(run.handle, run.h, static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t k = idx(i, j, ld);
      EXPECT_NEAR(g_h[k], r_h[k], tol(hnorm)) << ctx << " H(" << i << "," << j << ")";
    }
  }
  if (wantz) {
    const auto g_z = from_device(run.handle, run.z, static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const std::size_t k = idx(i, j, ld);
        EXPECT_NEAR(g_z[k], r_z[k], tol(hnorm)) << ctx << " Z(" << i << "," << j << ")";
      }
    }
  }
  const auto g_sr = from_device(run.handle, run.sr, static_cast<std::size_t>(kbot));
  const auto g_si = from_device(run.handle, run.si, static_cast<std::size_t>(kbot));
  for (int i = 0; i < kbot; ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_NEAR(g_sr[u], r_sr[u], tol(hnorm)) << ctx << " SR(" << i << ")";
    EXPECT_NEAR(g_si[u], r_si[u], tol(hnorm)) << ctx << " SI(" << i << ")";
  }
}

// The window eigenvalues over [kwtop, kbot] (1-based) as a set, sorted by
// (real, imag), so the comparison is independent of the Schur-diagonal order.
template<typename T>
std::vector<std::pair<T, T>> sorted_window_eigs(const std::vector<T> &sr, const std::vector<T> &si,
                                               int kwtop, int kbot) {
  std::vector<std::pair<T, T>> e;
  for (int i = kwtop - 1; i < kbot; ++i) {
    e.emplace_back(sr[static_cast<std::size_t>(i)], si[static_cast<std::size_t>(i)]);
  }
  std::sort(e.begin(), e.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first < b.first : a.second < b.second;
  });
  return e;
}

// ---- Large window (JW > 75): invariants, the laqr4 oracle's philosophy ----
template<typename T>
void run_case_large(bool wantz_dummy, int n, const std::vector<T> &h0, int ktop, int kbot, int nw,
                   const char *ctx) {
  (void)wantz_dummy;
  const int ld = n;
  const int jw = std::min(nw, kbot - ktop + 1);
  const int kwtop = kbot - jw + 1;
  ASSERT_GT(jw, 75) << ctx << " (case must exercise the laqr4 path)";

  T hnorm = T{1};
  for (const T e : h0) {
    hnorm = std::max(hnorm, std::abs(e));
  }

  // Reference: whole-matrix window, WANTT + WANTZ, Z starts as I.
  std::vector<T> r_h = h0;
  std::vector<T> r_z(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_z[idx(i, i, ld)] = T{1};
  }
  std::vector<T> r_sr(static_cast<std::size_t>(kbot), T{0}), r_si(static_cast<std::size_t>(kbot),
                                                                  T{0});
  int r_ns = -1, r_nd = -1;
  ref_laqr3(1, 1, n, ktop, kbot, nw, r_h.data(), ld, 1, n, r_z.data(), ld, &r_ns, &r_nd,
            r_sr.data(), r_si.data());

  auto run = run_device<T>(/*wantt=*/true, /*wantz=*/true, n, ktop, kbot, nw, h0, ctx);

  // 1) Window eigenvalues vs the reference, as a sorted set. ND + NS = JW on both.
  EXPECT_EQ(run.ns + run.nd, jw) << ctx << " ns+nd";
  EXPECT_EQ(r_ns + r_nd, jw) << ctx << " ref ns+nd";
  const auto g_sr = from_device(run.handle, run.sr, static_cast<std::size_t>(kbot));
  const auto g_si = from_device(run.handle, run.si, static_cast<std::size_t>(kbot));
  const auto ge = sorted_window_eigs(g_sr, g_si, kwtop, kbot);
  const auto re = sorted_window_eigs(r_sr, r_si, kwtop, kbot);
  ASSERT_EQ(ge.size(), re.size()) << ctx;
  for (std::size_t i = 0; i < ge.size(); ++i) {
    EXPECT_NEAR(ge[i].first, re[i].first, tol(hnorm)) << ctx << " eig.re(" << i << ")";
    EXPECT_NEAR(ge[i].second, re[i].second, tol(hnorm)) << ctx << " eig.im(" << i << ")";
  }

  // 2) The device's OWN pass is a valid orthogonal similarity on the whole
  //    matrix: Z orthogonal and Z * H_out * Z^T == H_in.
  const auto g_h = from_device(run.handle, run.h, static_cast<std::size_t>(n) * n);
  const auto g_z = from_device(run.handle, run.z, static_cast<std::size_t>(n) * n);

  const T otol = T{1024} * eps<T>() * static_cast<T>(n);
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      T s{0};
      for (int k = 0; k < n; ++k) {
        s += g_z[idx(k, a, ld)] * g_z[idx(k, b, ld)];
      }
      EXPECT_NEAR(s, a == b ? T{1} : T{0}, otol) << ctx << " ZtZ(" << a << "," << b << ")";
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      T s{0};
      for (int p = 0; p < n; ++p) {
        T tz{0};
        for (int q = 0; q < n; ++q) {
          tz += g_h[idx(p, q, ld)] * g_z[idx(j, q, ld)];
        }
        s += g_z[idx(i, p, ld)] * tz;
      }
      EXPECT_NEAR(s, h0[idx(i, j, ld)], tol(hnorm)) << ctx << " recon(" << i << "," << j << ")";
    }
  }
}

// Set entry (i,j), 0-based, in an ld-leading-dimension column-major matrix.
template<typename T>
void put(std::vector<T> &a, int i, int j, int ld, T v) {
  a[idx(i, j, ld)] = v;
}

// A real upper-Hessenberg H of order n with a WELL-SEPARATED spectrum (geometric
// diagonal, ratio r). The small-window cases want the subdiagonal weak near the
// window top (a small spike) so some eigenvalues deflate and others survive; the
// large-window cases just want a graded, unambiguous spectrum.
template<typename T>
std::vector<T> make_hess(int n, int kwtop, double r) {
  std::vector<T> h(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= std::min(j + 1, n - 1); ++i) {
      T v;
      if (i == j) {
        v = static_cast<T>(2.0 * std::pow(r, i));
      } else if (i == j + 1) {
        const T base = static_cast<T>(0.15 + 0.02 * j);
        v = (j + 1 >= kwtop - 1) ? static_cast<T>(base * 0.02) : base;
      } else {
        v = static_cast<T>(0.1 * (j + 1) - 0.05 * (i + 1));
      }
      put(h, i, j, n, v);
    }
  }
  return h;
}

// As make_hess but with H(ktop, ktop-1) == 0 so rows/cols ktop..kbot form an
// isolated diagonal block, as ?laqr3 assumes.
template<typename T>
std::vector<T> make_hess_isolated(int n, int ktop, int kwtop, double r) {
  std::vector<T> h = make_hess<T>(n, kwtop, r);
  if (ktop > 1) {
    put(h, ktop - 1, ktop - 2, n, T{0});
  }
  return h;
}

template<typename T>
void run_all() {
  // --- Small windows (JW <= 75): delegate to calaman.laqr2, element-wise. The
  //     laqr2 suite's cases, which already walk WANTT +/- WANTZ, a KTOP-isolated
  //     sub-block, and the tiny-window paths. ---
  run_case_small<T>(true, false, 8, make_hess<T>(8, 4, 1.8), 1, 8, 5, "small whole noz");
  run_case_small<T>(true, true, 8, make_hess<T>(8, 4, 1.8), 1, 8, 5, "small whole z");
  run_case_small<T>(false, true, 8, make_hess<T>(8, 4, 1.8), 1, 8, 5, "small whole noT z");
  run_case_small<T>(true, true, 10, make_hess<T>(10, 7, 1.8), 1, 10, 4, "small win4 z");
  run_case_small<T>(true, true, 10, make_hess_isolated<T>(10, 3, 5, 1.8), 3, 9, 5, "small ktop3 z");
  run_case_small<T>(true, true, 6, make_hess<T>(6, 5, 1.8), 1, 6, 2, "small win2 z");

  // --- Large windows (JW > 75): the recursive calaman.laqr4 path, invariants. ---
  // Whole-matrix window, no spike (KWTOP == KTOP, S == 0): the Schur form comes
  // straight from laqr4 and every eigenvalue deflates.
  run_case_large<T>(true, 80, make_hess<T>(80, 1, 1.12), 1, 80, 80, "large whole n80");
  // A window inside a larger matrix (KWTOP = 11 > KTOP = 1), so the spike is live
  // and the deflation / re-Hessenberg tail runs on a JW = 80 > NMIN window.
  run_case_large<T>(true, 90, make_hess<T>(90, 11, 1.12), 1, 90, 80, "large spike n90");
}

} // namespace

TEST(Laqr3OracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(Laqr3OracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

} // namespace calaman
