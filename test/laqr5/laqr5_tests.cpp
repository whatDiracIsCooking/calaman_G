// Oracle test for calaman.laqr5: one non-accumulated multishift QR sweep on the
// device must agree with reference LAPACK -- ?laqr5 called with KACC22 = 0 -- in
// the SAME precision on the host. The device path must reproduce the swept
// Hessenberg H, the accumulator Z (when WANTZ), and the reordered shift pairs
// SR / SI.
//
// KACC22 MUST MATCH between the two paths: the reference's KACC22 >= 1
// accumulation reorders the far-from-diagonal arithmetic (U plus DGEMM block
// multiplies), which is numerically distinct from the direct KACC22 = 0 sweep
// the device implements. The oracle is pinned to KACC22 = 0, so both run the
// identical algorithm and only rounding (device FMA contraction) separates the
// numbers -- a relative tolerance scaled by ||H|| covers it.
//
// ?laqr5 is a computational routine with no LAPACKE C binding, so the oracle
// calls the Fortran symbol slaqr5_ / dlaqr5_ directly, exactly as the laexc /
// lanv2 / lasy2 suites call their auxiliaries. Fortran passes every scalar by
// reference; WANTT / WANTZ are LOGICALs, which gfortran represents as a 4-byte
// int (0 / 1). KTOP / KBOT / ILOZ / IHIZ are 1-based in the ABI, and laqr5 takes
// the same 1-based indices. The reference writes the V / U / WV / WH workspace;
// with KACC22 = 0 only V (leading dimension >= 3) is touched, but all four are
// dimensioned so the Fortran bounds hold.
//
// The cases walk a chain of a few double-shift bulges (NSHFTS even) through the
// whole active block of a real upper-Hessenberg H: WANTT with and without WANTZ,
// a mix of real and complex-conjugate shift pairs, and a KTOP > 1 isolated
// sub-block. Shift values are arbitrary (both paths take the same ones), chosen
// modest so the safmin rescaling in ?larfg does not fire.
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
import calaman.laqr5;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?laqr5: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; WANTT / WANTZ (LOGICAL) are 4-byte ints (0 / 1).
extern "C" {
void slaqr5_(const int *wantt, const int *wantz, const int *kacc22, const int *n, const int *ktop,
             const int *kbot, const int *nshfts, float *sr, float *si, float *h, const int *ldh,
             const int *iloz, const int *ihiz, float *z, const int *ldz, float *v, const int *ldv,
             float *u, const int *ldu, const int *nv, float *wv, const int *ldwv, const int *nh,
             float *wh, const int *ldwh);
void dlaqr5_(const int *wantt, const int *wantz, const int *kacc22, const int *n, const int *ktop,
             const int *kbot, const int *nshfts, double *sr, double *si, double *h, const int *ldh,
             const int *iloz, const int *ihiz, double *z, const int *ldz, double *v, const int *ldv,
             double *u, const int *ldu, const int *nv, double *wv, const int *ldwv, const int *nh,
             double *wh, const int *ldwh);
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

// Reference dispatch (KACC22 = 0), same precision as the device path. All
// workspace arrays are dimensioned to the reference's declared bounds.
void ref_laqr5(int wantt, int wantz, int n, int ktop, int kbot, int nshfts, float *sr, float *si,
               float *h, int ldh, int iloz, int ihiz, float *z, int ldz) {
  const int kacc22 = 0;
  const int ldv = 3, ldu = 2 * nshfts, nv = n, ldwv = n, nh = n, ldwh = 2 * nshfts;
  std::vector<float> v(static_cast<std::size_t>(ldv) * (nshfts / 2 + 1));
  std::vector<float> u(static_cast<std::size_t>(ldu) * (2 * nshfts));
  std::vector<float> wv(static_cast<std::size_t>(ldwv) * (2 * nshfts));
  std::vector<float> wh(static_cast<std::size_t>(ldwh) * nh);
  slaqr5_(&wantt, &wantz, &kacc22, &n, &ktop, &kbot, &nshfts, sr, si, h, &ldh, &iloz, &ihiz, z, &ldz,
          v.data(), &ldv, u.data(), &ldu, &nv, wv.data(), &ldwv, &nh, wh.data(), &ldwh);
}
void ref_laqr5(int wantt, int wantz, int n, int ktop, int kbot, int nshfts, double *sr, double *si,
               double *h, int ldh, int iloz, int ihiz, double *z, int ldz) {
  const int kacc22 = 0;
  const int ldv = 3, ldu = 2 * nshfts, nv = n, ldwv = n, nh = n, ldwh = 2 * nshfts;
  std::vector<double> v(static_cast<std::size_t>(ldv) * (nshfts / 2 + 1));
  std::vector<double> u(static_cast<std::size_t>(ldu) * (2 * nshfts));
  std::vector<double> wv(static_cast<std::size_t>(ldwv) * (2 * nshfts));
  std::vector<double> wh(static_cast<std::size_t>(ldwh) * nh);
  dlaqr5_(&wantt, &wantz, &kacc22, &n, &ktop, &kbot, &nshfts, sr, si, h, &ldh, &iloz, &ihiz, z, &ldz,
          v.data(), &ldv, u.data(), &ldu, &nv, wv.data(), &ldwv, &nh, wh.data(), &ldwh);
}

// Column-major (i,j) index into a matrix with leading dimension ld, 0-based.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// A relative tolerance scaled by ||H||: tight enough to catch a wrong entry, yet
// generous for the reflector products both paths accumulate (device FMA
// contraction is the only divergence given identical KACC22).
template<typename T>
T tol(T hnorm) {
  return T{256} * eps<T>() * (hnorm + T{1});
}

// One case: laqr5 on the device must match the reference. h0 is an n-by-n
// column-major real upper-Hessenberg matrix; the shifts are sr0 + i*si0 (length
// nshfts, even). ktop/kbot/iloz/ihiz are 1-based.
template<typename T>
void run_case(bool wantt, bool wantz, int n, const std::vector<T> &h0, int ktop, int kbot,
              const std::vector<T> &sr0, const std::vector<T> &si0, const char *ctx) {
  const int ld = n;
  const int nshfts = static_cast<int>(sr0.size());
  const int iloz = 1, ihiz = n;

  T hnorm = T{1};
  for (const T e : h0) {
    hnorm = std::max(hnorm, std::abs(e));
  }

  // Reference overwrites H, Z, SR, SI in place.
  std::vector<T> r_h = h0;
  std::vector<T> r_z(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_z[idx(i, i, ld)] = T{1};
  }
  std::vector<T> r_sr = sr0, r_si = si0;
  ref_laqr5(wantt ? 1 : 0, wantz ? 1 : 0, n, ktop, kbot, nshfts, r_sr.data(), r_si.data(),
            r_h.data(), ld, iloz, ihiz, r_z.data(), ld);

  auto handle = shared_device();
  auto d_h = to_device(handle, h0);
  std::vector<T> z0(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    z0[idx(i, i, ld)] = T{1};
  }
  auto d_z = to_device(handle, z0);
  auto d_sr = to_device(handle, sr0);
  auto d_si = to_device(handle, si0);

  const auto status = laqr5<T>(handle->stream().get(), wantt, wantz, n, ktop, kbot, nshfts,
                               d_sr.data(), d_si.data(), d_h.data(), ld, iloz, ihiz, d_z.data(), ld);
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

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
  // The shift shuffle must match too (SR / SI are reordered in place).
  const auto g_sr = from_device(handle, d_sr, static_cast<std::size_t>(nshfts));
  const auto g_si = from_device(handle, d_si, static_cast<std::size_t>(nshfts));
  for (int i = 0; i < nshfts; ++i) {
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

// A dense-ish real upper-Hessenberg H of order n: full strict-upper triangle, a
// non-trivial first subdiagonal, zero below. Distinct diagonal so the sweep is
// well focused.
template<typename T>
std::vector<T> make_hess(int n) {
  std::vector<T> h(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= std::min(j + 1, n - 1); ++i) {
      T v;
      if (i == j) {
        v = static_cast<T>(n - i); // distinct decreasing diagonal
      } else if (i == j + 1) {
        v = static_cast<T>(0.5 + 0.1 * j); // first subdiagonal
      } else {
        v = static_cast<T>(0.3 * (j + 1) - 0.15 * (i + 1)); // strict upper
      }
      put(h, i, j, n, v);
    }
  }
  return h;
}

// A KTOP-isolated active block: as make_hess but with H(ktop, ktop-1) == 0 so
// rows/cols ktop..kbot form an isolated diagonal block, as ?laqr5 assumes.
template<typename T>
std::vector<T> make_hess_isolated(int n, int ktop) {
  std::vector<T> h = make_hess<T>(n);
  put(h, ktop - 1, ktop - 2, n, T{0}); // H(ktop, ktop-1) = 0, 0-based
  return h;
}

template<typename T>
void run_all() {
  // Two real shift pairs over the whole 6x6 block, Schur factor only.
  {
    std::vector<T> sr{T{1.5}, T{1.5}, T{-0.5}, T{-0.5}};
    std::vector<T> si{T{0}, T{0}, T{0}, T{0}};
    run_case<T>(/*wantt=*/true, /*wantz=*/false, 6, make_hess<T>(6), 1, 6, sr, si, "real2 noz");
  }
  // Same, accumulating Z.
  {
    std::vector<T> sr{T{1.5}, T{1.5}, T{-0.5}, T{-0.5}};
    std::vector<T> si{T{0}, T{0}, T{0}, T{0}};
    run_case<T>(/*wantt=*/true, /*wantz=*/true, 6, make_hess<T>(6), 1, 6, sr, si, "real2 z");
  }
  // Two complex-conjugate shift pairs, with Z.
  {
    std::vector<T> sr{T{2}, T{2}, T{1}, T{1}};
    std::vector<T> si{T{1}, T{-1}, T{0.5}, T{-0.5}};
    run_case<T>(/*wantt=*/true, /*wantz=*/true, 7, make_hess<T>(7), 1, 7, sr, si, "cplx2 z");
  }
  // Mixed real + complex pair, three bulges, larger block.
  {
    std::vector<T> sr{T{1}, T{1}, T{0.8}, T{0.8}, T{-1.2}, T{-1.2}};
    std::vector<T> si{T{0}, T{0}, T{0.6}, T{-0.6}, T{0}, T{0}};
    run_case<T>(/*wantt=*/true, /*wantz=*/true, 9, make_hess<T>(9), 1, 9, sr, si, "mixed3 z");
  }
  // KTOP > 1 isolated sub-block (ktop=3, kbot=8) in an order-9 matrix.
  {
    std::vector<T> sr{T{1.1}, T{1.1}, T{0.4}, T{0.4}};
    std::vector<T> si{T{0.7}, T{-0.7}, T{0}, T{0}};
    run_case<T>(/*wantt=*/true, /*wantz=*/true, 9, make_hess_isolated<T>(9, 3), 3, 8, sr, si,
                "ktop3 z");
  }
}

} // namespace

TEST(Laqr5OracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(Laqr5OracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

} // namespace calaman
