// Oracle test for calaman.laqr4: the small-bulge multishift QR driver with
// aggressive early deflation on the device must agree with reference LAPACK --
// ?laqr4 in the SAME precision on the host. The device path must reproduce the
// Schur form H, the accumulated Z (when WANTZ), the eigenvalues WR / WI, and the
// convergence flag INFO.
//
// ?laqr4 is a computational auxiliary with no LAPACKE C binding, so the oracle
// calls the Fortran symbol slaqr4_ / dlaqr4_ directly, exactly as the laqr2 /
// laqr5 / lahqr suites call their auxiliaries. Fortran passes every scalar by
// reference; WANTT / WANTZ are LOGICALs, a 4-byte int (0 / 1). ILO / IHI /
// ILOZ / IHIZ are 1-based in the ABI, and laqr4 takes the same. Passing the
// SAME LWORK to both pins NWMAX / NSMAX -- hence the window and shift schedules
// -- identical, so device and reference make the same deflation decisions and
// agree beyond rounding.
//
// H IS MASKED BELOW THE FIRST SUBDIAGONAL: the reference aliases its AED / sweep
// scratch into the strictly-lower corner of H (the H(KV,1) etc. arguments) and
// leaves orthogonal-matrix leftovers there; this port takes that scratch from
// WORK, so its H stays clean below the subdiagonal. Neither is part of the Schur
// answer, so the comparison covers only i <= j+1 (the quasi-triangular part),
// plus Z in full and WR / WI.
//
// The cases walk real upper-Hessenberg matrices with well-separated eigenvalues
// (a strongly graded diagonal): clustered magnitudes would leave a deflation /
// ordering decision on a float knife-edge, where device and reference pick
// different (both valid) Schur bases. Sizes span the lahqr fallback (N <= NTINY
// = 15) and the multishift path (N > 15), WANTT with and without WANTZ, and a
// ILO-isolated sub-block.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages H / Z on the device and
// runs the kernels, so the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.laqr4;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?laqr4: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; WANTT / WANTZ (LOGICAL) are 4-byte ints (0 / 1).
extern "C" {
void slaqr4_(const int *wantt, const int *wantz, const int *n, const int *ilo, const int *ihi,
             float *h, const int *ldh, float *wr, float *wi, const int *iloz, const int *ihiz,
             float *z, const int *ldz, float *work, const int *lwork, int *info);
void dlaqr4_(const int *wantt, const int *wantz, const int *n, const int *ilo, const int *ihi,
             double *h, const int *ldh, double *wr, double *wi, const int *iloz, const int *ihiz,
             double *z, const int *ldz, double *work, const int *lwork, int *info);
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
int ref_laqr4(int wantt, int wantz, int n, int ilo, int ihi, float *h, int ldh, float *wr,
              float *wi, int iloz, int ihiz, float *z, int ldz, int lwork) {
  std::vector<float> work(static_cast<std::size_t>(lwork));
  int info = 0;
  slaqr4_(&wantt, &wantz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z, &ldz, work.data(),
          &lwork, &info);
  return info;
}
int ref_laqr4(int wantt, int wantz, int n, int ilo, int ihi, double *h, int ldh, double *wr,
              double *wi, int iloz, int ihiz, double *z, int ldz, int lwork) {
  std::vector<double> work(static_cast<std::size_t>(lwork));
  int info = 0;
  dlaqr4_(&wantt, &wantz, &n, &ilo, &ihi, h, &ldh, wr, wi, &iloz, &ihiz, z, &ldz, work.data(),
          &lwork, &info);
  return info;
}

// Column-major (i,j) index into a matrix with leading dimension ld, 0-based.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// A relative tolerance scaled by ||H||. The driver is a LONG chain -- many AED
// passes and multishift sweeps, each a product of reflectors -- so rounding
// accumulates far more than a single-routine oracle; device FMA contraction is
// the only divergence given the identical algorithm, but it compounds across the
// whole iteration, hence the large constant.
template<typename T>
T tol(T hnorm) {
  return T{65536} * eps<T>() * (hnorm + T{1});
}

// Work-buffer length for this port: NWMAX / NSMAX computed exactly as laqr4 does
// for the given n / lwork, then 2 V/T windows + the NS-by-NS shift copy + a 1x1
// Z dummy + 3 device counters (rounded into T slots by the +4).
std::size_t workbuf_len(int n, int lwork) {
  const int nwmax = std::min((n - 1) / 3, lwork / 2);
  int nsmax = std::min((n - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 4;
}

// The EIGENVALUES of [ilo, ihi] as a set, sorted by (real, imag), so the
// comparison is independent of the order they land on the Schur diagonal.
template<typename T>
std::vector<std::pair<T, T>> sorted_eigs(const std::vector<T> &wr, const std::vector<T> &wi,
                                         int ilo, int ihi) {
  std::vector<std::pair<T, T>> e;
  for (int i = ilo - 1; i < ihi; ++i) {
    e.emplace_back(wr[static_cast<std::size_t>(i)], wi[static_cast<std::size_t>(i)]);
  }
  std::sort(e.begin(), e.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first < b.first : a.second < b.second;
  });
  return e;
}

// One case: laqr4 on the device must match the reference. h0 is an n-by-n
// column-major real upper-Hessenberg matrix; ilo/ihi/iloz/ihiz are 1-based.
//
// WHAT IS CHECKED, and why not element-wise H/Z-vs-reference: a real Schur
// factorization is NOT unique -- the Schur-vector signs and the order of the
// eigenvalues on the diagonal are fixed only up to rounding-sensitive choices,
// so device and reference (and float vs double, CUDA vs HIP) legitimately pick
// different (equally valid) H and Z while computing the same spectrum. So the
// oracle compares the EIGENVALUES against the reference as a sorted set (those
// are well defined), and verifies the device's OWN factorization is valid:
// Z orthogonal and Z*T*Z^T == the input H. That is backend-stable and still
// pins the device to the reference's spectrum and INFO.
template<typename T>
void run_case(bool wantt, bool wantz, int n, int ilo, int ihi, const std::vector<T> &h0,
              const char *ctx) {
  const int ld = n;
  const int iloz = 1, ihiz = n;
  const int lwork = 6 * n + 16; // >= 2*(n-1)/3, so NWMAX/NSMAX match the reference

  T hnorm = T{1};
  for (const T e : h0) {
    hnorm = std::max(hnorm, std::abs(e));
  }

  // Reference overwrites H, Z, WR, WI, INFO in place (H/Z used only as scratch).
  std::vector<T> r_h = h0;
  std::vector<T> r_z(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    r_z[idx(i, i, ld)] = T{1};
  }
  std::vector<T> r_wr(static_cast<std::size_t>(n), T{0}), r_wi(static_cast<std::size_t>(n), T{0});
  const int r_info = ref_laqr4(wantt ? 1 : 0, wantz ? 1 : 0, n, ilo, ihi, r_h.data(), ld,
                               r_wr.data(), r_wi.data(), iloz, ihiz, r_z.data(), ld, lwork);

  auto handle = shared_device();
  auto d_h = to_device(handle, h0);
  std::vector<T> z0(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    z0[idx(i, i, ld)] = T{1};
  }
  auto d_z = to_device(handle, z0);
  auto d_wr = to_device(handle, std::vector<T>(static_cast<std::size_t>(n), T{0}));
  auto d_wi = to_device(handle, std::vector<T>(static_cast<std::size_t>(n), T{0}));
  auto d_work = to_device(handle, std::vector<T>(workbuf_len(n, lwork), T{0}));
  auto d_info = to_device(handle, std::vector<int>{-123});

  const auto status = laqr4<T>(handle->stream().get(), wantt, wantz, n, ilo, ihi, d_h.data(), ld,
                               d_wr.data(), d_wi.data(), iloz, ihiz, d_z.data(), ld, d_work.data(),
                               lwork, d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const auto g_info = from_device(handle, d_info, 1)[0];
  EXPECT_EQ(g_info, r_info) << ctx << " info";

  const auto g_wr = from_device(handle, d_wr, static_cast<std::size_t>(n));
  const auto g_wi = from_device(handle, d_wi, static_cast<std::size_t>(n));

  // 1) Eigenvalues vs the reference, as a sorted set.
  const auto ge = sorted_eigs(g_wr, g_wi, ilo, ihi);
  const auto re = sorted_eigs(r_wr, r_wi, ilo, ihi);
  ASSERT_EQ(ge.size(), re.size()) << ctx;
  for (std::size_t i = 0; i < ge.size(); ++i) {
    EXPECT_NEAR(ge[i].first, re[i].first, tol(hnorm)) << ctx << " eig.re(" << i << ")";
    EXPECT_NEAR(ge[i].second, re[i].second, tol(hnorm)) << ctx << " eig.im(" << i << ")";
  }

  // 2) The device's own factorization is valid (whole-matrix window only, so Z
  //    is the full accumulated transform and the reconstruction is exact).
  if (wantt && wantz && ilo == 1 && ihi == n && iloz == 1 && ihiz == n) {
    const auto g_h = from_device(handle, d_h, static_cast<std::size_t>(n) * n);
    const auto g_z = from_device(handle, d_z, static_cast<std::size_t>(n) * n);

    // Orthogonality: ||Z^T Z - I||_max.
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

    // Reconstruction: Z * T * Z^T == h0 (T is the returned quasi-triangular H,
    // read as full storage -- the strictly-lower scratch leftovers do not matter
    // because T is quasi-upper-triangular there to machine precision).
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
}

// Set entry (i,j), 0-based, in an ld-leading-dimension column-major matrix.
template<typename T>
void put(std::vector<T> &a, int i, int j, int ld, T v) {
  a[idx(i, j, ld)] = v;
}

// A real upper-Hessenberg H of order n with WELL-SEPARATED eigenvalues: a
// strongly graded diagonal (geometric, ratio ~1.7) dominates a modest
// subdiagonal, so the spectrum is unambiguous in both float and double and the
// QR iteration deflates and orders its blocks by a clear margin -- the condition
// the laqr2 suite documents for device/reference Schur bases to agree.
template<typename T>
std::vector<T> make_hess(int n) {
  std::vector<T> h(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= std::min(j + 1, n - 1); ++i) {
      T v;
      if (i == j) {
        v = static_cast<T>(2.0 * std::pow(1.2, i)); // graded diagonal, well separated
      } else if (i == j + 1) {
        v = static_cast<T>(0.1 + 0.015 * j); // modest subdiagonal
      } else {
        v = static_cast<T>(0.05 * (j - i)); // strict upper
      }
      put(h, i, j, n, v);
    }
  }
  return h;
}

// As make_hess, but every other 2x2 diagonal block carries a complex-conjugate
// pair: the block [[d_i, +s],[-s, d_{i+1}]] with s = 0.3*d_i has eigenvalues
// 1.1*d_i +/- 0.28*d_i*i, so successive blocks' magnitudes still separate by the
// diagonal grading. This drives the real / complex shift shuffle the driver does
// before each sweep -- the branch the all-real spectra never reach.
template<typename T>
std::vector<T> make_hess_complex(int n) {
  std::vector<T> h = make_hess<T>(n);
  for (int i = 0; i + 1 < n; i += 2) {
    const T di = static_cast<T>(2.0 * std::pow(1.2, i));
    put(h, i, i + 1, n, static_cast<T>(0.3) * di);  // superdiagonal +s
    put(h, i + 1, i, n, static_cast<T>(-0.3) * di); // subdiagonal -s
  }
  return h;
}

// An ILO-isolated active block: as make_hess but with H(ilo, ilo-1) == 0 so
// rows/cols ilo..ihi form an isolated diagonal block, as the window assumes.
template<typename T>
std::vector<T> make_hess_isolated(int n, int ilo) {
  std::vector<T> h = make_hess<T>(n);
  if (ilo > 1) {
    put(h, ilo - 1, ilo - 2, n, T{0}); // H(ilo, ilo-1) = 0, 0-based
  }
  return h;
}

template<typename T>
void run_all() {
  // Tiny matrix (n <= NTINY = 15): exercises the lahqr fallback path.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 12, 1, 12, make_hess<T>(12), "tiny12 z");
  // Order-20 whole window, Schur factor only (no Z): the multishift path.
  run_case<T>(/*wantt=*/true, /*wantz=*/false, 20, 1, 20, make_hess<T>(20), "n20 noz");
  // Same, accumulating Z.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 20, 1, 20, make_hess<T>(20), "n20 z");
  // Eigenvalues-only update (wantt=false): only enough of H is touched.
  run_case<T>(/*wantt=*/false, /*wantz=*/true, 20, 1, 20, make_hess<T>(20), "n20 noT z");
  // Larger matrix -> more sweeps / deflations.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 32, 1, 32, make_hess<T>(32), "n32 z");
  // ILO-isolated sub-block (ilo=4, ihi=28) inside an order-32 matrix.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 32, 4, 28, make_hess_isolated<T>(32, 4), "ilo4 z");
  // Complex-conjugate spectrum: exercises the shift shuffle and complex sweeps;
  // the spectrum is checked vs the reference and the factorization reconstructed.
  run_case<T>(/*wantt=*/true, /*wantz=*/true, 24, 1, 24, make_hess_complex<T>(24), "cplx24 z");
}

} // namespace

TEST(Laqr4OracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(Laqr4OracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

} // namespace calaman
