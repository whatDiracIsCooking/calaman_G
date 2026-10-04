// Oracle test for calaman.hseqr: the ?hseqr driver on the device -- eigenvalues
// or the full Schur form of an upper Hessenberg matrix, dispatching lahqr vs
// laqr0 at the NMIN = 75 crossover -- must agree with reference LAPACK ?hseqr in
// the SAME precision. ?hseqr HAS a LAPACKE C binding (unlike the ?laqr* / ?lahqr
// auxiliaries its suites call by Fortran symbol), so the oracle is
// LAPACKE_shseqr / LAPACKE_dhseqr with LAPACK_COL_MAJOR -- which also sizes the
// reference workspace internally.
//
// WHAT IS CHECKED, and why not element-wise H/Z-vs-reference: a real Schur
// factorization is NOT unique -- the Schur-vector signs and the order of the
// eigenvalues on the diagonal are fixed only up to rounding-sensitive choices,
// so device and reference (and float vs double, CUDA vs HIP) legitimately pick
// different (equally valid) H and Z while computing the same spectrum. So the
// oracle compares the EIGENVALUES against the reference as a sorted set over the
// whole [1, n] (both the window ?hseqr factors and the entries it copies off the
// diagonal outside it), checks INFO, and -- for the full Schur form with vectors
// on the whole window -- verifies the device's OWN factorization is valid:
// Z orthogonal and Z*T*Z^T == the input H. That is backend-stable and still pins
// the device to the reference's spectrum.
//
// The cases walk real upper-Hessenberg matrices with well-separated eigenvalues
// (a strongly graded diagonal): clustered magnitudes would leave a deflation /
// ordering decision on a float knife-edge. Sizes span the lahqr path (n <= NMIN
// = 75) and the laqr0 path (n > NMIN), JOB in {Schur, Eigenvalues}, COMPZ in
// {Identity, Vectors, None}, and an ILO-isolated sub-block (whose outside
// eigenvalues must be copied off the diagonal).
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages H / Z on the device and
// runs the dispatched kernels, so the suite is excluded by `ctest -LE gpu`.
// Built only when calaman::lapack_reference exists; its CMakeLists.txt returns
// early otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.hseqr;
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

// Reference ?hseqr through LAPACKE (column-major), same precision as the device.
int ref_hseqr(char job, char compz, int n, int ilo, int ihi, float *h, int ldh, float *wr,
              float *wi, float *z, int ldz) {
  return LAPACKE_shseqr(LAPACK_COL_MAJOR, job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz);
}
int ref_hseqr(char job, char compz, int n, int ilo, int ihi, double *h, int ldh, double *wr,
              double *wi, double *z, int ldz) {
  return LAPACKE_dhseqr(LAPACK_COL_MAJOR, job, compz, n, ilo, ihi, h, ldh, wr, wi, z, ldz);
}

// Column-major (i,j) index into a matrix with leading dimension ld, 0-based.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// A relative tolerance scaled by ||H||. hseqr is a LONG chain -- many QR sweeps,
// each a product of reflectors -- so rounding accumulates far more than a
// single-routine oracle; device FMA contraction is the only divergence given the
// identical algorithm, but it compounds across the whole iteration, hence the
// large constant (matching the laqr0 suite).
template<typename T>
T tol(T hnorm) {
  return T{65536} * eps<T>() * (hnorm + T{1});
}

// laqr0's work-buffer carve, which hseqr forwards unchanged: 2 V/T windows + the
// NS-by-NS shift copy + a 1x1 Z dummy + the recursive-laqr4 scratch + 3 device
// counters. hseqr never reads it on the lahqr path, but always takes it.
std::size_t laqr4_workbuf_len(int sub, int lwork) {
  const int nwmax = std::min((sub - 1) / 3, lwork / 2);
  int nsmax = std::min((sub - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 4;
}
std::size_t workbuf_len(int n, int lwork) {
  const int nwmax = std::min((n - 1) / 3, lwork / 2);
  int nsmax = std::min((n - 3) / 6, 2 * lwork / 3);
  nsmax -= nsmax % 2;
  const int ldv = std::max(1, nwmax);
  const int ldsc = std::max(1, nsmax);
  return 2 * static_cast<std::size_t>(ldv) * std::max(1, nwmax) +
         static_cast<std::size_t>(ldsc) * std::max(1, nsmax) + 1 + laqr4_workbuf_len(nwmax, lwork) +
         4;
}

// The full spectrum [1, n] as a set, sorted by (real, imag) so the comparison is
// independent of the order the eigenvalues land on the diagonal.
template<typename T>
std::vector<std::pair<T, T>> sorted_eigs(const std::vector<T> &wr, const std::vector<T> &wi,
                                         int n) {
  std::vector<std::pair<T, T>> e;
  for (int i = 0; i < n; ++i) {
    e.emplace_back(wr[static_cast<std::size_t>(i)], wi[static_cast<std::size_t>(i)]);
  }
  std::sort(e.begin(), e.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first < b.first : a.second < b.second;
  });
  return e;
}

// One case: hseqr on the device must match the reference ?hseqr. h0 is an n-by-n
// column-major real upper-Hessenberg matrix; ilo/ihi are 1-based. `recon` turns
// off the O(n^4) reconstruction for the large case (orthogonality, O(n^3), and
// the eigenvalue-set match still run).
template<typename T>
void run_case(HseqrJob job, HseqrCompz compz, int n, int ilo, int ihi, const std::vector<T> &h0,
              const char *ctx, bool recon = true) {
  const int ld = n;
  const int lwork = 6 * n + 16; // >= 2*(n-1)/3, so laqr0's NWMAX/NSMAX match
  const char rjob = (job == HseqrJob::Schur) ? 'S' : 'E';
  const char rcompz =
      (compz == HseqrCompz::Identity) ? 'I' : (compz == HseqrCompz::Vectors ? 'V' : 'N');
  const bool wantt = (job == HseqrJob::Schur);
  const bool wantz = (compz != HseqrCompz::None);

  T hnorm = T{1};
  for (const T e : h0) {
    hnorm = std::max(hnorm, std::abs(e));
  }

  // Z input: identity for COMPZ='V' (accumulate onto it), otherwise ignored
  // ('I' overwrites it, 'N' does not touch it).
  std::vector<T> z0(static_cast<std::size_t>(n) * n, T{0});
  if (compz == HseqrCompz::Vectors) {
    for (int i = 0; i < n; ++i) {
      z0[idx(i, i, ld)] = T{1};
    }
  }

  // Reference overwrites H, Z, WR, WI in place.
  std::vector<T> r_h = h0;
  std::vector<T> r_z = z0;
  std::vector<T> r_wr(static_cast<std::size_t>(n), T{0}), r_wi(static_cast<std::size_t>(n), T{0});
  const int r_info = ref_hseqr(rjob, rcompz, n, ilo, ihi, r_h.data(), ld, r_wr.data(), r_wi.data(),
                               r_z.data(), ld);

  auto handle = shared_device();
  auto d_h = to_device(handle, h0);
  auto d_z = to_device(handle, z0);
  auto d_wr = to_device(handle, std::vector<T>(static_cast<std::size_t>(n), T{0}));
  auto d_wi = to_device(handle, std::vector<T>(static_cast<std::size_t>(n), T{0}));
  auto d_work = to_device(handle, std::vector<T>(workbuf_len(n, lwork), T{0}));
  auto d_info = to_device(handle, std::vector<int>{-123});

  const auto status =
      hseqr<T>(handle->stream().get(), job, compz, n, ilo, ihi, d_h.data(), ld, d_wr.data(),
               d_wi.data(), d_z.data(), ld, d_work.data(), lwork, d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const auto g_info = from_device(handle, d_info, 1)[0];
  EXPECT_EQ(g_info, r_info) << ctx << " info";

  const auto g_wr = from_device(handle, d_wr, static_cast<std::size_t>(n));
  const auto g_wi = from_device(handle, d_wi, static_cast<std::size_t>(n));

  // 1) Eigenvalues vs the reference, as a sorted set over all of [1, n].
  const auto ge = sorted_eigs(g_wr, g_wi, n);
  const auto re = sorted_eigs(r_wr, r_wi, n);
  ASSERT_EQ(ge.size(), re.size()) << ctx;
  for (std::size_t i = 0; i < ge.size(); ++i) {
    EXPECT_NEAR(ge[i].first, re[i].first, tol(hnorm)) << ctx << " eig.re(" << i << ")";
    EXPECT_NEAR(ge[i].second, re[i].second, tol(hnorm)) << ctx << " eig.im(" << i << ")";
  }

  // 2) The device's own factorization is valid (full Schur form with vectors on
  //    the whole window: Z is the full accumulated transform, reconstruction exact).
  if (wantt && wantz && ilo == 1 && ihi == n) {
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

    // Reconstruction: Z * T * Z^T == h0 (T is the returned quasi-triangular H).
    if (recon) {
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
}

// Set entry (i,j), 0-based, in an ld-leading-dimension column-major matrix.
template<typename T>
void put(std::vector<T> &a, int i, int j, int ld, T v) {
  a[idx(i, j, ld)] = v;
}

// A real upper-Hessenberg H of order n with WELL-SEPARATED eigenvalues: a
// strongly graded diagonal (geometric, ratio 1.2) dominates a modest
// subdiagonal, so the spectrum is unambiguous in both float and double.
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
// pair, driving the real / complex shift shuffle the all-real spectra never reach.
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
// rows/cols ilo..ihi form an isolated diagonal block; the eigenvalues outside
// [ilo, ihi] must be copied off the diagonal by the driver.
template<typename T>
std::vector<T> make_hess_isolated(int n, int ilo, int ihi) {
  std::vector<T> h = make_hess<T>(n);
  if (ilo > 1) {
    put(h, ilo - 1, ilo - 2, n, T{0}); // H(ilo, ilo-1) = 0, 0-based
  }
  if (ihi < n) {
    put(h, ihi, ihi - 1, n, T{0}); // H(ihi+1, ihi) = 0, 0-based
  }
  return h;
}

// A large, well-separated real spectrum for the laqr0 dispatch (n > NMIN = 75).
// The geometric grading make_hess uses overflows at large n, so this uses an
// ARITHMETIC diagonal (1, 2, ..., n) with gaps of 1 and a weak subdiagonal: the
// eigenvalues stay real, bounded (||H|| ~ n), and separated by a clear margin.
template<typename T>
std::vector<T> make_hess_arith(int n) {
  std::vector<T> h(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= std::min(j + 1, n - 1); ++i) {
      T v;
      if (i == j) {
        v = static_cast<T>(i + 1); // 1, 2, ..., n: gaps of 1, well separated
      } else if (i == j + 1) {
        v = static_cast<T>(0.05); // weak subdiagonal (eigenvalues ~ diagonal)
      } else {
        v = static_cast<T>(0.02 / (j - i + 1)); // decaying strict upper
      }
      put(h, i, j, n, v);
    }
  }
  return h;
}

template<typename T>
void run_all() {
  using J = HseqrJob;
  using C = HseqrCompz;
  // Tiny window (n <= NMIN = 75): the lahqr dispatch.
  run_case<T>(J::Schur, C::Identity, 12, 1, 12, make_hess<T>(12), "tiny12 S/I");
  // Full Schur form, Schur vectors accumulated onto a supplied identity Z.
  run_case<T>(J::Schur, C::Vectors, 20, 1, 20, make_hess<T>(20), "n20 S/V");
  // Full Schur form, no Z (COMPZ='N'): only the eigenvalue set is checked.
  run_case<T>(J::Schur, C::None, 20, 1, 20, make_hess<T>(20), "n20 S/N");
  // Eigenvalues only (JOB='E'), no Z.
  run_case<T>(J::Eigenvalues, C::None, 20, 1, 20, make_hess<T>(20), "n20 E/N");
  // Larger lahqr window.
  run_case<T>(J::Schur, C::Identity, 32, 1, 32, make_hess<T>(32), "n32 S/I");
  // ILO/IHI-isolated sub-block inside an order-40 matrix: the outside
  // eigenvalues come off the diagonal, the window [6, 34] through lahqr.
  run_case<T>(J::Schur, C::Identity, 40, 6, 34, make_hess_isolated<T>(40, 6, 34), "iso40 S/I");
  // Complex-conjugate spectrum: the shift shuffle and complex sweeps.
  run_case<T>(J::Schur, C::Identity, 24, 1, 24, make_hess_complex<T>(24), "cplx24 S/I");
}

} // namespace

TEST(HseqrOracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(HseqrOracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

// The laqr0 dispatch (n > NMIN = 75). N = 120 with an arithmetic diagonal keeps
// the spectrum real, bounded and well separated; double only (the gap-to-
// tolerance margin) and no O(n^4) reconstruction (the O(n^3) orthogonality check
// and the eigenvalue-set match still run).
TEST(HseqrOracleTests, LaqrDispatchDouble) {
  run_case<double>(HseqrJob::Schur, HseqrCompz::Identity, 120, 1, 120, make_hess_arith<double>(120),
                   "laqr0-dispatch n120", /*recon=*/false);
}

} // namespace calaman
