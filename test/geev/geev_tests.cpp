// Oracle test for calaman.geev: the ?geev driver on the device -- eigenvalues
// and optional left/right eigenvectors of a general real matrix (the full chain
// gebal -> gehrd -> orghr/hseqr -> trevc3 -> back-transform -> gebak ->
// normalize) -- must agree with reference LAPACK ?geev in the SAME precision.
// ?geev HAS a LAPACKE C binding, so the oracle is LAPACKE_sgeev / LAPACKE_dgeev
// with LAPACK_COL_MAJOR, which sizes the reference workspace internally.
//
// WHAT IS CHECKED, and why not element-wise V-vs-reference: neither the Schur
// form nor the eigenvectors are unique. The order the eigenvalues land on the
// diagonal is a rounding-sensitive choice, so device and reference (and float vs
// double, CUDA vs HIP) legitimately produce the same spectrum in a DIFFERENT
// column order -- which makes a per-column V comparison meaningless. And an
// eigenvector is defined only up to a scalar (a sign for a real one, a phase for
// a complex pair). So the oracle (1) compares the EIGENVALUES against the
// reference as a sorted set over [1, n], and (2) validates the device's OWN
// eigenpairs by the scale/sign/phase-INVARIANT residual against the ORIGINAL A:
// for a real right eigenvector v with eigenvalue wr, ||(A - wr*I) v||_inf; for a
// complex pair v = vr + i*vi the coupled A*vr = wr*vr - wi*vi, A*vi = wi*vr +
// wr*vi; left eigenvectors satisfy the transposed relations. That is the
// hseqr/trevc3 suites' philosophy (check the invariant, not the non-unique
// factor), backend-stable, and still pins the device to the reference spectrum.
//
// The cases walk general real matrices with well-separated eigenvalues (a graded
// diagonal dominating modest off-diagonals): clustered magnitudes would leave a
// deflation/ordering decision on a float knife-edge. They cover purely real and
// complex-conjugate spectra, JOBVL/JOBVR in {N,V} (all four combinations), a
// gebal-isolated eigenvalue (ilo/ihi != 1/n), the n > NMIN = 75 laqr0 dispatch,
// and a tiny-norm float matrix that drives ?geev's lange/lascl scaling branch.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A on the device and runs
// the whole kernel chain, so the suite is excluded by `ctest -LE gpu`. Built
// only when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cmath>
#include <vector>

import std;

import wwr.runtime_api;
import wwr.blas;
import wwr.solver;
import wwr.extension.memory_buffer;
import calaman.geev;
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

// Reference ?geev through LAPACKE (column-major), same precision as the device.
int ref_geev(char jvl, char jvr, int n, float *a, int lda, float *wr, float *wi, float *vl,
             int ldvl, float *vr, int ldvr) {
  return LAPACKE_sgeev(LAPACK_COL_MAJOR, jvl, jvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}
int ref_geev(char jvl, char jvr, int n, double *a, int lda, double *wr, double *wi, double *vl,
             int ldvl, double *vr, int ldvr) {
  return LAPACKE_dgeev(LAPACK_COL_MAJOR, jvl, jvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}

// Column-major (i,j) index into a matrix with leading dimension ld, 0-based.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// A relative tolerance scaled by ||A||. geev is a LONG chain (balance, Hessenberg
// reduction, QR iteration, back-substitution, back-transform, unbalance), so
// rounding accumulates far more than a single routine; the large constant
// matches the hseqr suite, with a float bump for the extra float cancellation.
template<typename T>
T tol(T anorm) {
  const T c = std::is_same_v<T, float> ? T{262144} : T{65536};
  return c * eps<T>() * (anorm + T{1});
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

// Per-column block tag from the returned (wr, wi): ip = 0 for a real eigenvalue,
// +1 for the first column of a complex pair, -1 for its second. geev writes
// wi == 0 exactly for real eigenvalues and +/-im for a pair, as ?lanv2 does.
template<typename T>
std::vector<int> block_tags(const std::vector<T> &wi, int n) {
  std::vector<int> ip(static_cast<std::size_t>(n), 0);
  for (int i = 0; i < n;) {
    if (wi[static_cast<std::size_t>(i)] == T{0} || i + 1 >= n) {
      ip[static_cast<std::size_t>(i)] = 0;
      i += 1;
    } else {
      ip[static_cast<std::size_t>(i)] = 1;
      ip[static_cast<std::size_t>(i + 1)] = -1;
      i += 2;
    }
  }
  return ip;
}

// Right-eigenvector residual against the ORIGINAL A: for a real column j,
// ||(A - wr*I) v||_inf; for a complex pair (j, j+1) with v = vr + i*vi, the max
// of ||A vr - wr vr + wi vi|| and ||A vi - wi vr - wr vi||.
template<typename T>
void check_right(int n, const std::vector<T> &a, int ld, const std::vector<T> &vr,
                 const std::vector<T> &wr, const std::vector<T> &wi, const std::vector<int> &ip,
                 T anorm, const char *ctx) {
  const T tl = tol(anorm);
  for (int j = 0; j < n;) {
    if (ip[static_cast<std::size_t>(j)] == 0) {
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T acc = -wr[static_cast<std::size_t>(j)] * vr[idx(i, j, ld)];
        for (int k = 0; k < n; ++k) {
          acc += a[idx(i, k, ld)] * vr[idx(k, j, ld)];
        }
        r = std::max(r, std::abs(acc));
      }
      EXPECT_LE(r, tl) << ctx << " right real residual col " << j;
      j += 1;
    } else {
      const int jr = j, ji = j + 1;
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T accr = -wr[static_cast<std::size_t>(jr)] * vr[idx(i, jr, ld)] +
                 wi[static_cast<std::size_t>(jr)] * vr[idx(i, ji, ld)];
        T acci = -wi[static_cast<std::size_t>(jr)] * vr[idx(i, jr, ld)] -
                 wr[static_cast<std::size_t>(jr)] * vr[idx(i, ji, ld)];
        for (int k = 0; k < n; ++k) {
          accr += a[idx(i, k, ld)] * vr[idx(k, jr, ld)];
          acci += a[idx(i, k, ld)] * vr[idx(k, ji, ld)];
        }
        r = std::max(r, std::max(std::abs(accr), std::abs(acci)));
      }
      EXPECT_LE(r, tl) << ctx << " right complex residual cols " << jr << "," << ji;
      j += 2;
    }
  }
}

// Left-eigenvector residual against the ORIGINAL A: u^H A = w u^H, i.e.
// A^T ur = wr ur + wi ui and A^T ui = wr ui - wi ur for the pair at columns
// (jr real, ji imag) with eigenvalue wr + i*wi; (A^T - wr) u for the real case.
template<typename T>
void check_left(int n, const std::vector<T> &a, int ld, const std::vector<T> &vl,
                const std::vector<T> &wr, const std::vector<T> &wi, const std::vector<int> &ip,
                T anorm, const char *ctx) {
  const T tl = tol(anorm);
  for (int j = 0; j < n;) {
    if (ip[static_cast<std::size_t>(j)] == 0) {
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T acc = -wr[static_cast<std::size_t>(j)] * vl[idx(i, j, ld)];
        for (int k = 0; k < n; ++k) {
          acc += a[idx(k, i, ld)] * vl[idx(k, j, ld)]; // (A^T)_{ik} = A_{ki}
        }
        r = std::max(r, std::abs(acc));
      }
      EXPECT_LE(r, tl) << ctx << " left real residual col " << j;
      j += 1;
    } else {
      const int jr = j, ji = j + 1;
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T accr = -wr[static_cast<std::size_t>(jr)] * vl[idx(i, jr, ld)] -
                 wi[static_cast<std::size_t>(jr)] * vl[idx(i, ji, ld)];
        T acci = wi[static_cast<std::size_t>(jr)] * vl[idx(i, jr, ld)] -
                 wr[static_cast<std::size_t>(jr)] * vl[idx(i, ji, ld)];
        for (int k = 0; k < n; ++k) {
          accr += a[idx(k, i, ld)] * vl[idx(k, jr, ld)];
          acci += a[idx(k, i, ld)] * vl[idx(k, ji, ld)];
        }
        r = std::max(r, std::max(std::abs(accr), std::abs(acci)));
      }
      EXPECT_LE(r, tl) << ctx << " left complex residual cols " << jr << "," << ji;
      j += 2;
    }
  }
}

// One case: geev on the device must match the reference ?geev. a0 is an n-by-n
// column-major real matrix; wantvl/wantvr pick the sides.
template<typename T>
void run_case(bool wantvl, bool wantvr, int n, const std::vector<T> &a0, const char *ctx) {
  const int ld = n;
  const char jvl = wantvl ? 'V' : 'N';
  const char jvr = wantvr ? 'V' : 'N';

  T anorm = T{1};
  for (const T e : a0) {
    anorm = std::max(anorm, std::abs(e));
  }

  // Reference overwrites a; vl/vr sized n-by-n for both sides regardless.
  std::vector<T> r_a = a0;
  std::vector<T> r_wr(static_cast<std::size_t>(n), T{0}), r_wi(static_cast<std::size_t>(n), T{0});
  std::vector<T> r_vl(static_cast<std::size_t>(n) * n, T{0});
  std::vector<T> r_vr(static_cast<std::size_t>(n) * n, T{0});
  const int r_info =
      ref_geev(jvl, jvr, n, r_a.data(), ld, r_wr.data(), r_wi.data(), r_vl.data(), ld, r_vr.data(),
               ld);

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS) << ctx;
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS)
      << ctx;
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS) << ctx;
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()), wwr::WWRSOLVER_STATUS_SUCCESS)
      << ctx;

  const GeevVectors ejvl = wantvl ? GeevVectors::Vectors : GeevVectors::None;
  const GeevVectors ejvr = wantvr ? GeevVectors::Vectors : GeevVectors::None;

  auto d_a = to_device(handle, a0);
  auto d_wr = to_device(handle, std::vector<T>(static_cast<std::size_t>(n), T{0}));
  auto d_wi = to_device(handle, std::vector<T>(static_cast<std::size_t>(n), T{0}));
  auto d_vl = to_device(handle, std::vector<T>(static_cast<std::size_t>(n) * n, T{0}));
  auto d_vr = to_device(handle, std::vector<T>(static_cast<std::size_t>(n) * n, T{0}));
  auto d_info = to_device(handle, std::vector<int>{-123});

  const std::size_t bytes = geev_bufferSize<T>(solver, n, ejvl, ejvr);
  DeviceBuffer<std::byte> d_work(bytes == 0 ? 1 : bytes, handle);

  const auto status =
      geev<T>(blas, solver, ejvl, ejvr, n, d_a.data(), ld, d_wr.data(), d_wi.data(), d_vl.data(),
              ld, d_vr.data(), ld, d_work.data(), bytes, d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  wwr::wwrsolverDnDestroy(solver);
  wwr::wwrblasDestroy(blas);

  const auto g_info = from_device(handle, d_info, 1)[0];
  EXPECT_EQ(g_info, 0) << ctx << " info";
  EXPECT_EQ(g_info, r_info) << ctx << " info vs ref";

  const auto g_wr = from_device(handle, d_wr, static_cast<std::size_t>(n));
  const auto g_wi = from_device(handle, d_wi, static_cast<std::size_t>(n));

  // 1) Eigenvalues vs the reference, as a sorted set over all of [1, n].
  const auto ge = sorted_eigs(g_wr, g_wi, n);
  const auto re = sorted_eigs(r_wr, r_wi, n);
  ASSERT_EQ(ge.size(), re.size()) << ctx;
  for (std::size_t i = 0; i < ge.size(); ++i) {
    EXPECT_NEAR(ge[i].first, re[i].first, tol(anorm)) << ctx << " eig.re(" << i << ")";
    EXPECT_NEAR(ge[i].second, re[i].second, tol(anorm)) << ctx << " eig.im(" << i << ")";
  }

  // 2) The device's OWN eigenpairs are valid: residual against the original A.
  const auto ip = block_tags(g_wi, n);
  if (wantvr) {
    const auto g_vr = from_device(handle, d_vr, static_cast<std::size_t>(n) * n);
    check_right<T>(n, a0, ld, g_vr, g_wr, g_wi, ip, anorm, ctx);
  }
  if (wantvl) {
    const auto g_vl = from_device(handle, d_vl, static_cast<std::size_t>(n) * n);
    check_left<T>(n, a0, ld, g_vl, g_wr, g_wi, ip, anorm, ctx);
  }
}

template<typename T>
void put(std::vector<T> &a, int i, int j, int ld, T v) {
  a[idx(i, j, ld)] = v;
}

// A general real matrix with a WELL-SEPARATED real spectrum: a strongly graded
// diagonal (geometric, ratio 1.2) dominating modest, decaying off-diagonals, so
// the matrix is diagonally dominant and the eigenvalues are unambiguous in both
// float and double.
template<typename T>
std::vector<T> make_real(int n) {
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (i == j) {
        put(a, i, j, n, static_cast<T>(2.0 * std::pow(1.2, i)));
      } else {
        put(a, i, j, n, static_cast<T>(0.05 / (std::abs(i - j) + 1.0)));
      }
    }
  }
  return a;
}

// As make_real, but every other 2x2 diagonal block carries a complex-conjugate
// pair: equal diagonals d within the block and off-diagonals +/-c give d +/- i*c.
template<typename T>
std::vector<T> make_complex(int n) {
  std::vector<T> a = make_real<T>(n);
  for (int i = 0; i + 1 < n; i += 2) {
    const T d = static_cast<T>(2.0 * std::pow(1.2, i));
    put(a, i, i, n, d);
    put(a, i + 1, i + 1, n, d);                         // equal diagonals
    put(a, i, i + 1, n, static_cast<T>(0.3) * d);       // +c
    put(a, i + 1, i, n, static_cast<T>(-0.3) * d);      // -c  => d +/- i*(0.3 d)
  }
  return a;
}

// make_real with an eigenvalue gebal isolates: zero the whole last row except the
// diagonal, so A(n, 1:n-1) = 0 and row n isolates eigenvalue A(n,n) -- gebal
// permutes it out, leaving ihi < n (and the driver copies it off the diagonal).
template<typename T>
std::vector<T> make_isolated(int n) {
  std::vector<T> a = make_real<T>(n);
  for (int k = 0; k < n - 1; ++k) {
    put(a, n - 1, k, n, T{0}); // last row, off-diagonal, zeroed
  }
  return a;
}

// A large, well-separated real spectrum for the laqr0 dispatch (n > NMIN = 75).
// The geometric grading overflows at large n, so this uses an ARITHMETIC diagonal
// (1..n) with weak, decaying off-diagonals: real, bounded (||A|| ~ n), separated.
template<typename T>
std::vector<T> make_real_arith(int n) {
  std::vector<T> a(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (i == j) {
        put(a, i, j, n, static_cast<T>(i + 1));
      } else {
        put(a, i, j, n, static_cast<T>(0.02 / (std::abs(i - j) + 1.0)));
      }
    }
  }
  return a;
}

template<typename T>
void run_all() {
  // All four JOBVL/JOBVR combinations on a well-separated real spectrum.
  run_case<T>(false, false, 16, make_real<T>(16), "real16 N/N");
  run_case<T>(true, false, 16, make_real<T>(16), "real16 V/N");
  run_case<T>(false, true, 16, make_real<T>(16), "real16 N/V");
  run_case<T>(true, true, 16, make_real<T>(16), "real16 V/V");
  // Complex-conjugate spectrum: left, right, both.
  run_case<T>(true, false, 16, make_complex<T>(16), "cplx16 V/N");
  run_case<T>(false, true, 16, make_complex<T>(16), "cplx16 N/V");
  run_case<T>(true, true, 16, make_complex<T>(16), "cplx16 V/V");
  // A single 2x2 block: a pure complex-conjugate pair, both sides.
  run_case<T>(true, true, 2, make_complex<T>(2), "cplx2 V/V");
  // A gebal-isolated eigenvalue (ihi < n), both sides.
  run_case<T>(true, true, 10, make_isolated<T>(10), "iso10 V/V");
  // Eigenvalues only, a larger real spectrum.
  run_case<T>(false, false, 24, make_real<T>(24), "real24 N/N");
}

} // namespace

TEST(GeevOracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(GeevOracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

// The n > NMIN = 75 laqr0 dispatch inside ?hseqr. N = 100 with an arithmetic
// diagonal keeps the spectrum real, bounded and well separated; double only (the
// gap-to-tolerance margin). Both sides, so the back-transform runs at scale too.
TEST(GeevOracleTests, LaqrDispatchDouble) {
  run_case<double>(true, true, 100, make_real_arith<double>(100), "laqr0-dispatch n100");
}

// ?geev's lange/lascl scaling branch: a tiny-norm float matrix (max element well
// below SMLNUM = sqrt(tiny)/eps ~ 9e-13 for float) is scaled UP before balancing
// and its eigenvalues scaled back. Eigenvalues only -- the branch is about the
// scalar scaling, not the vectors.
TEST(GeevOracleTests, ScalingBranchFloat) {
  std::vector<float> a = make_real<float>(12);
  for (float &e : a) {
    e *= 1e-18f; // ||A|| ~ 2e-18 < SMLNUM: triggers SCALEA / CSCALE = SMLNUM
  }
  run_case<float>(false, false, 12, a, "tiny-norm scaling");
}

} // namespace calaman
