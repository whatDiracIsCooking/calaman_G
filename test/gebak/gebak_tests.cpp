// Oracle suite for calaman.gebak -- backward transformation of eigenvectors.
// Unlike gebal, gebak has NO balancing-convention divergence from a reference
// LAPACK: it simply undoes a scaling and a permutation, so LAPACKE_?gebak is a
// true independent oracle. Each case stages V and scale on the device, runs the
// device gebak, and diffs against LAPACKE_?gebak on a host copy of the same
// inputs. All scale factors are powers of two, so right-vector scaling (V * s),
// left-vector scaling (V * 1/s) and the permutation are all EXACT, and the
// comparison is bit-for-bit -- the device and the reference do identical IEEE
// operations.
//
// The cases exercise the parts gebak can get wrong independently:
//   - Scaling, both sides, over a multi-row window (right multiplies by scale(i),
//     left by its reciprocal).
//   - The permutation loop, whose index walk is the one subtle part: the low block
//     [1, ilo) is replayed in REVERSE (i = ilo - ii) and the high block (ihi, n]
//     forward, with self-maps (k == i) skipped. One n = 6, ilo = 3, ihi = 4 case
//     drives swaps in both blocks.
//   - Both together, and the ilo == ihi window where LAPACK skips scaling.
//   - Quick returns: JOB = None, m = 0.
//
// REQUIRES_GPU (labeled `gpu`): every case runs the device kernels, so
// `ctest -LE gpu` excludes it. The oracle is LAPACKE_?gebak via
// calaman::lapack_reference -- guard on that target in CMakeLists.txt and skip the
// whole suite when it is absent, so a missing oracle is a missing TIER, not a
// silent pass (docs/architecture.md §3).

#include <gtest/gtest.h>
#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.gebak;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
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

// ── host element helpers, one per gebak element type ─────────────────────────
//
// gebak's d_scale is real even for complex T, so `real` is the scale type; `make`
// builds a value from (re, im); `eq` is bit-exact equality. Complex goes through
// the host wwrC* accessors in wwr.complex. `lapacke_ptr` reinterprets a host
// buffer of the element type as the layout-compatible LAPACKE complex pointer --
// the same bridge test/cg_unitary uses for ?heevd.

template<typename T>
struct elem;

template<>
struct elem<float> {
  using real = float;
  static float make(double re, double) { return static_cast<float>(re); }
  static bool eq(float a, float b) { return a == b; }
  static float *lapacke_ptr(float *p) { return p; }
};

template<>
struct elem<double> {
  using real = double;
  static double make(double re, double) { return re; }
  static bool eq(double a, double b) { return a == b; }
  static double *lapacke_ptr(double *p) { return p; }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static bool eq(wwr::wwrFloatComplex a, wwr::wwrFloatComplex b) {
    return wwr::wwrCrealf(a) == wwr::wwrCrealf(b) && wwr::wwrCimagf(a) == wwr::wwrCimagf(b);
  }
  static lapack_complex_float *lapacke_ptr(wwr::wwrFloatComplex *p) {
    return reinterpret_cast<lapack_complex_float *>(p);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static bool eq(wwr::wwrDoubleComplex a, wwr::wwrDoubleComplex b) {
    return wwr::wwrCreal(a) == wwr::wwrCreal(b) && wwr::wwrCimag(a) == wwr::wwrCimag(b);
  }
  static lapack_complex_double *lapacke_ptr(wwr::wwrDoubleComplex *p) {
    return reinterpret_cast<lapack_complex_double *>(p);
  }
};

template<typename T>
using RealOf = typename elem<T>::real;

// ── the reference: LAPACKE_?gebak over a host copy of V ──────────────────────

char job_char(GebakJob j) {
  switch (j) {
  case GebakJob::None: return 'N';
  case GebakJob::Permute: return 'P';
  case GebakJob::Scale: return 'S';
  case GebakJob::Both: return 'B';
  }
  return 'N';
}
char side_char(GebakSide s) { return s == GebakSide::Left ? 'L' : 'R'; }

template<typename T>
lapack_int ref_gebak(GebakJob job, GebakSide side, int n, int ilo, int ihi,
                     const std::vector<RealOf<T>> &scale, int m, std::vector<T> &V, int ldv);

template<>
lapack_int ref_gebak<float>(GebakJob job, GebakSide side, int n, int ilo, int ihi,
                            const std::vector<float> &scale, int m, std::vector<float> &V,
                            int ldv) {
  return LAPACKE_sgebak(LAPACK_COL_MAJOR, job_char(job), side_char(side), n, ilo, ihi, scale.data(),
                        m, V.data(), ldv);
}
template<>
lapack_int ref_gebak<double>(GebakJob job, GebakSide side, int n, int ilo, int ihi,
                             const std::vector<double> &scale, int m, std::vector<double> &V,
                             int ldv) {
  return LAPACKE_dgebak(LAPACK_COL_MAJOR, job_char(job), side_char(side), n, ilo, ihi, scale.data(),
                        m, V.data(), ldv);
}
template<>
lapack_int ref_gebak<wwr::wwrFloatComplex>(GebakJob job, GebakSide side, int n, int ilo, int ihi,
                                           const std::vector<float> &scale, int m,
                                           std::vector<wwr::wwrFloatComplex> &V, int ldv) {
  return LAPACKE_cgebak(LAPACK_COL_MAJOR, job_char(job), side_char(side), n, ilo, ihi, scale.data(),
                        m, elem<wwr::wwrFloatComplex>::lapacke_ptr(V.data()), ldv);
}
template<>
lapack_int ref_gebak<wwr::wwrDoubleComplex>(GebakJob job, GebakSide side, int n, int ilo, int ihi,
                                            const std::vector<double> &scale, int m,
                                            std::vector<wwr::wwrDoubleComplex> &V, int ldv) {
  return LAPACKE_zgebak(LAPACK_COL_MAJOR, job_char(job), side_char(side), n, ilo, ihi, scale.data(),
                        m, elem<wwr::wwrDoubleComplex>::lapacke_ptr(V.data()), ldv);
}

// Run the device gebak and the reference on identical inputs; assert bit-for-bit.
template<typename T>
void expect_matches_reference(GebakJob job, GebakSide side, int n, int ilo, int ihi,
                              const std::vector<RealOf<T>> &scale, int m, const std::vector<T> &V0,
                              int ldv, const char *ctx) {
  auto handle = shared_device();
  auto stream = handle->stream().get();

  auto d_V = to_device(handle, V0);
  auto d_scale = to_device(handle, scale);

  const Status st =
      gebak<T>(stream, job, side, n, ilo, ihi, d_scale.data(), m, d_V.data(), ldv);
  ASSERT_EQ(st, wwr::wwrSuccess) << ctx;
  const std::vector<T> got = from_device(handle, d_V, V0.size());

  std::vector<T> expect = V0;
  const lapack_int rc = ref_gebak<T>(job, side, n, ilo, ihi, scale, m, expect, ldv);
  ASSERT_EQ(rc, 0) << ctx << ": LAPACKE_?gebak returned " << rc;

  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t o = static_cast<std::size_t>(j) * ldv + i;
      EXPECT_TRUE(elem<T>::eq(got[o], expect[o]))
          << ctx << ": mismatch at V(" << i << "," << j << ")";
    }
  }
}

// A dense n-by-m V with power-of-two entries, column-major (ldv may exceed n).
template<typename T>
std::vector<T> dense_V(int n, int m, int ldv) {
  std::vector<T> V(static_cast<std::size_t>(ldv) * m, elem<T>::make(0, 0));
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < n; ++i) {
      const double re = std::ldexp(1.0, ((i + 2 * j) % 5) - 2);
      const double im = std::ldexp(1.0, ((i + j) % 3) - 1);
      V[static_cast<std::size_t>(j) * ldv + i] = elem<T>::make(re, im);
    }
  }
  return V;
}

// ── cases ────────────────────────────────────────────────────────────────────

// Scaling only, over a genuine multi-row window, both sides. scale(i) powers of
// two so V * s and V * 1/s are exact. ldv > n to catch a leading-dimension slip.
template<typename T>
void scale_only() {
  using R = RealOf<T>;
  const int n = 5, m = 3, ldv = 7, ilo = 2, ihi = 5;
  auto V = dense_V<T>(n, m, ldv);
  std::vector<R> scale(static_cast<std::size_t>(n), R(1));
  scale[1] = static_cast<R>(std::ldexp(1.0, 3));
  scale[2] = static_cast<R>(std::ldexp(1.0, -2));
  scale[3] = static_cast<R>(std::ldexp(1.0, 1));
  scale[4] = static_cast<R>(std::ldexp(1.0, -4));
  expect_matches_reference<T>(GebakJob::Scale, GebakSide::Right, n, ilo, ihi, scale, m, V, ldv,
                              "scale_only/Right");
  expect_matches_reference<T>(GebakJob::Scale, GebakSide::Left, n, ilo, ihi, scale, m, V, ldv,
                              "scale_only/Left");
}

// Permutation that drives swaps in both blocks: low block [1,ilo) replayed in
// reverse with a self-map skip, high block (ihi,n] forward. n=6, ilo=3, ihi=4.
// Outside [ilo,ihi] scale holds 1-based interchange targets; inside, the (unused
// by Permute) D diagonal.
template<typename T>
std::vector<RealOf<T>> perm_scale() {
  using R = RealOf<T>;
  std::vector<R> s(6);
  s[0] = R(1);                              // i=1 (ii=2): self-map -> skip
  s[1] = R(1);                              // i=2 (ii=1): swap rows 2,1
  s[2] = static_cast<R>(std::ldexp(1.0, 2)); // middle (D); ignored by Permute
  s[3] = static_cast<R>(std::ldexp(1.0, -1)); // middle (D)
  s[4] = R(2);                              // i=5 (ii=5): swap rows 5,2
  s[5] = R(3);                              // i=6 (ii=6): swap rows 6,3
  return s;
}

template<typename T>
void permute_only() {
  const int n = 6, m = 4, ldv = 6, ilo = 3, ihi = 4;
  auto V = dense_V<T>(n, m, ldv);
  const auto scale = perm_scale<T>();
  expect_matches_reference<T>(GebakJob::Permute, GebakSide::Right, n, ilo, ihi, scale, m, V, ldv,
                              "permute_only/Right");
  expect_matches_reference<T>(GebakJob::Permute, GebakSide::Left, n, ilo, ihi, scale, m, V, ldv,
                              "permute_only/Left");
}

// Both halves together: the middle entries of scale are now the D diagonal and
// must scale rows 3,4 (ilo<ihi) before the permutation runs.
template<typename T>
void both() {
  const int n = 6, m = 4, ldv = 8, ilo = 3, ihi = 4;
  auto V = dense_V<T>(n, m, ldv);
  const auto scale = perm_scale<T>();
  expect_matches_reference<T>(GebakJob::Both, GebakSide::Right, n, ilo, ihi, scale, m, V, ldv,
                              "both/Right");
  expect_matches_reference<T>(GebakJob::Both, GebakSide::Left, n, ilo, ihi, scale, m, V, ldv,
                              "both/Left");
}

// ilo == ihi: LAPACK skips scaling entirely (GO TO 30); only the permutation of
// the surrounding blocks runs. Here n=4, ilo=ihi=2: low block i=1, high block
// i=3,4.
template<typename T>
void single_index_window() {
  using R = RealOf<T>;
  const int n = 4, m = 2, ldv = 4, ilo = 2, ihi = 2;
  auto V = dense_V<T>(n, m, ldv);
  std::vector<R> scale(static_cast<std::size_t>(n), R(1));
  scale[0] = R(1);                               // i=1: self-map -> skip
  scale[1] = static_cast<R>(std::ldexp(1.0, 5)); // middle D -- must be IGNORED (ilo==ihi)
  scale[2] = R(4);                               // i=3: swap rows 3,4
  scale[3] = R(3);                               // i=4: swap rows 4,3
  expect_matches_reference<T>(GebakJob::Both, GebakSide::Right, n, ilo, ihi, scale, m, V, ldv,
                              "single_index_window/Right");
}

// Quick returns: JOB = None leaves V untouched; m = 0 has nothing to do. Both must
// agree with the reference (which also no-ops) and return success.
template<typename T>
void quick_returns() {
  using R = RealOf<T>;
  const int n = 4, m = 3, ldv = 4, ilo = 2, ihi = 3;
  auto V = dense_V<T>(n, m, ldv);
  std::vector<R> scale(static_cast<std::size_t>(n), static_cast<R>(std::ldexp(1.0, 2)));
  expect_matches_reference<T>(GebakJob::None, GebakSide::Right, n, ilo, ihi, scale, m, V, ldv,
                              "quick_returns/None");
  expect_matches_reference<T>(GebakJob::Both, GebakSide::Right, n, ilo, ihi, scale, /*m=*/0, V, ldv,
                              "quick_returns/m0");
}

} // namespace

TEST(GebakOracleTests, ScaleOnly) {
  scale_only<float>();
  scale_only<double>();
  scale_only<wwr::wwrFloatComplex>();
  scale_only<wwr::wwrDoubleComplex>();
}

TEST(GebakOracleTests, PermuteOnly) {
  permute_only<float>();
  permute_only<double>();
  permute_only<wwr::wwrFloatComplex>();
  permute_only<wwr::wwrDoubleComplex>();
}

TEST(GebakOracleTests, Both) {
  both<float>();
  both<double>();
  both<wwr::wwrFloatComplex>();
  both<wwr::wwrDoubleComplex>();
}

TEST(GebakOracleTests, SingleIndexWindow) {
  single_index_window<float>();
  single_index_window<double>();
  single_index_window<wwr::wwrFloatComplex>();
  single_index_window<wwr::wwrDoubleComplex>();
}

TEST(GebakOracleTests, QuickReturns) {
  quick_returns<float>();
  quick_returns<double>();
  quick_returns<wwr::wwrFloatComplex>();
  quick_returns<wwr::wwrDoubleComplex>();
}

} // namespace calaman
