// Oracle test for calaman.trevc3: the left/right eigenvectors of a real
// quasi-triangular (Schur) matrix on the device must agree with reference
// LAPACK -- ?trevc3 with HOWMNY = 'A' -- computed in the SAME precision on the
// host. The device fills VR (right) and/or VL (left), one column per real
// eigenvalue and two (real part, imaginary part) per complex-conjugate pair, in
// the diagonal order of T, each normalized to unit infinity norm.
//
// WHAT IS COMPARED, and why not entry-for-entry: an eigenvector is defined only
// up to a scalar, and ?trevc3 pins it by the infinity-norm convention. Device
// and reference run the same back-substitution with the same convention, so they
// agree up to the one degree of freedom the convention leaves -- an overall sign.
// The primary check is therefore the scale/sign-INVARIANT eigen-residual: for a
// real right eigenvector v with eigenvalue lambda, ||(T - lambda*I) v|| must be
// negligible next to ||T||; for a complex pair the real/imaginary parts satisfy
// the coupled relation T*vr = wr*vr - wi*vi, T*vi = wi*vr + wr*vi. Left vectors
// satisfy the transposed relations. As a second, tighter cross-check the device
// VR/VL is compared to the reference's after resolving the sign (dotting the two
// columns), which must match to the shared tolerance.
//
// ?trevc3 is a computational routine with no LAPACKE C binding, so the oracle
// calls the Fortran symbol strevc3_ / dtrevc3_ directly, exactly as the laqr5 /
// laexc suites call their auxiliaries. Fortran passes every scalar by reference;
// SELECT is a LOGICAL array (unused for HOWMNY = 'A') and the SIDE / HOWMNY
// characters pass with an implicit trailing hidden length each.
//
// The cases build real upper quasi-triangular T directly: a purely real-spectrum
// triangular T, a T with one 2x2 (complex-conjugate) block, and a mixed T with
// both -- exercising the real 1x1, complex 2x2, and interleaved back-substitution
// paths for right-only, left-only and both sides.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages T on the device and runs
// the kernel, so the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.trevc3;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?trevc3: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; SIDE / HOWMNY are single characters (trailing hidden
// length each). SELECT is a 4-byte-int LOGICAL array, unused for HOWMNY = 'A'.
extern "C" {
void strevc3_(const char *side, const char *howmny, int *select, const int *n, const float *t,
              const int *ldt, float *vl, const int *ldvl, float *vr, const int *ldvr, const int *mm,
              int *m, float *work, const int *lwork, int *info, int side_len, int howmny_len);
void dtrevc3_(const char *side, const char *howmny, int *select, const int *n, const double *t,
              const int *ldt, double *vl, const int *ldvl, double *vr, const int *ldvr,
              const int *mm, int *m, double *work, const int *lwork, int *info, int side_len,
              int howmny_len);
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

void ref_trevc3(char side, int n, const float *t, int ldt, float *vl, int ldvl, float *vr,
                int ldvr) {
  const char howmny = 'A';
  std::vector<int> select(n, 0);
  const int mm = n;
  int m = 0, info = 0;
  const int lwork = std::max(1, 5 * n);
  std::vector<float> work(static_cast<std::size_t>(lwork));
  strevc3_(&side, &howmny, select.data(), &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, &m, work.data(),
           &lwork, &info, 1, 1);
}
void ref_trevc3(char side, int n, const double *t, int ldt, double *vl, int ldvl, double *vr,
                int ldvr) {
  const char howmny = 'A';
  std::vector<int> select(n, 0);
  const int mm = n;
  int m = 0, info = 0;
  const int lwork = std::max(1, 5 * n);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dtrevc3_(&side, &howmny, select.data(), &n, t, &ldt, vl, &ldvl, vr, &ldvr, &mm, &m, work.data(),
           &lwork, &info, 1, 1);
}

// Column-major (i,j) index, 0-based, leading dimension ld.
std::size_t idx(int i, int j, int ld) {
  return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ld);
}

// The (wr, wi) spectrum of a real quasi-triangular T and, per column, whether it
// is the first (ip=+1), second (ip=-1) of a complex pair, or real (ip=0). A 2x2
// block sits where T(i+1,i) != 0.
template<typename T>
struct Spectrum {
  std::vector<T> wr, wi;
  std::vector<int> ip;
};

template<typename T>
Spectrum<T> spectrum(int n, const std::vector<T> &t, int ld) {
  Spectrum<T> s;
  s.wr.assign(n, T{0});
  s.wi.assign(n, T{0});
  s.ip.assign(n, 0);
  int i = 0;
  while (i < n) {
    const bool pair = (i + 1 < n) && (t[idx(i + 1, i, ld)] != T{0});
    if (!pair) {
      s.wr[i] = t[idx(i, i, ld)];
      s.wi[i] = T{0};
      s.ip[i] = 0;
      i += 1;
    } else {
      const T a = t[idx(i, i, ld)];
      const T d = t[idx(i + 1, i + 1, ld)];
      const T b = t[idx(i, i + 1, ld)];
      const T c = t[idx(i + 1, i, ld)];
      const T re = (a + d) * T{0.5};
      const T im = std::sqrt(std::abs(b)) * std::sqrt(std::abs(c));
      s.wr[i] = re;
      s.wi[i] = im;
      s.ip[i] = 1;
      s.wr[i + 1] = re;
      s.wi[i + 1] = -im;
      s.ip[i + 1] = -1;
      i += 2;
    }
  }
  return s;
}

// ||T|| (max abs entry), the scale the residual tolerance rides on.
template<typename T>
T tnorm(const std::vector<T> &t) {
  T m = T{1};
  for (const T e : t) {
    m = std::max(m, std::abs(e));
  }
  return m;
}

template<typename T>
T tol(T norm) {
  return T{256} * eps<T>() * (norm + T{1});
}

// Right-eigenvector residual: for a real column j, ||(T - wr*I) v||_inf; for a
// complex pair (columns j, j+1) with v = vr + i*vi, the max of the two coupled
// residuals ||T vr - wr vr + wi vi|| and ||T vi - wi vr - wr vi||.
template<typename T>
void check_right(int n, const std::vector<T> &t, int ld, const std::vector<T> &vr,
                 const Spectrum<T> &s, const char *ctx) {
  const T tl = tol(tnorm(t));
  for (int j = 0; j < n;) {
    if (s.ip[j] == 0) {
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T acc = -s.wr[j] * vr[idx(i, j, ld)];
        for (int k = 0; k < n; ++k) {
          acc += t[idx(i, k, ld)] * vr[idx(k, j, ld)];
        }
        r = std::max(r, std::abs(acc));
      }
      EXPECT_LE(r, tl) << ctx << " right real residual col " << j;
      j += 1;
    } else {
      const int jr = j, ji = j + 1;
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T accr = -s.wr[jr] * vr[idx(i, jr, ld)] + s.wi[jr] * vr[idx(i, ji, ld)];
        T acci = -s.wi[jr] * vr[idx(i, jr, ld)] - s.wr[jr] * vr[idx(i, ji, ld)];
        for (int k = 0; k < n; ++k) {
          accr += t[idx(i, k, ld)] * vr[idx(k, jr, ld)];
          acci += t[idx(i, k, ld)] * vr[idx(k, ji, ld)];
        }
        r = std::max(r, std::max(std::abs(accr), std::abs(acci)));
      }
      EXPECT_LE(r, tl) << ctx << " right complex residual cols " << jr << "," << ji;
      j += 2;
    }
  }
}

// Left-eigenvector residual: a left eigenvector u satisfies u^H T = lambda u^H,
// i.e. T^T ur = wr ur + wi ui and T^T ui = -wi ur + wr ui for the conjugate
// eigenvalue wr + i*wi stored at the column (?trevc normalizes the left vector of
// the eigenvalue wr - i*wi into the pair). Checking (T^T - wr) u for the real
// case and the coupled complex relation is sign/scale invariant.
template<typename T>
void check_left(int n, const std::vector<T> &t, int ld, const std::vector<T> &vl,
                const Spectrum<T> &s, const char *ctx) {
  const T tl = tol(tnorm(t));
  for (int j = 0; j < n;) {
    if (s.ip[j] == 0) {
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T acc = -s.wr[j] * vl[idx(i, j, ld)];
        for (int k = 0; k < n; ++k) {
          acc += t[idx(k, i, ld)] * vl[idx(k, j, ld)]; // (T^T)_{ik} = T_{ki}
        }
        r = std::max(r, std::abs(acc));
      }
      EXPECT_LE(r, tl) << ctx << " left real residual col " << j;
      j += 1;
    } else {
      const int jr = j, ji = j + 1;
      // Left vector corresponds to eigenvalue wr + i*wi (column jr real, ji imag):
      // T^T ur = wr ur - wi ui, T^T ui = wi ur + wr ui.
      T r = T{0};
      for (int i = 0; i < n; ++i) {
        T accr = -s.wr[jr] * vl[idx(i, jr, ld)] - s.wi[jr] * vl[idx(i, ji, ld)];
        T acci = s.wi[jr] * vl[idx(i, jr, ld)] - s.wr[jr] * vl[idx(i, ji, ld)];
        for (int k = 0; k < n; ++k) {
          accr += t[idx(k, i, ld)] * vl[idx(k, jr, ld)];
          acci += t[idx(k, i, ld)] * vl[idx(k, ji, ld)];
        }
        r = std::max(r, std::max(std::abs(accr), std::abs(acci)));
      }
      EXPECT_LE(r, tl) << ctx << " left complex residual cols " << jr << "," << ji;
      j += 2;
    }
  }
}

// Sign-resolved column-wise agreement with the reference: for each output column
// pick the sign s in {+1,-1} minimizing ||dev - s*ref||, then require it small.
// Both paths use the same infinity-norm normalization, so the only slack is the
// overall sign and rounding.
template<typename T>
void check_against_ref(int n, const std::vector<T> &dev, const std::vector<T> &ref, int ld,
                       T norm, const char *ctx) {
  const T tl = tol(norm);
  for (int j = 0; j < n; ++j) {
    T dplus = T{0}, dminus = T{0};
    for (int i = 0; i < n; ++i) {
      const T a = dev[idx(i, j, ld)];
      const T b = ref[idx(i, j, ld)];
      dplus = std::max(dplus, std::abs(a - b));
      dminus = std::max(dminus, std::abs(a + b));
    }
    EXPECT_LE(std::min(dplus, dminus), tl) << ctx << " vs ref col " << j;
  }
}

template<typename T>
void run_case(bool want_left, bool want_right, int n, const std::vector<T> &t, const char *ctx) {
  const int ld = n;
  const Spectrum<T> s = spectrum<T>(n, t, ld);
  const T norm = tnorm(t);

  // Reference VR / VL (HOWMNY = 'A'): pass only the sides requested so an unused
  // side's array may be a 1-element placeholder.
  std::vector<T> r_vr(static_cast<std::size_t>(n) * n, T{0});
  std::vector<T> r_vl(static_cast<std::size_t>(n) * n, T{0});
  {
    std::vector<T> tt = t; // reference reads T but may touch workspace; copy to be safe
    if (want_right && want_left) {
      ref_trevc3('B', n, tt.data(), ld, r_vl.data(), ld, r_vr.data(), ld);
    } else if (want_right) {
      ref_trevc3('R', n, tt.data(), ld, r_vl.data(), 1, r_vr.data(), ld);
    } else {
      ref_trevc3('L', n, tt.data(), ld, r_vl.data(), ld, r_vr.data(), 1);
    }
  }

  auto handle = shared_device();
  auto d_t = to_device(handle, t);
  std::vector<T> zero_n2(static_cast<std::size_t>(n) * n, T{0});
  auto d_vl = to_device(handle, zero_n2);
  auto d_vr = to_device(handle, zero_n2);
  std::vector<T> zero_work(static_cast<std::size_t>(3) * n, T{0});
  auto d_work = to_device(handle, zero_work);

  const auto status = trevc3<T>(handle->stream().get(), want_left, want_right, n, d_t.data(), ld,
                                d_vl.data(), ld, d_vr.data(), ld, d_work.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  if (want_right) {
    const auto g_vr = from_device(handle, d_vr, static_cast<std::size_t>(n) * n);
    check_right<T>(n, t, ld, g_vr, s, ctx);
    check_against_ref<T>(n, g_vr, r_vr, ld, norm, ctx);
  }
  if (want_left) {
    const auto g_vl = from_device(handle, d_vl, static_cast<std::size_t>(n) * n);
    check_left<T>(n, t, ld, g_vl, s, ctx);
    check_against_ref<T>(n, g_vl, r_vl, ld, norm, ctx);
  }
}

template<typename T>
void put(std::vector<T> &a, int i, int j, int ld, T v) {
  a[idx(i, j, ld)] = v;
}

// A purely-real-spectrum upper-triangular T (distinct diagonal, dense strict
// upper triangle), so every eigenvector is a real 1x1 block.
template<typename T>
std::vector<T> make_real_triangular(int n) {
  std::vector<T> t(static_cast<std::size_t>(n) * n, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= j; ++i) {
      if (i == j) {
        put(t, i, j, n, static_cast<T>(n - i)); // distinct decreasing diagonal
      } else {
        put(t, i, j, n, static_cast<T>(0.4 * (j + 1) - 0.2 * (i + 1)));
      }
    }
  }
  return t;
}

// A quasi-triangular T with one standardized 2x2 complex block at (c,c), the
// rest real-triangular. The 2x2 has equal diagonals and b*c < 0 so the pair is
// complex-conjugate, matching a real Schur form.
template<typename T>
std::vector<T> make_one_complex(int n, int c) {
  std::vector<T> t = make_real_triangular<T>(n);
  const T mu = static_cast<T>(n - c); // shared real part
  put(t, c, c, n, mu);
  put(t, c + 1, c + 1, n, mu);
  put(t, c, c + 1, n, static_cast<T>(0.9));  // b > 0
  put(t, c + 1, c, n, static_cast<T>(-0.7)); // c < 0  => complex pair
  return t;
}

// Mixed spectrum: two complex 2x2 blocks interleaved with real diagonal entries.
template<typename T>
std::vector<T> make_mixed(int n) {
  std::vector<T> t = make_real_triangular<T>(n);
  // block at rows 1..2 (0-based)
  put(t, 1, 1, n, static_cast<T>(n - 1));
  put(t, 2, 2, n, static_cast<T>(n - 1));
  put(t, 1, 2, n, static_cast<T>(1.1));
  put(t, 2, 1, n, static_cast<T>(-0.8));
  // block at rows 4..5 (0-based), present when n >= 6
  if (n >= 6) {
    put(t, 4, 4, n, static_cast<T>(n - 4));
    put(t, 5, 5, n, static_cast<T>(n - 4));
    put(t, 4, 5, n, static_cast<T>(0.6));
    put(t, 5, 4, n, static_cast<T>(-1.3));
  }
  return t;
}

template<typename T>
void run_all() {
  // Purely real spectrum: right, left, both.
  run_case<T>(false, true, 6, make_real_triangular<T>(6), "real R");
  run_case<T>(true, false, 6, make_real_triangular<T>(6), "real L");
  run_case<T>(true, true, 6, make_real_triangular<T>(6), "real B");
  // One complex-conjugate pair.
  run_case<T>(false, true, 6, make_one_complex<T>(6, 2), "cplx1 R");
  run_case<T>(true, false, 6, make_one_complex<T>(6, 2), "cplx1 L");
  run_case<T>(true, true, 6, make_one_complex<T>(6, 2), "cplx1 B");
  // Mixed real + two complex blocks.
  run_case<T>(false, true, 7, make_mixed<T>(7), "mixed R");
  run_case<T>(true, false, 7, make_mixed<T>(7), "mixed L");
  run_case<T>(true, true, 7, make_mixed<T>(7), "mixed B");
}

} // namespace

TEST(Trevc3OracleTests, MatchesReferenceFloat) {
  run_all<float>();
}

TEST(Trevc3OracleTests, MatchesReferenceDouble) {
  run_all<double>();
}

} // namespace calaman
