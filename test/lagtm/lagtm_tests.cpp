// Oracle test for calaman.lagtm -- B := alpha * op(A) * X + beta * B for a
// general tridiagonal A = (dl, d, du). The oracle is the reference ?lagtm
// (s/d/c/z) in the SAME precision on the host, called as the Fortran symbol:
// neither LAPACKE nor lapack.h declares it. The wwr complex types are
// layout-compatible with the Fortran COMPLEX the symbols take.
//
// Out-of-range reads are caught by garbage: dl/d/du are views at an offset into
// NaN-filled buffers, and X's rows n..ldx-1 are NaN, so a stray read turns a
// result into NaN. B's rows n..ldb-1 hold a finite sentinel that must survive
// bit-for-bit. Every case stages data on the device, so the suite is
// REQUIRES_GPU (labeled `gpu`). Built only when calaman::lapack_reference
// exists; see this directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <cstddef>

extern "C" {
void slagtm_(const char *trans, const int *n, const int *nrhs, const float *alpha,
             const float *dl, const float *d, const float *du, const float *x, const int *ldx,
             const float *beta, float *b, const int *ldb, std::size_t trans_len);
void dlagtm_(const char *trans, const int *n, const int *nrhs, const double *alpha,
             const double *dl, const double *d, const double *du, const double *x,
             const int *ldx, const double *beta, double *b, const int *ldb,
             std::size_t trans_len);
// COMPLEX / COMPLEX*16 arrays as interleaved (re, im) pairs; ALPHA, BETA real.
void clagtm_(const char *trans, const int *n, const int *nrhs, const float *alpha,
             const void *dl, const void *d, const void *du, const void *x, const int *ldx,
             const float *beta, void *b, const int *ldb, std::size_t trans_len);
void zlagtm_(const char *trans, const int *n, const int *nrhs, const double *alpha,
             const void *dl, const void *d, const void *du, const void *x, const int *ldx,
             const double *beta, void *b, const int *ldb, std::size_t trans_len);
}

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lagtm;
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
using test::factorization_tol;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

using cd = std::complex<double>;

template<typename T>
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
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
std::vector<T> to_host(const DeviceBuffer<T> &device, std::size_t n) {
  const auto handle = shared_device();
  HostBuffer<T> host(n == 0 ? 1 : n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

// Per-type element construction, host read-out and the same-precision oracle.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool is_complex = false;
  static float make(double re, double) { return static_cast<float>(re); }
  static cd to(float v) { return {v, 0.0}; }
  static void ref(char tr, int n, int nrhs, float alpha, const float *dl, const float *d,
                  const float *du, const float *x, int ldx, float beta, float *b, int ldb) {
    slagtm_(&tr, &n, &nrhs, &alpha, dl, d, du, x, &ldx, &beta, b, &ldb, 1);
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool is_complex = false;
  static double make(double re, double) { return re; }
  static cd to(double v) { return {v, 0.0}; }
  static void ref(char tr, int n, int nrhs, double alpha, const double *dl, const double *d,
                  const double *du, const double *x, int ldx, double beta, double *b, int ldb) {
    dlagtm_(&tr, &n, &nrhs, &alpha, dl, d, du, x, &ldx, &beta, b, &ldb, 1);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  using C = wwr::wwrFloatComplex;
  static constexpr bool is_complex = true;
  static C make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static cd to(C v) { return {wwr::wwrCrealf(v), wwr::wwrCimagf(v)}; }
  static void ref(char tr, int n, int nrhs, float alpha, const C *dl, const C *d, const C *du,
                  const C *x, int ldx, float beta, C *b, int ldb) {
    clagtm_(&tr, &n, &nrhs, &alpha, dl, d, du, x, &ldx, &beta, b, &ldb, 1);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static constexpr bool is_complex = true;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static cd to(C v) { return {wwr::wwrCreal(v), wwr::wwrCimag(v)}; }
  static void ref(char tr, int n, int nrhs, double alpha, const C *dl, const C *d, const C *du,
                  const C *x, int ldx, double beta, C *b, int ldb) {
    zlagtm_(&tr, &n, &nrhs, &alpha, dl, d, du, x, &ldx, &beta, b, &ldb, 1);
  }
};

struct TransCase {
  char c;
  Trans trans;
};
constexpr TransCase kTransCases[] = {{'N', Trans::N}, {'T', Trans::T}, {'C', Trans::C}};
// The values DLAGTM honours (LAPACK 3.12: both alpha and beta in {-1, 0, 1}).
constexpr double kAdmissible[] = {-1.0, 0.0, 1.0};

// Slots of NaN before and after each diagonal view.
constexpr std::size_t kPad = 3;
// B's padding rows: finite, so "unchanged" is an exact compare.
constexpr double kSentinel = -777.25;

struct Shape {
  std::size_t n, nrhs, ldx, ldb;
};

// One problem: the three diagonals (padded views), X (NaN padding rows) and the
// initial B (sentinel padding rows), all from one seed.
template<typename T>
struct Problem {
  Shape s;
  std::vector<T> dl, d, du, x, b;

  Problem(Shape shape, std::uint32_t seed)
      : s(shape), dl(s.n + 2 * kPad), d(s.n + 2 * kPad), du(s.n + 2 * kPad),
        x(s.ldx * s.nrhs + 1), b(s.ldb * s.nrhs + 1) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto rnd = [&] {
      const double re = dist(gen);
      return elem<T>::make(re, dist(gen));
    };
    const std::size_t ne = s.n == 0 ? 0 : s.n - 1;
    for (std::size_t k = 0; k < d.size(); ++k) {
      const bool in_d = k >= kPad && k < kPad + s.n;
      const bool in_e = k >= kPad && k < kPad + ne;
      dl[k] = in_e ? rnd() : elem<T>::make(nan, nan);
      d[k] = in_d ? rnd() : elem<T>::make(nan, nan);
      du[k] = in_e ? rnd() : elem<T>::make(nan, nan);
    }
    for (std::size_t k = 0; k < x.size(); ++k) {
      x[k] = k % s.ldx < s.n && k / s.ldx < s.nrhs ? rnd() : elem<T>::make(nan, nan);
    }
    for (std::size_t k = 0; k < b.size(); ++k) {
      b[k] = k % s.ldb < s.n && k / s.ldb < s.nrhs ? rnd() : elem<T>::make(kSentinel, kSentinel);
    }
  }

  // |op(A)| * |X| + |B| at (i, j): the magnitude the element's error scales with.
  double magnitude(std::size_t i, std::size_t j) const {
    // A NaN planted in an operand the call must not read contributes nothing.
    const auto a = [](const T &v) {
      const double m = std::abs(elem<T>::to(v));
      return std::isfinite(m) ? m : 0.0;
    };
    double m = a(b[i + j * s.ldb]) + a(d[kPad + i]) * a(x[i + j * s.ldx]);
    if (i > 0) {
      m += std::max(a(dl[kPad + i - 1]), a(du[kPad + i - 1])) * a(x[i - 1 + j * s.ldx]);
    }
    if (i + 1 < s.n) {
      m += std::max(a(dl[kPad + i]), a(du[kPad + i])) * a(x[i + 1 + j * s.ldx]);
    }
    return m;
  }
};

template<typename T>
std::vector<T> run_reference(const Problem<T> &p, char tr, double alpha, double beta) {
  using R = typename elem<T>::R;
  std::vector<T> b = p.b;
  elem<T>::ref(tr, static_cast<int>(p.s.n), static_cast<int>(p.s.nrhs), static_cast<R>(alpha),
               p.dl.data() + kPad, p.d.data() + kPad, p.du.data() + kPad, p.x.data(),
               static_cast<int>(p.s.ldx), static_cast<R>(beta), b.data(),
               static_cast<int>(p.s.ldb));
  return b;
}

template<typename T>
std::vector<T> run_device(const Problem<T> &p, Trans trans, double alpha, double beta) {
  using R = typename elem<T>::R;
  auto d_dl = to_device(p.dl);
  auto d_d = to_device(p.d);
  auto d_du = to_device(p.du);
  auto d_x = to_device(p.x);
  auto d_b = to_device(p.b);
  const Status st = lagtm<T>(shared_device()->stream().get(), trans, p.s.n, p.s.nrhs,
                             static_cast<R>(alpha), d_dl.data() + kPad, d_d.data() + kPad,
                             d_du.data() + kPad, d_x.data(), p.s.ldx, static_cast<R>(beta),
                             d_b.data(), p.s.ldb);
  EXPECT_TRUE(st.ok()) << "lagtm returned " << st.name() << ": " << st.message();
  return to_host(d_b, p.b.size());
}

// In-range elements within the shared tolerance of the expected value scaled by
// the element's magnitude (scaled by max(|alpha|, |beta|, 1)); padding exact.
template<typename T>
void expect_close(const Problem<T> &p, const std::vector<T> &got, const std::vector<T> &want,
                  double scale, const std::string &what) {
  using R = typename elem<T>::R;
  for (std::size_t k = 0; k < got.size(); ++k) {
    const std::size_t i = k % p.s.ldb;
    const std::size_t j = k / p.s.ldb;
    const cd g = elem<T>::to(got[k]);
    const cd w = elem<T>::to(want[k]);
    if (i < p.s.n && j < p.s.nrhs) {
      ASSERT_TRUE(std::isfinite(w.real()) && std::isfinite(w.imag())) << what;
      const double tol =
          factorization_tol<R>(static_cast<R>(scale * p.magnitude(i, j)), 4, 4);
      ASSERT_LE(std::abs(g - w), tol) << what << " at (" << i << ", " << j << ")";
    } else {
      ASSERT_EQ(g, w) << what << ": padding slot " << k << " was written";
    }
  }
}

template<typename T>
void sweep_admissible(Shape shape) {
  const Problem<T> p(shape, static_cast<std::uint32_t>(31 * shape.n + 7 * shape.nrhs + 3));
  for (const TransCase tc : kTransCases) {
    for (const double alpha : kAdmissible) {
      for (const double beta : kAdmissible) {
        const auto want = run_reference(p, tc.c, alpha, beta);
        const auto got = run_device(p, tc.trans, alpha, beta);
        std::ostringstream what;
        what << "trans=" << tc.c << " alpha=" << alpha << " beta=" << beta << " n=" << shape.n
             << " nrhs=" << shape.nrhs << " ldx=" << shape.ldx << " ldb=" << shape.ldb;
        expect_close(p, got, want, 1.0, what.str());
      }
    }
  }
}

// n = 1 (dl/du empty), one and several right-hand sides, tight and padded
// leading dimensions, and sizes around and past the 128-thread block.
constexpr Shape kShapes[] = {{1, 1, 1, 1},       {1, 3, 4, 2},     {2, 1, 2, 2},
                             {3, 2, 3, 5},       {7, 5, 9, 11},    {128, 1, 128, 130},
                             {129, 4, 131, 129}, {257, 3, 260, 300}, {1000, 2, 1003, 1000}};

template<typename T>
void run_shapes() {
  for (const Shape s : kShapes) {
    sweep_admissible<T>(s);
    if (testing::Test::HasFatalFailure()) {
      return;
    }
  }
}

// ========================================================================
// Device: agreement with the reference ?lagtm on every trans and every
// admissible (alpha, beta), per type
// ========================================================================

TEST(LagtmOracleTests, Float) {
  run_shapes<float>();
}
TEST(LagtmOracleTests, Double) {
  run_shapes<double>();
}
TEST(LagtmOracleTests, ComplexFloat) {
  run_shapes<wwr::wwrFloatComplex>();
}
TEST(LagtmOracleTests, ComplexDouble) {
  run_shapes<wwr::wwrDoubleComplex>();
}

// 'C' must actually conjugate: for complex data it differs from 'T' (so the
// oracle sweep above is not passing C as T), and each matches its oracle.
template<typename T>
void expect_conj_differs() {
  const Problem<T> p({5, 2, 6, 7}, 17);
  const auto t = run_device(p, Trans::T, 1.0, 0.0);
  const auto c = run_device(p, Trans::C, 1.0, 0.0);
  double diff = 0.0;
  for (std::size_t k = 0; k < t.size(); ++k) {
    diff = std::max(diff, std::abs(elem<T>::to(t[k]) - elem<T>::to(c[k])));
  }
  EXPECT_GT(diff, 1.0);
  expect_close(p, c, run_reference(p, 'C', 1.0, 0.0), 1.0, "conj");
}

TEST(LagtmOracleTests, ConjTransConjugates) {
  expect_conj_differs<wwr::wwrFloatComplex>();
  expect_conj_differs<wwr::wwrDoubleComplex>();
}

// Generalised scalars (the documented divergence): alpha * (A X) + beta * B,
// with A X from the oracle at alpha = 1, beta = 0, combined on the host.
template<typename T>
void expect_general_scalars() {
  const Problem<T> p({33, 3, 35, 34}, 23);
  for (const TransCase tc : kTransCases) {
    constexpr std::array<std::pair<double, double>, 3> kScalars{
        {{2.5, -0.75}, {-0.5, 3.0}, {0.0, 0.5}}};
    for (const auto [alpha, beta] : kScalars) {
      const auto ax = run_reference(p, tc.c, 1.0, 0.0);
      std::vector<T> want = p.b;
      for (std::size_t j = 0; j < p.s.nrhs; ++j) {
        for (std::size_t i = 0; i < p.s.n; ++i) {
          const std::size_t k = i + j * p.s.ldb;
          const cd v = alpha * elem<T>::to(ax[k]) + beta * elem<T>::to(p.b[k]);
          want[k] = elem<T>::make(v.real(), v.imag());
        }
      }
      std::ostringstream what;
      what << "general trans=" << tc.c << " alpha=" << alpha << " beta=" << beta;
      expect_close(p, run_device(p, tc.trans, alpha, beta), want, 4.0, what.str());
    }
  }
}

TEST(LagtmOracleTests, GeneralScalars) {
  expect_general_scalars<float>();
  expect_general_scalars<double>();
  expect_general_scalars<wwr::wwrFloatComplex>();
  expect_general_scalars<wwr::wwrDoubleComplex>();
}

// beta == 0 overwrites B without reading it, and alpha == 0 never reads A or X:
// a NaN planted there does not reach the result, as in DLAGTM.
template<typename T>
void expect_unread_operands() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const TransCase tc : kTransCases) {
    Problem<T> p({6, 2, 6, 6}, 41);
    for (std::size_t k = 0; k < p.b.size() - 1; ++k) {
      p.b[k] = elem<T>::make(nan, nan);
    }
    expect_close(p, run_device(p, tc.trans, 1.0, 0.0), run_reference(p, tc.c, 1.0, 0.0), 1.0,
                 "beta=0 read B");

    Problem<T> q({6, 2, 6, 6}, 43);
    std::fill(q.x.begin(), q.x.end(), elem<T>::make(nan, nan));
    std::fill(q.d.begin(), q.d.end(), elem<T>::make(nan, nan));
    for (const double beta : kAdmissible) {
      expect_close(q, run_device(q, tc.trans, 0.0, beta), run_reference(q, tc.c, 0.0, beta), 1.0,
                   "alpha=0 read A or X");
    }
  }
}

TEST(LagtmOracleTests, UnreadOperands) {
  expect_unread_operands<double>();
  expect_unread_operands<wwr::wwrFloatComplex>();
}

// n == 0 or nrhs == 0: nothing is written, as DLAGTM returns at once.
TEST(LagtmOracleTests, EmptyLeavesBUntouched) {
  using T = wwr::wwrDoubleComplex;
  for (const Shape s : {Shape{0, 3, 1, 1}, Shape{4, 0, 4, 4}}) {
    const Problem<T> p(s, 5);
    for (const TransCase tc : kTransCases) {
      const auto got = run_device(p, tc.trans, 1.0, 0.0);
      const auto ref = run_reference(p, tc.c, 1.0, 0.0);
      for (std::size_t k = 0; k < got.size(); ++k) {
        EXPECT_EQ(elem<T>::to(got[k]), elem<T>::to(p.b[k])) << "trans=" << tc.c;
        EXPECT_EQ(elem<T>::to(ref[k]), elem<T>::to(p.b[k])) << "oracle trans=" << tc.c;
      }
    }
  }
}

} // namespace
} // namespace calaman
