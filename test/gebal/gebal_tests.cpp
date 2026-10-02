// Spec test for calaman.gebal -- matrix balancing. There is no device oracle to
// diff against, and the off-diagonal 1-norm convention can differ from a modern
// reference LAPACK by a radix step (see src/gebal/README.md), so this checks the
// SPECIFICATION rather than a reference result:
//
//   1. Exact reproduce. Undoing the reported permutation and scaling reconstructs
//      the input BIT FOR BIT -- the scale factors are powers of two, so the
//      similarity B = D^-1 P^T A P D is exact, not merely accurate. A = P D B D^-1
//      P^T is rebuilt on the host by replaying gebal's own swap order and the
//      diagonal D, and compared with ==. This is the headline test: it validates
//      permutation and scaling together, for every element type.
//   2. Analytic 2x2. [[1, 2^10], [2^-10, 1]] balances to [[1, 1], [1, 1]] with
//      D = diag(2^10, 1) -- the one case whose exact answer is known in closed
//      form.
//   3. Idempotence (stationarity, black-box). A second Scale pass over an
//      already-balanced matrix must accept nothing: B is unchanged and every
//      scale factor is 1. The negative control is case 2, where the first pass
//      demonstrably DID change the matrix.
//   4. Structural. A diagonal matrix isolates every eigenvalue (ilo == ihi == 1);
//      a matrix with one isolatable column drives a real permutation swap and
//      bounds ilo/ihi; n == 1 is the trivial fixed point.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/scale/work on the device
// and runs the mark/pick/swap/sweep kernels, so `ctest -LE gpu` excludes it. It
// needs no reference LAPACK -- the spec is its own oracle -- so unlike the linalg
// suites it does not guard on calaman::lapack_reference.

#include <gtest/gtest.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.wrappers.common;
import wwr.extension.memory_buffer;
import calaman.gebal;
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

// ── host element helpers, one per gebal element type ─────────────────────────
//
// gebal's d_scale is real even for complex T, so `real` is the scale/norm type;
// `make` builds a value from (re, im); `scale_by` multiplies by a real power of
// two (what reconstruction does); `eq` is bit-exact equality. Complex goes
// through the host wwrC* accessors in wwr.complex -- the member .x/.y is not
// portable to hipComplex.

template<typename T>
struct elem;

template<>
struct elem<float> {
  using real = float;
  static float make(double re, double) { return static_cast<float>(re); }
  static float scale_by(float x, float s) { return x * s; }
  static bool eq(float a, float b) { return a == b; }
};

template<>
struct elem<double> {
  using real = double;
  static double make(double re, double) { return re; }
  static double scale_by(double x, double s) { return x * s; }
  static bool eq(double a, double b) { return a == b; }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using real = float;
  static wwr::wwrFloatComplex make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static wwr::wwrFloatComplex scale_by(wwr::wwrFloatComplex x, float s) {
    return wwr::make_wwrFloatComplex(wwr::wwrCrealf(x) * s, wwr::wwrCimagf(x) * s);
  }
  static bool eq(wwr::wwrFloatComplex a, wwr::wwrFloatComplex b) {
    return wwr::wwrCrealf(a) == wwr::wwrCrealf(b) && wwr::wwrCimagf(a) == wwr::wwrCimagf(b);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using real = double;
  static wwr::wwrDoubleComplex make(double re, double im) {
    return wwr::make_wwrDoubleComplex(re, im);
  }
  static wwr::wwrDoubleComplex scale_by(wwr::wwrDoubleComplex x, double s) {
    return wwr::make_wwrDoubleComplex(wwr::wwrCreal(x) * s, wwr::wwrCimag(x) * s);
  }
  static bool eq(wwr::wwrDoubleComplex a, wwr::wwrDoubleComplex b) {
    return wwr::wwrCreal(a) == wwr::wwrCreal(b) && wwr::wwrCimag(a) == wwr::wwrCimag(b);
  }
};

template<typename T>
using RealOf = typename elem<T>::real;

// ── driver wrapper ───────────────────────────────────────────────────────────

template<typename T>
struct GebalRun {
  std::vector<T> B;              // balanced matrix, column-major
  std::vector<RealOf<T>> scale;  // D diagonal inside [ilo,ihi], swap index outside
  int ilo = 0;
  int ihi = 0;
  wwr::wwrError_t status{};
};

// ldexp as a power-of-two literal of the element's real type.
template<typename T>
RealOf<T> pow2(int e) {
  return static_cast<RealOf<T>>(std::ldexp(1.0, e));
}

template<typename T>
GebalRun<T> run(std::shared_ptr<DeviceHandle> handle, GebalJob job, int n, const std::vector<T> &A,
                int lda) {
  using R = RealOf<T>;
  auto stream = handle->stream().get();

  auto d_A = to_device(handle, A);
  std::vector<R> scale_init(static_cast<std::size_t>(n == 0 ? 1 : n), R{0});
  auto d_scale = to_device(handle, scale_init);

  int lwork = 0;
  gebal_bufferSize<T>(n, &lwork);
  std::vector<int> work_init(static_cast<std::size_t>(std::max(lwork, 1)), 0);
  auto d_work = to_device(handle, work_init);

  GebalRun<T> r;
  r.status = gebal<T>(stream, job, n, d_A.data(), lda, &r.ilo, &r.ihi, d_scale.data(), d_work.data());
  r.B = from_device(handle, d_A, static_cast<std::size_t>(lda) * n);
  r.scale = from_device(handle, d_scale, static_cast<std::size_t>(n));
  return r;
}

// Reconstruct A = P D B D^-1 P^T on the host from gebal's outputs and assert it
// equals the original bit for bit. The permutation is rebuilt by replaying
// gebal's OWN swap order (row phase top-down over the high block, then column
// phase bottom-up over the low block), so perm[pos] is the original index now at
// position pos.
template<typename T>
void expect_reproduces(const std::vector<T> &A, const GebalRun<T> &g, int n, int lda,
                       const char *ctx) {
  using R = RealOf<T>;

  std::vector<int> perm(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    perm[static_cast<std::size_t>(i)] = i;
  }
  // High block: 0-based positions [ihi, n-1] were isolated by the row phase, in
  // order l = n-1 downto ihi. scale[l] holds the 1-based index l was swapped with.
  for (int l = n - 1; l >= g.ihi; --l) {
    const int tgt = static_cast<int>(std::lround(static_cast<double>(g.scale[l]))) - 1;
    if (tgt != l) {
      std::swap(perm[static_cast<std::size_t>(tgt)], perm[static_cast<std::size_t>(l)]);
    }
  }
  // Low block: 0-based positions [0, ilo-2] were isolated by the column phase, in
  // order k = 0 up to ilo-2.
  for (int k = 0; k <= g.ilo - 2; ++k) {
    const int tgt = static_cast<int>(std::lround(static_cast<double>(g.scale[k]))) - 1;
    if (tgt != k) {
      std::swap(perm[static_cast<std::size_t>(tgt)], perm[static_cast<std::size_t>(k)]);
    }
  }

  // D is the reported scale inside [ilo, ihi] (1-based) and 1 elsewhere.
  const auto d = [&](int pos) -> R {
    return (pos >= g.ilo - 1 && pos <= g.ihi - 1) ? g.scale[static_cast<std::size_t>(pos)] : R{1};
  };

  std::vector<T> rec(static_cast<std::size_t>(lda) * n, elem<T>::make(0, 0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const R f = d(i) / d(j); // A(p_i, p_j) = B(i,j) * d_i / d_j
      const T v = elem<T>::scale_by(g.B[static_cast<std::size_t>(j) * lda + i], f);
      rec[static_cast<std::size_t>(perm[static_cast<std::size_t>(j)]) * lda +
          perm[static_cast<std::size_t>(i)]] = v;
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_TRUE(elem<T>::eq(rec[static_cast<std::size_t>(j) * lda + i],
                              A[static_cast<std::size_t>(j) * lda + i]))
          << ctx << ": reproduce mismatch at (" << i << "," << j << ")";
    }
  }
}

// ── cases ────────────────────────────────────────────────────────────────────

// A dense, imbalanced matrix with no isolatable row or column: the permutation is
// trivial (ilo=1, ihi=n) and only scaling runs. All entries are powers of two, so
// the balanced result and its reconstruction are exact.
template<typename T>
void dense_scaling() {
  auto handle = shared_device();
  const int n = 4;
  const int lda = n;
  std::vector<T> A(static_cast<std::size_t>(lda) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      A[static_cast<std::size_t>(j) * lda + i] =
          (i == j) ? elem<T>::make(1.0, 0.0)
                   : static_cast<T>(elem<T>::make(std::ldexp(1.0, (i - j) * 3), 0.0));
    }
  }
  auto g = run<T>(handle, GebalJob::Both, n, A, lda);
  ASSERT_EQ(g.status, wwr::wwrSuccess);
  EXPECT_EQ(g.ilo, 1);
  EXPECT_EQ(g.ihi, n);
  expect_reproduces<T>(A, g, n, lda, "dense_scaling");
}

// One isolatable column (column 2 has only its diagonal nonzero) sitting in an
// otherwise dense, imbalanced matrix: the column phase swaps it to the front (a
// real transposition), giving ilo=2, ihi=4. Exercises permutation + scaling and
// the swap kernels.
template<typename T>
void permutation_swap() {
  auto handle = shared_device();
  const int n = 4;
  const int lda = n;
  std::vector<T> A(static_cast<std::size_t>(lda) * n, elem<T>::make(0, 0));
  const auto set = [&](int r, int c, int e) {
    A[static_cast<std::size_t>(c) * lda + r] = static_cast<T>(elem<T>::make(std::ldexp(1.0, e), 0));
  };
  // diagonal
  set(0, 0, 0);
  set(1, 1, 0);
  set(2, 2, 2);
  set(3, 3, 0);
  // column 2 off-diagonal stays zero -> isolatable column.
  // row 2 is dense (so row 2 is not isolatable as a row).
  set(2, 0, 1);
  set(2, 1, 0);
  set(2, 3, 2);
  // remaining fill so no other row/column isolates, with imbalance.
  set(1, 0, 3);
  set(3, 0, -2);
  set(0, 1, 2);
  set(3, 1, 1);
  set(0, 3, 1);
  set(1, 3, -1);

  auto g = run<T>(handle, GebalJob::Both, n, A, lda);
  ASSERT_EQ(g.status, wwr::wwrSuccess);
  EXPECT_EQ(g.ilo, 2); // one column isolated to the front
  EXPECT_EQ(g.ihi, 4);
  expect_reproduces<T>(A, g, n, lda, "permutation_swap/Both");

  // Permute only: values are permuted, never scaled. Same ilo/ihi, exact reproduce
  // with an all-ones D.
  auto gp = run<T>(handle, GebalJob::Permute, n, A, lda);
  ASSERT_EQ(gp.status, wwr::wwrSuccess);
  EXPECT_EQ(gp.ilo, 2);
  EXPECT_EQ(gp.ihi, 4);
  expect_reproduces<T>(A, gp, n, lda, "permutation_swap/Permute");
}

// The one closed-form case: [[1, 2^10], [2^-10, 1]] balances to [[1,1],[1,1]]
// with D = diag(2^10, 1). Also the idempotence (stationarity) check: a second
// Scale pass over the balanced matrix must change nothing.
template<typename T>
void analytic_2x2() {
  auto handle = shared_device();
  const int n = 2;
  const int lda = n;
  std::vector<T> A(4);
  A[0] = elem<T>::make(1.0, 0.0);                      // (0,0)
  A[1] = static_cast<T>(elem<T>::make(std::ldexp(1.0, -10), 0.0)); // (1,0)
  A[2] = static_cast<T>(elem<T>::make(std::ldexp(1.0, 10), 0.0));  // (0,1)
  A[3] = elem<T>::make(1.0, 0.0);                      // (1,1)

  auto g = run<T>(handle, GebalJob::Scale, n, A, lda);
  ASSERT_EQ(g.status, wwr::wwrSuccess);
  EXPECT_EQ(g.ilo, 1);
  EXPECT_EQ(g.ihi, 2);
  for (int idx = 0; idx < 4; ++idx) {
    EXPECT_TRUE(elem<T>::eq(g.B[static_cast<std::size_t>(idx)], elem<T>::make(1.0, 0.0)))
        << "analytic_2x2 B[" << idx << "] != 1";
  }
  EXPECT_EQ(static_cast<double>(g.scale[0]), std::ldexp(1.0, 10));
  EXPECT_EQ(static_cast<double>(g.scale[1]), 1.0);
  expect_reproduces<T>(A, g, n, lda, "analytic_2x2");

  // Negative control: the first pass genuinely changed the matrix.
  EXPECT_FALSE(elem<T>::eq(g.B[2], A[2]));

  // Idempotence: Scale again over the balanced matrix -- nothing accepted.
  auto g2 = run<T>(handle, GebalJob::Scale, n, g.B, lda);
  ASSERT_EQ(g2.status, wwr::wwrSuccess);
  for (int idx = 0; idx < 4; ++idx) {
    EXPECT_TRUE(elem<T>::eq(g2.B[static_cast<std::size_t>(idx)], g.B[static_cast<std::size_t>(idx)]))
        << "idempotence B[" << idx << "] changed on second pass";
  }
  EXPECT_EQ(static_cast<double>(g2.scale[0]), 1.0);
  EXPECT_EQ(static_cast<double>(g2.scale[1]), 1.0);
}

// A diagonal matrix isolates every eigenvalue: ilo == ihi == 1, the matrix is
// untouched, and the reconstruction is trivial.
template<typename T>
void diagonal() {
  auto handle = shared_device();
  const int n = 3;
  const int lda = n;
  std::vector<T> A(static_cast<std::size_t>(lda) * n, elem<T>::make(0, 0));
  A[0] = elem<T>::make(3.0, 0.0);
  A[1 * lda + 1] = elem<T>::make(5.0, 0.0);
  A[2 * lda + 2] = elem<T>::make(7.0, 0.0);

  auto g = run<T>(handle, GebalJob::Both, n, A, lda);
  ASSERT_EQ(g.status, wwr::wwrSuccess);
  EXPECT_EQ(g.ilo, 1);
  EXPECT_EQ(g.ihi, 1);
  for (std::size_t idx = 0; idx < A.size(); ++idx) {
    EXPECT_TRUE(elem<T>::eq(g.B[idx], A[idx])) << "diagonal B[" << idx << "] changed";
  }
  expect_reproduces<T>(A, g, n, lda, "diagonal");
}

// n == 1 is the trivial fixed point: ilo == ihi == 1, nothing changes.
template<typename T>
void one_by_one() {
  auto handle = shared_device();
  const int n = 1;
  const int lda = 1;
  std::vector<T> A(1, elem<T>::make(5.0, 0.0));
  auto g = run<T>(handle, GebalJob::Both, n, A, lda);
  ASSERT_EQ(g.status, wwr::wwrSuccess);
  EXPECT_EQ(g.ilo, 1);
  EXPECT_EQ(g.ihi, 1);
  EXPECT_TRUE(elem<T>::eq(g.B[0], A[0]));
  expect_reproduces<T>(A, g, n, lda, "one_by_one");
}

} // namespace

TEST(GebalSpecTests, DenseScaling) {
  dense_scaling<float>();
  dense_scaling<double>();
  dense_scaling<wwr::wwrFloatComplex>();
  dense_scaling<wwr::wwrDoubleComplex>();
}

TEST(GebalSpecTests, PermutationSwap) {
  permutation_swap<float>();
  permutation_swap<double>();
  permutation_swap<wwr::wwrFloatComplex>();
  permutation_swap<wwr::wwrDoubleComplex>();
}

TEST(GebalSpecTests, Analytic2x2) {
  analytic_2x2<float>();
  analytic_2x2<double>();
  analytic_2x2<wwr::wwrFloatComplex>();
  analytic_2x2<wwr::wwrDoubleComplex>();
}

TEST(GebalSpecTests, Diagonal) {
  diagonal<float>();
  diagonal<double>();
  diagonal<wwr::wwrFloatComplex>();
  diagonal<wwr::wwrDoubleComplex>();
}

TEST(GebalSpecTests, OneByOne) {
  one_by_one<float>();
  one_by_one<double>();
  one_by_one<wwr::wwrFloatComplex>();
  one_by_one<wwr::wwrDoubleComplex>();
}

} // namespace calaman
