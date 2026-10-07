// Oracle test for calaman.lacrm -- C := A * B with A complex M-by-N, B real
// N-by-N. The oracle is LAPACKE_clacrm / LAPACKE_zlacrm in the SAME precision.
//
// Each component of C is held to the shared tolerance scaled by ITS OWN
// magnitude (sum_k |Re A(i,k)| |B(k,j)| for the real part, likewise imaginary),
// which is valid because each plane is one real gemm. That is what lets the
// skewed cases -- imaginary part 1e4 times the real part, and the reverse --
// catch a dropped or swapped plane in both precisions.
//
// Out-of-range reads are caught by garbage: A's rows m..lda-1 and B's rows
// n..ldb-1 are NaN, so a stray read turns a result into NaN; C starts NaN in
// range (lacrm must not read it) and holds a finite sentinel in its padding
// rows, which must survive bit-for-bit. REQUIRES_GPU (labeled `gpu`).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.lacrm;
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

// Per-type element construction, component read-out and the same-precision oracle.
template<typename T>
struct elem;

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  using C = wwr::wwrFloatComplex;
  static C make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static double re(C v) { return wwr::wwrCrealf(v); }
  static double im(C v) { return wwr::wwrCimagf(v); }
  static int ref(int m, int n, const C *a, int lda, const float *b, int ldb, C *c, int ldc) {
    return LAPACKE_clacrm(LAPACK_COL_MAJOR, m, n, reinterpret_cast<const lapack_complex_float *>(a),
                          lda, b, ldb, reinterpret_cast<lapack_complex_float *>(c), ldc);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double re(C v) { return wwr::wwrCreal(v); }
  static double im(C v) { return wwr::wwrCimag(v); }
  static int ref(int m, int n, const C *a, int lda, const double *b, int ldb, C *c, int ldc) {
    return LAPACKE_zlacrm(LAPACK_COL_MAJOR, m, n,
                          reinterpret_cast<const lapack_complex_double *>(a), lda, b, ldb,
                          reinterpret_cast<lapack_complex_double *>(c), ldc);
  }
};

constexpr double kSentinel = -777.25;

struct Shape {
  int m, n, lda, ldb, ldc;
};

// One problem from one seed. re_scale / im_scale skew A's two components.
template<typename T>
struct Problem {
  using R = typename elem<T>::R;
  Shape s;
  std::vector<T> a, c;
  std::vector<R> b;

  Problem(Shape shape, std::uint32_t seed, double re_scale = 1.0, double im_scale = 1.0)
      : s(shape), a(static_cast<std::size_t>(s.lda) * s.n + 1),
        c(static_cast<std::size_t>(s.ldc) * s.n + 1), b(static_cast<std::size_t>(s.ldb) * s.n + 1) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const auto in = [](std::size_t k, int ld, int rows, int cols) {
      return static_cast<int>(k % ld) < rows && static_cast<int>(k / ld) < cols;
    };
    for (std::size_t k = 0; k < a.size(); ++k) {
      const double re = dist(gen) * re_scale;
      const double im = dist(gen) * im_scale;
      a[k] = in(k, s.lda, s.m, s.n) ? elem<T>::make(re, im) : elem<T>::make(nan, nan);
    }
    for (std::size_t k = 0; k < b.size(); ++k) {
      b[k] = in(k, s.ldb, s.n, s.n) ? static_cast<R>(dist(gen)) : static_cast<R>(nan);
    }
    for (std::size_t k = 0; k < c.size(); ++k) {
      c[k] = in(k, s.ldc, s.m, s.n) ? elem<T>::make(nan, nan) : elem<T>::make(kSentinel, kSentinel);
    }
  }

  // sum_k |part(A(i,k))| |B(k,j)|: the magnitude one component's error scales with.
  template<typename Part>
  double magnitude(std::size_t i, std::size_t j, Part part) const {
    double m = 0.0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(s.n); ++k) {
      m += std::abs(part(a[i + k * s.lda])) * std::abs(static_cast<double>(b[k + j * s.ldb]));
    }
    return m;
  }
};

template<typename T>
std::vector<T> run_reference(const Problem<T> &p) {
  std::vector<T> c = p.c;
  if (p.s.m > 0 && p.s.n > 0) {
    const int info =
        elem<T>::ref(p.s.m, p.s.n, p.a.data(), p.s.lda, p.b.data(), p.s.ldb, c.data(), p.s.ldc);
    EXPECT_EQ(info, 0) << "LAPACKE ?lacrm info";
  }
  return c;
}

template<typename T>
std::vector<T> run_device(const Problem<T> &p) {
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  auto d_a = to_device(p.a);
  auto d_b = to_device(p.b);
  auto d_c = to_device(p.c);
  const std::size_t bytes = lacrm_bufferSize<T>(p.s.m, p.s.n);
  DeviceBuffer<std::byte> d_work(bytes == 0 ? 1 : bytes, handle);
  const Status st = lacrm<T>(blas, p.s.m, p.s.n, d_a.data(), p.s.lda, d_b.data(), p.s.ldb,
                             d_c.data(), p.s.ldc, d_work.data(), bytes);
  EXPECT_TRUE(st.ok()) << "lacrm returned " << st.name() << ": " << st.message();
  auto out = to_host(d_c, p.c.size());
  wwr::wwrblasDestroy(blas);
  return out;
}

// Each in-range component within the shared tolerance of the oracle's, scaled by
// that component's own magnitude; padding exact.
template<typename T>
void expect_close(const Problem<T> &p, const std::vector<T> &got, const std::vector<T> &want,
                  const std::string &what) {
  using R = typename elem<T>::R;
  const auto re = [](const T &v) { return elem<T>::re(v); };
  const auto im = [](const T &v) { return elem<T>::im(v); };
  for (std::size_t k = 0; k < got.size(); ++k) {
    const std::size_t i = k % p.s.ldc;
    const std::size_t j = k / p.s.ldc;
    if (static_cast<int>(i) < p.s.m && static_cast<int>(j) < p.s.n) {
      const auto n = static_cast<std::size_t>(p.s.n);
      const double tol_re = factorization_tol<R>(static_cast<R>(p.magnitude(i, j, re)), n, n);
      const double tol_im = factorization_tol<R>(static_cast<R>(p.magnitude(i, j, im)), n, n);
      ASSERT_TRUE(std::isfinite(re(want[k])) && std::isfinite(im(want[k]))) << what;
      ASSERT_LE(std::abs(re(got[k]) - re(want[k])), tol_re)
          << what << " real part at (" << i << ", " << j << ")";
      ASSERT_LE(std::abs(im(got[k]) - im(want[k])), tol_im)
          << what << " imag part at (" << i << ", " << j << ")";
    } else {
      ASSERT_EQ(re(got[k]), re(want[k])) << what << ": padding slot " << k << " was written";
      ASSERT_EQ(im(got[k]), im(want[k])) << what << ": padding slot " << k << " was written";
    }
  }
}

template<typename T>
void check(Shape s, double re_scale = 1.0, double im_scale = 1.0) {
  const Problem<T> p(s, static_cast<std::uint32_t>(31 * s.m + 7 * s.n + s.lda), re_scale,
                     im_scale);
  std::ostringstream what;
  what << "m=" << s.m << " n=" << s.n << " lda=" << s.lda << " ldb=" << s.ldb
       << " ldc=" << s.ldc << " re_scale=" << re_scale << " im_scale=" << im_scale;
  expect_close(p, run_device(p), run_reference(p), what.str());
}

// Rectangular both ways, square, M = N = 1, and sizes past a 128-thread block,
// with lda, ldb and ldc padded and different from each other.
constexpr Shape kShapes[] = {{1, 1, 1, 1, 1},     {1, 1, 3, 2, 4},       {5, 3, 7, 4, 9},
                             {3, 5, 4, 8, 6},     {17, 9, 20, 11, 19},   {64, 64, 64, 64, 64},
                             {130, 33, 131, 40, 135}, {200, 7, 203, 9, 201}, {9, 150, 12, 151, 10}};

template<typename T>
void run_shapes() {
  for (const Shape s : kShapes) {
    check<T>(s);
    if (testing::Test::HasFatalFailure()) {
      return;
    }
  }
}

TEST(LacrmOracleTests, ComplexFloat) {
  run_shapes<wwr::wwrFloatComplex>();
}
TEST(LacrmOracleTests, ComplexDouble) {
  run_shapes<wwr::wwrDoubleComplex>();
}

// A's imaginary part 1e4 times its real part (and the reverse): with per-component
// tolerances a dropped, zeroed or swapped plane cannot pass.
TEST(LacrmOracleTests, SkewedComponents) {
  for (const Shape s : {Shape{6, 4, 8, 5, 7}, Shape{33, 17, 35, 19, 34}}) {
    check<wwr::wwrFloatComplex>(s, 1.0, 1e4);
    check<wwr::wwrFloatComplex>(s, 1e4, 1.0);
    check<wwr::wwrDoubleComplex>(s, 1.0, 1e4);
    check<wwr::wwrDoubleComplex>(s, 1e4, 1.0);
  }
}

// M = 0, N = 0 or both: nothing is written, as ZLACRM returns at once.
TEST(LacrmOracleTests, EmptyLeavesCUntouched) {
  using T = wwr::wwrDoubleComplex;
  for (const Shape s : {Shape{0, 3, 1, 3, 1}, Shape{4, 0, 4, 1, 5}, Shape{0, 0, 1, 1, 1}}) {
    EXPECT_EQ(lacrm_bufferSize<T>(s.m, s.n), 0u);
    const Problem<T> p(s, 5);
    const auto got = run_device(p);
    for (std::size_t k = 0; k < got.size(); ++k) {
      EXPECT_EQ(elem<T>::re(got[k]), elem<T>::re(p.c[k])) << "slot " << k;
      EXPECT_EQ(elem<T>::im(got[k]), elem<T>::im(p.c[k])) << "slot " << k;
    }
  }
}

// Bad leading dimensions and an undersized workspace are rejected up front.
TEST(LacrmOracleTests, RejectsBadArguments) {
  using T = wwr::wwrDoubleComplex;
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  const Problem<T> p({4, 3, 4, 3, 4}, 9);
  auto d_a = to_device(p.a);
  auto d_b = to_device(p.b);
  auto d_c = to_device(p.c);
  const std::size_t bytes = lacrm_bufferSize<T>(4, 3);
  DeviceBuffer<std::byte> d_work(bytes, handle);
  const auto call = [&](int lda, int ldb, int ldc, std::size_t work) {
    return lacrm<T>(blas, 4, 3, d_a.data(), lda, d_b.data(), ldb, d_c.data(), ldc, d_work.data(),
                    work);
  };
  EXPECT_FALSE(call(3, 3, 4, bytes).ok()) << "lda < m";
  EXPECT_FALSE(call(4, 2, 4, bytes).ok()) << "ldb < n";
  EXPECT_FALSE(call(4, 3, 3, bytes).ok()) << "ldc < m";
  EXPECT_FALSE(call(4, 3, 4, bytes - 1).ok()) << "undersized workspace";
  EXPECT_TRUE(call(4, 3, 4, bytes).ok());
  wwr::wwrStreamSynchronize(handle->stream().get());
  wwr::wwrblasDestroy(blas);
}

} // namespace
} // namespace calaman
