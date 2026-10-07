// Oracle test for calaman.hfrk -- the rank-k update of a Hermitian (c/z, ?hfrk)
// or symmetric (s/d, ?sfrk) matrix in Rectangular Full Packed storage. The
// oracle is LAPACKE_?hfrk / LAPACKE_?sfrk in the SAME precision.
//
// Every TRANSR x UPLO x TRANS combination runs at both parities of n. Both
// results are unpacked by LAPACKE_?tfttr and compared per element against the
// shared tolerance scaled by that element's own magnitude,
// |alpha| sum_l |op(A)(i,l)| |op(A)(j,l)| + |beta| |C(i,j)|, per component.
// The input C is random complex, so its diagonal carries a non-zero imaginary
// part that the update must drop, and a missed conjugation shows up as an
// O(1) error in the imaginary parts.
//
// Out-of-range reads are caught by garbage: A's padding rows hold a large
// finite value, and the device C carries trailing sentinels past its n(n+1)/2
// elements that must survive bit-for-bit. REQUIRES_GPU (labeled `gpu`).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.hfrk;
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
  const std::size_t n = std::max<std::size_t>(host.size(), 1);
  HostBuffer<T> staging(n);
  std::copy(host.begin(), host.end(), staging.data());
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, staging, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return device;
}

template<typename T>
std::vector<T> to_host(const DeviceBuffer<T> &device, std::size_t n) {
  const auto handle = shared_device();
  HostBuffer<T> host(std::max<std::size_t>(n, 1));
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

// Per-type element construction, component read-out and the same-precision
// oracle (?hfrk for complex, ?sfrk for real) plus ?tfttr to unpack a result.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr char kTransposed = 'T';
  static float make(double re, double) { return static_cast<float>(re); }
  static double re(float v) { return v; }
  static double im(float) { return 0.0; }
  static int ref(char tr, char up, char t, int n, int k, float al, const float *a, int lda,
                 float be, float *c) {
    return LAPACKE_ssfrk(LAPACK_COL_MAJOR, tr, up, t, n, k, al, a, lda, be, c);
  }
  static int unpack(char tr, char up, int n, const float *arf, float *a) {
    return LAPACKE_stfttr(LAPACK_COL_MAJOR, tr, up, n, arf, a, std::max(1, n));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr char kTransposed = 'T';
  static double make(double re, double) { return re; }
  static double re(double v) { return v; }
  static double im(double) { return 0.0; }
  static int ref(char tr, char up, char t, int n, int k, double al, const double *a, int lda,
                 double be, double *c) {
    return LAPACKE_dsfrk(LAPACK_COL_MAJOR, tr, up, t, n, k, al, a, lda, be, c);
  }
  static int unpack(char tr, char up, int n, const double *arf, double *a) {
    return LAPACKE_dtfttr(LAPACK_COL_MAJOR, tr, up, n, arf, a, std::max(1, n));
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  using C = wwr::wwrFloatComplex;
  using L = lapack_complex_float;
  static constexpr char kTransposed = 'C';
  static C make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static double re(C v) { return wwr::wwrCrealf(v); }
  static double im(C v) { return wwr::wwrCimagf(v); }
  static int ref(char tr, char up, char t, int n, int k, float al, const C *a, int lda, float be,
                 C *c) {
    return LAPACKE_chfrk(LAPACK_COL_MAJOR, tr, up, t, n, k, al, reinterpret_cast<const L *>(a), lda,
                         be, reinterpret_cast<L *>(c));
  }
  static int unpack(char tr, char up, int n, const C *arf, C *a) {
    return LAPACKE_ctfttr(LAPACK_COL_MAJOR, tr, up, n, reinterpret_cast<const L *>(arf),
                          reinterpret_cast<L *>(a), std::max(1, n));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  using L = lapack_complex_double;
  static constexpr char kTransposed = 'C';
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double re(C v) { return wwr::wwrCreal(v); }
  static double im(C v) { return wwr::wwrCimag(v); }
  static int ref(char tr, char up, char t, int n, int k, double al, const C *a, int lda, double be,
                 C *c) {
    return LAPACKE_zhfrk(LAPACK_COL_MAJOR, tr, up, t, n, k, al, reinterpret_cast<const L *>(a), lda,
                         be, reinterpret_cast<L *>(c));
  }
  static int unpack(char tr, char up, int n, const C *arf, C *a) {
    return LAPACKE_ztfttr(LAPACK_COL_MAJOR, tr, up, n, reinterpret_cast<const L *>(arf),
                          reinterpret_cast<L *>(a), std::max(1, n));
  }
};

constexpr double kGarbage = 1.0e15;
constexpr double kSentinel = -777.25;
constexpr std::size_t kSlack = 7;

struct Case {
  bool transr; // the transposed RFP layout
  Uplo uplo;
  bool trans; // C += alpha A**H A rather than alpha A A**H
  int n, k;
  double alpha, beta;
};

std::string describe(const Case &c) {
  std::ostringstream os;
  os << "transr=" << (c.transr ? "T/C" : "N") << " uplo=" << (c.uplo == Uplo::U ? 'U' : 'L')
     << " trans=" << (c.trans ? "T/C" : "N") << " n=" << c.n << " k=" << c.k << " alpha=" << c.alpha
     << " beta=" << c.beta;
  return os.str();
}

template<typename T>
struct Problem {
  using R = typename elem<T>::R;
  Case c;
  int rows, cols, lda; // A is rows x cols, op(A) is n x k
  std::vector<T> a, arf;

  explicit Problem(const Case &cs)
      : c(cs), rows(cs.trans ? cs.k : cs.n), cols(cs.trans ? cs.n : cs.k),
        lda(std::max(1, rows) + 2) {
    std::mt19937 gen(static_cast<std::uint32_t>(97 * c.n + 13 * c.k + 5 * c.transr +
                                                3 * (c.uplo == Uplo::U) + c.trans));
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    a.resize(static_cast<std::size_t>(lda) * std::max(cols, 1));
    for (std::size_t q = 0; q < a.size(); ++q) {
      const bool in = static_cast<int>(q % lda) < rows;
      const double re = dist(gen);
      const double im = dist(gen);
      a[q] = in ? elem<T>::make(re, im) : elem<T>::make(kGarbage, kGarbage);
    }
    arf.resize(static_cast<std::size_t>(c.n) * (c.n + 1) / 2);
    for (T &v : arf) {
      const double re = dist(gen);
      const double im = dist(gen);
      v = elem<T>::make(re, im);
    }
  }

  char tr() const { return c.transr ? elem<T>::kTransposed : 'N'; }
  char up() const { return c.uplo == Uplo::U ? 'U' : 'L'; }
  char tc() const { return c.trans ? elem<T>::kTransposed : 'N'; }

  // op(A)(i, l): A(i, l) for trans N, conj(A(l, i)) otherwise; only |.| is used.
  double op_abs(int i, int l, double (*part)(T)) const {
    const T v = c.trans ? a[l + static_cast<std::size_t>(i) * lda]
                        : a[i + static_cast<std::size_t>(l) * lda];
    return std::abs(part(v));
  }

  std::vector<T> full(const std::vector<T> &rfp) const {
    std::vector<T> f(static_cast<std::size_t>(std::max(c.n, 1)) * std::max(c.n, 1));
    if (c.n > 0) {
      EXPECT_EQ(elem<T>::unpack(tr(), up(), c.n, rfp.data(), f.data()), 0) << "?tfttr info";
    }
    return f;
  }
};

template<typename T>
std::vector<T> run_reference(const Problem<T> &p) {
  using R = typename elem<T>::R;
  std::vector<T> c = p.arf;
  const int info = elem<T>::ref(p.tr(), p.up(), p.tc(), p.c.n, p.c.k, static_cast<R>(p.c.alpha),
                                p.a.data(), p.lda, static_cast<R>(p.c.beta), c.data());
  EXPECT_EQ(info, 0) << "LAPACKE ?hfrk/?sfrk info";
  return c;
}

template<typename T>
std::vector<T> run_device(const Problem<T> &p) {
  using R = typename elem<T>::R;
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  std::vector<T> c = p.arf;
  for (std::size_t s = 0; s < kSlack; ++s) {
    c.push_back(elem<T>::make(kSentinel, kSentinel));
  }
  auto d_a = to_device(p.a);
  auto d_c = to_device(c);
  constexpr Trans kT = elem<T>::kTransposed == 'C' ? Trans::C : Trans::T;
  const Status st =
      hfrk<T>(blas, p.c.transr ? kT : Trans::N, p.c.uplo, p.c.trans ? kT : Trans::N, p.c.n, p.c.k,
              static_cast<R>(p.c.alpha), d_a.data(), p.lda, static_cast<R>(p.c.beta), d_c.data());
  EXPECT_TRUE(st.ok()) << "hfrk returned " << st.name() << ": " << st.message();
  auto out = to_host(d_c, c.size());
  wwr::wwrblasDestroy(blas);
  return out;
}

template<typename T>
void check(const Case &cs) {
  using R = typename elem<T>::R;
  const Problem<T> p(cs);
  const std::string what = describe(cs);
  const auto got = run_device(p);
  const auto want = run_reference(p);
  const std::size_t nt = p.arf.size();
  for (std::size_t s = 0; s < kSlack; ++s) {
    ASSERT_EQ(elem<T>::re(got[nt + s]), kSentinel) << what << ": slack slot " << s << " written";
    ASSERT_EQ(elem<T>::im(got[nt + s]), elem<T>::im(elem<T>::make(kSentinel, kSentinel)))
        << what << ": slack slot " << s << " written";
  }
  const std::vector<T> got_rfp(got.begin(), got.begin() + nt);
  const auto g = p.full(got_rfp);
  const auto w = p.full(want);
  const auto c0 = p.full(p.arf);
  const auto n = static_cast<std::size_t>(std::max(cs.n, 1));
  const auto kk = static_cast<std::size_t>(cs.k + 1);
  const auto mod = [](T v) { return std::abs(elem<T>::re(v)) + std::abs(elem<T>::im(v)); };
  for (int j = 0; j < cs.n; ++j) {
    for (int i = 0; i < cs.n; ++i) {
      if (cs.uplo == Uplo::U ? i > j : i < j) {
        continue;
      }
      double mag = std::abs(cs.beta) * mod(c0[i + j * n]);
      for (int l = 0; l < cs.k; ++l) {
        const auto row = [&](int r) {
          return p.op_abs(r, l, elem<T>::re) + p.op_abs(r, l, elem<T>::im);
        };
        mag += std::abs(cs.alpha) * row(i) * row(j);
      }
      const double tol = factorization_tol<R>(static_cast<R>(mag), kk, kk);
      const T gv = g[i + j * n];
      const T wv = w[i + j * n];
      ASSERT_TRUE(std::isfinite(elem<T>::re(wv)) && std::isfinite(elem<T>::im(wv))) << what;
      ASSERT_LE(std::abs(elem<T>::re(gv) - elem<T>::re(wv)), tol)
          << what << " real part at (" << i << ", " << j << ")";
      ASSERT_LE(std::abs(elem<T>::im(gv) - elem<T>::im(wv)), tol)
          << what << " imag part at (" << i << ", " << j << ")";
    }
  }
}

// Every layout x trans branch, run over the given (n, k, alpha, beta) grid.
template<typename T>
void run_branches(std::initializer_list<int> ns, std::initializer_list<int> ks,
                  std::initializer_list<std::pair<double, double>> scalars) {
  for (const bool transr : {false, true}) {
    for (const Uplo uplo : {Uplo::U, Uplo::L}) {
      for (const bool trans : {false, true}) {
        for (const int n : ns) {
          for (const int k : ks) {
            for (const auto [alpha, beta] : scalars) {
              check<T>({transr, uplo, trans, n, k, alpha, beta});
              if (testing::Test::HasFatalFailure()) {
                return;
              }
            }
          }
        }
      }
    }
  }
}

// Odd and even orders from the n = 1 edge up past a few BLAS tiles, with k
// below, near and above n.
template<typename T>
void run_shapes() {
  run_branches<T>({1, 2, 3, 4, 5, 6, 7, 16, 17, 64, 65}, {1, 5, 17}, {{1.3, -0.7}});
}

TEST(HfrkOracleTests, Float) {
  run_shapes<float>();
}
TEST(HfrkOracleTests, Double) {
  run_shapes<double>();
}
TEST(HfrkOracleTests, ComplexFloat) {
  run_shapes<wwr::wwrFloatComplex>();
}
TEST(HfrkOracleTests, ComplexDouble) {
  run_shapes<wwr::wwrDoubleComplex>();
}

// k = 0 (pure beta scaling), alpha = 0, beta = 0, beta = 1, and both zero, on
// every branch at both parities.
template<typename T>
void run_edges() {
  run_branches<T>({4, 5}, {0}, {{1.3, 0.5}, {1.3, 1.0}, {1.3, 0.0}});
  run_branches<T>({4, 5}, {3},
                  {{0.0, 0.5}, {0.0, 1.0}, {0.0, 0.0}, {1.3, 0.0}, {1.3, 1.0}, {-2.0, 1.0}});
}

TEST(HfrkOracleTests, EdgeScalars) {
  run_edges<float>();
  run_edges<double>();
  run_edges<wwr::wwrFloatComplex>();
  run_edges<wwr::wwrDoubleComplex>();
}

// n = 0 enqueues nothing: the device array is left exactly as it was.
TEST(HfrkOracleTests, EmptyLeavesCUntouched) {
  using T = wwr::wwrDoubleComplex;
  for (const int k : {0, 3}) {
    const Problem<T> p({false, Uplo::L, false, 0, k, 1.3, -0.7});
    const auto got = run_device(p);
    for (std::size_t s = 0; s < got.size(); ++s) {
      EXPECT_EQ(elem<T>::re(got[s]), kSentinel) << "slot " << s;
    }
  }
}

// Bad TRANSR / TRANS letters for the element type, a negative order, and a
// short lda are rejected before anything is enqueued.
TEST(HfrkOracleTests, RejectsBadArguments) {
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<wwr::wwrDoubleComplex> d_z(64, handle);
  DeviceBuffer<double> d_d(64, handle);
  const auto z = [&](Trans tr, Trans t, int n, int k, int lda) {
    return hfrk<wwr::wwrDoubleComplex>(blas, tr, Uplo::U, t, n, k, 1.0, d_z.data(), lda, 0.5,
                                       d_z.data());
  };
  const auto d = [&](Trans tr, Trans t, int n, int k, int lda) {
    return sfrk<double>(blas, tr, Uplo::L, t, n, k, 1.0, d_d.data(), lda, 0.5, d_d.data());
  };
  EXPECT_FALSE(z(Trans::T, Trans::N, 4, 2, 4).ok()) << "complex transr T";
  EXPECT_FALSE(z(Trans::N, Trans::T, 4, 2, 4).ok()) << "complex trans T";
  EXPECT_FALSE(d(Trans::C, Trans::N, 4, 2, 4).ok()) << "real transr C";
  EXPECT_FALSE(d(Trans::N, Trans::C, 4, 2, 4).ok()) << "real trans C";
  EXPECT_FALSE(z(Trans::N, Trans::N, -1, 2, 4).ok()) << "n < 0";
  EXPECT_FALSE(z(Trans::N, Trans::N, 4, -1, 4).ok()) << "k < 0";
  EXPECT_FALSE(z(Trans::N, Trans::N, 4, 2, 3).ok()) << "lda < n, trans N";
  EXPECT_FALSE(d(Trans::N, Trans::T, 4, 3, 2).ok()) << "lda < k, trans T";
  EXPECT_TRUE(z(Trans::C, Trans::C, 4, 2, 2).ok());
  EXPECT_TRUE(d(Trans::T, Trans::T, 4, 3, 3).ok());
  wwr::wwrStreamSynchronize(handle->stream().get());
  wwr::wwrblasDestroy(blas);
}

} // namespace
} // namespace calaman
