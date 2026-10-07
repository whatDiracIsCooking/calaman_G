// Oracle test for calaman.tfsm -- the triangular solve with A in Rectangular
// Full Packed storage, ?tfsm for s/d/c/z. The oracle is LAPACKE_?tfsm in the
// SAME precision, on the RFP array LAPACKE_?trttf packs from a full triangle.
//
// Every TRANSR x SIDE x UPLO x TRANS x DIAG combination runs at both parities
// of the triangular order (m for side L, n for side R), with TRANS = C for the
// complex types and T for the real ones. A is diagonally dominant -- off-
// diagonal entries scaled by 1/order -- so the solve is well conditioned and a
// tolerance failure means a wrong branch, not a hard matrix. Results are
// compared per element against the shared tolerance scaled by max |X| plus
// |alpha| max |B|.
//
// Out-of-range reads are caught by garbage: with DIAG = U the stored diagonal
// of A is 1e8, so reading it shrinks X by that factor; B sits in an ldb > m
// buffer whose padding rows hold a sentinel that must survive bit-for-bit.
// REQUIRES_GPU (labeled `gpu`).

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.tfsm;
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

// Per-type element construction, component read-out, the packer (?trttf) and
// the same-precision oracle (?tfsm). The oracle is the _work entry point:
// LAPACKE_?tfsm's NaN check sizes A by n even for side L, reading past it.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr char kTransposed = 'T';
  static float make(double re, double) { return static_cast<float>(re); }
  static double re(float v) { return v; }
  static double im(float) { return 0.0; }
  static int pack(char tr, char up, int n, const float *a, float *arf) {
    return LAPACKE_strttf(LAPACK_COL_MAJOR, tr, up, n, a, std::max(1, n), arf);
  }
  static int ref(char tr, char sd, char up, char t, char dg, int m, int n, float al,
                 const float *a, float *b, int ldb) {
    return LAPACKE_stfsm_work(LAPACK_COL_MAJOR, tr, sd, up, t, dg, m, n, al, a, b, ldb);
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr char kTransposed = 'T';
  static double make(double re, double) { return re; }
  static double re(double v) { return v; }
  static double im(double) { return 0.0; }
  static int pack(char tr, char up, int n, const double *a, double *arf) {
    return LAPACKE_dtrttf(LAPACK_COL_MAJOR, tr, up, n, a, std::max(1, n), arf);
  }
  static int ref(char tr, char sd, char up, char t, char dg, int m, int n, double al,
                 const double *a, double *b, int ldb) {
    return LAPACKE_dtfsm_work(LAPACK_COL_MAJOR, tr, sd, up, t, dg, m, n, al, a, b, ldb);
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
  static int pack(char tr, char up, int n, const C *a, C *arf) {
    return LAPACKE_ctrttf(LAPACK_COL_MAJOR, tr, up, n, reinterpret_cast<const L *>(a),
                          std::max(1, n), reinterpret_cast<L *>(arf));
  }
  static int ref(char tr, char sd, char up, char t, char dg, int m, int n, C al, const C *a,
                 C *b, int ldb) {
    L l_al;
    std::memcpy(&l_al, &al, sizeof(L));
    return LAPACKE_ctfsm_work(LAPACK_COL_MAJOR, tr, sd, up, t, dg, m, n, l_al,
                              reinterpret_cast<const L *>(a), reinterpret_cast<L *>(b), ldb);
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
  static int pack(char tr, char up, int n, const C *a, C *arf) {
    return LAPACKE_ztrttf(LAPACK_COL_MAJOR, tr, up, n, reinterpret_cast<const L *>(a),
                          std::max(1, n), reinterpret_cast<L *>(arf));
  }
  static int ref(char tr, char sd, char up, char t, char dg, int m, int n, C al, const C *a,
                 C *b, int ldb) {
    L l_al;
    std::memcpy(&l_al, &al, sizeof(L));
    return LAPACKE_ztfsm_work(LAPACK_COL_MAJOR, tr, sd, up, t, dg, m, n, l_al,
                              reinterpret_cast<const L *>(a), reinterpret_cast<L *>(b), ldb);
  }
};

constexpr double kGarbage = 1.0e8;
constexpr double kSentinel = -777.25;
constexpr int kPad = 3; // ldb = max(1, m) + kPad

struct Case {
  bool transr; // the transposed RFP layout
  Side side;
  Uplo uplo;
  bool trans; // op(A) = A**T (real) / A**H (complex)
  Diag diag;
  int m, n;
  double alpha_re, alpha_im;
};

std::string describe(const Case &c) {
  std::ostringstream os;
  os << "transr=" << (c.transr ? "T/C" : "N") << " side=" << (c.side == Side::L ? 'L' : 'R')
     << " uplo=" << (c.uplo == Uplo::U ? 'U' : 'L') << " trans=" << (c.trans ? "T/C" : "N")
     << " diag=" << (c.diag == Diag::U ? 'U' : 'N') << " m=" << c.m << " n=" << c.n
     << " alpha=(" << c.alpha_re << ", " << c.alpha_im << ")";
  return os.str();
}

template<typename T>
struct Problem {
  Case c;
  int order, ldb;
  T alpha;
  std::vector<T> arf, b;

  explicit Problem(const Case &cs)
      : c(cs), order(cs.side == Side::L ? cs.m : cs.n), ldb(std::max(1, cs.m) + kPad),
        alpha(elem<T>::make(cs.alpha_re, cs.alpha_im)) {
    std::mt19937 gen(static_cast<std::uint32_t>(
        101 * c.m + 37 * c.n + 17 * c.transr + 11 * (c.side == Side::L) +
        7 * (c.uplo == Uplo::U) + 3 * c.trans + (c.diag == Diag::U)));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    const auto no = static_cast<std::size_t>(std::max(order, 1));
    // The triangle of A, diagonally dominant; the other triangle is garbage
    // (trttf never reads it), and so is the diagonal when diag = U.
    std::vector<T> a(no * no);
    for (int j = 0; j < order; ++j) {
      for (int i = 0; i < order; ++i) {
        const double re = dist(gen);
        const double im = dist(gen);
        T &v = a[i + j * no];
        if (i == j) {
          v = c.diag == Diag::U ? elem<T>::make(kGarbage, kGarbage)
                                : elem<T>::make(2.0 + std::abs(re), im);
        } else if (c.uplo == Uplo::U ? i < j : i > j) {
          v = elem<T>::make(re / order, im / order);
        } else {
          v = elem<T>::make(kGarbage, -kGarbage);
        }
      }
    }
    arf.resize(static_cast<std::size_t>(order) * (order + 1) / 2);
    if (order > 0) {
      EXPECT_EQ(elem<T>::pack(tr(), c.uplo == Uplo::U ? 'U' : 'L', order, a.data(), arf.data()),
                0)
          << "?trttf info";
    }
    b.resize(static_cast<std::size_t>(ldb) * std::max(c.n, 1));
    for (std::size_t q = 0; q < b.size(); ++q) {
      const double re = dist(gen);
      const double im = dist(gen);
      b[q] = static_cast<int>(q % ldb) < c.m ? elem<T>::make(re, im)
                                             : elem<T>::make(kSentinel, kSentinel);
    }
  }

  char tr() const { return c.transr ? elem<T>::kTransposed : 'N'; }
};

template<typename T>
std::vector<T> run_reference(const Problem<T> &p) {
  std::vector<T> b = p.b;
  const Case &c = p.c;
  const int info = elem<T>::ref(p.tr(), c.side == Side::L ? 'L' : 'R', c.uplo == Uplo::U ? 'U' : 'L',
                                c.trans ? elem<T>::kTransposed : 'N', c.diag == Diag::U ? 'U' : 'N',
                                c.m, c.n, p.alpha, p.arf.data(), b.data(), p.ldb);
  EXPECT_EQ(info, 0) << "LAPACKE ?tfsm info";
  return b;
}

template<typename T>
std::vector<T> run_device(const Problem<T> &p) {
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  auto d_a = to_device(p.arf);
  auto d_b = to_device(p.b);
  constexpr Trans kT = elem<T>::kTransposed == 'C' ? Trans::C : Trans::T;
  const Case &c = p.c;
  const Status st = tfsm<T>(blas, c.transr ? kT : Trans::N, c.side, c.uplo,
                            c.trans ? kT : Trans::N, c.diag, c.m, c.n, p.alpha, d_a.data(),
                            d_b.data(), p.ldb);
  EXPECT_TRUE(st.ok()) << "tfsm returned " << st.name() << ": " << st.message();
  auto out = to_host(d_b, p.b.size());
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
  const auto mod = [](T v) { return std::abs(elem<T>::re(v)) + std::abs(elem<T>::im(v)); };
  double scale = 0.0;
  for (std::size_t q = 0; q < p.b.size(); ++q) {
    if (static_cast<int>(q % p.ldb) < cs.m) {
      scale = std::max(scale, mod(want[q]) + mod(p.alpha) * mod(p.b[q]));
    }
  }
  const auto k = static_cast<std::size_t>(p.order + 1);
  const double tol = factorization_tol<R>(static_cast<R>(scale), k, k);
  for (int j = 0; j < cs.n; ++j) {
    for (int i = 0; i < p.ldb; ++i) {
      const std::size_t q = i + static_cast<std::size_t>(j) * p.ldb;
      if (i >= cs.m) {
        ASSERT_EQ(elem<T>::re(got[q]), elem<T>::re(p.b[q])) << what << ": padding row " << i;
        ASSERT_EQ(elem<T>::im(got[q]), elem<T>::im(p.b[q])) << what << ": padding row " << i;
        continue;
      }
      ASSERT_TRUE(std::isfinite(elem<T>::re(want[q])) && std::isfinite(elem<T>::im(want[q])))
          << what;
      ASSERT_LE(std::abs(elem<T>::re(got[q]) - elem<T>::re(want[q])), tol)
          << what << " real part at (" << i << ", " << j << ")";
      ASSERT_LE(std::abs(elem<T>::im(got[q]) - elem<T>::im(want[q])), tol)
          << what << " imag part at (" << i << ", " << j << ")";
    }
  }
}

// Every transr x side x uplo x trans x diag branch, over the given triangular
// orders, other dimensions and alphas.
template<typename T>
void run_branches(std::initializer_list<int> orders, std::initializer_list<int> others,
                  std::initializer_list<std::pair<double, double>> alphas) {
  for (const bool transr : {false, true}) {
    for (const Side side : {Side::L, Side::R}) {
      for (const Uplo uplo : {Uplo::U, Uplo::L}) {
        for (const bool trans : {false, true}) {
          for (const Diag diag : {Diag::N, Diag::U}) {
            for (const int order : orders) {
              for (const int other : others) {
                for (const auto [are, aim] : alphas) {
                  const int m = side == Side::L ? order : other;
                  const int n = side == Side::L ? other : order;
                  check<T>({transr, side, uplo, trans, diag, m, n, are, aim});
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
  }
}

// Odd and even orders from the order-1 edge (one empty RFP block) up past a
// few BLAS tiles, with the other dimension below, near and above it.
template<typename T>
void run_shapes() {
  run_branches<T>({1, 2, 3, 4, 5, 6, 7, 16, 17, 64, 65}, {1, 5, 17}, {{1.3, -0.4}});
}

TEST(TfsmOracleTests, Float) {
  run_shapes<float>();
}
TEST(TfsmOracleTests, Double) {
  run_shapes<double>();
}
TEST(TfsmOracleTests, ComplexFloat) {
  run_shapes<wwr::wwrFloatComplex>();
}
TEST(TfsmOracleTests, ComplexDouble) {
  run_shapes<wwr::wwrDoubleComplex>();
}

// alpha = 0 zeroes B (and only B's m-by-n block); alpha = 1 and a purely
// imaginary alpha, on every branch at both parities.
template<typename T>
void run_edges() {
  run_branches<T>({4, 5}, {3}, {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.75}});
}

TEST(TfsmOracleTests, EdgeAlphas) {
  run_edges<float>();
  run_edges<double>();
  run_edges<wwr::wwrFloatComplex>();
  run_edges<wwr::wwrDoubleComplex>();
}

// m = 0 or n = 0 enqueues nothing: B is left exactly as it was.
TEST(TfsmOracleTests, EmptyLeavesBUntouched) {
  using T = wwr::wwrDoubleComplex;
  for (const Side side : {Side::L, Side::R}) {
    for (const auto [m, n] : {std::pair{0, 4}, std::pair{4, 0}, std::pair{0, 0}}) {
      const Problem<T> p({false, side, Uplo::L, false, Diag::N, m, n, 1.3, -0.4});
      const auto got = run_device(p);
      for (std::size_t s = 0; s < got.size(); ++s) {
        EXPECT_EQ(elem<T>::re(got[s]), elem<T>::re(p.b[s])) << "slot " << s;
        EXPECT_EQ(elem<T>::im(got[s]), elem<T>::im(p.b[s])) << "slot " << s;
      }
    }
  }
}

// Bad TRANSR / TRANS letters for the element type, a negative dimension, and
// a short ldb are rejected before anything is enqueued.
TEST(TfsmOracleTests, RejectsBadArguments) {
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  DeviceBuffer<wwr::wwrDoubleComplex> d_z(64, handle);
  DeviceBuffer<double> d_d(64, handle);
  const auto z = [&](Trans tr, Trans t, int m, int n, int ldb) {
    return tfsm<wwr::wwrDoubleComplex>(blas, tr, Side::L, Uplo::U, t, Diag::N, m, n,
                                       wwr::make_wwrDoubleComplex(1.0, 0.0), d_z.data(),
                                       d_z.data() + 16, ldb);
  };
  const auto d = [&](Trans tr, Trans t, int m, int n, int ldb) {
    return tfsm<double>(blas, tr, Side::R, Uplo::L, t, Diag::U, m, n, 1.0, d_d.data(),
                        d_d.data() + 16, ldb);
  };
  EXPECT_FALSE(z(Trans::T, Trans::N, 4, 2, 4).ok()) << "complex transr T";
  EXPECT_FALSE(z(Trans::N, Trans::T, 4, 2, 4).ok()) << "complex trans T";
  EXPECT_FALSE(d(Trans::C, Trans::N, 4, 2, 4).ok()) << "real transr C";
  EXPECT_FALSE(d(Trans::N, Trans::C, 4, 2, 4).ok()) << "real trans C";
  EXPECT_FALSE(z(Trans::N, Trans::N, -1, 2, 4).ok()) << "m < 0";
  EXPECT_FALSE(z(Trans::N, Trans::N, 4, -1, 4).ok()) << "n < 0";
  EXPECT_FALSE(z(Trans::N, Trans::N, 4, 2, 3).ok()) << "ldb < m";
  EXPECT_TRUE(z(Trans::C, Trans::C, 4, 2, 4).ok());
  EXPECT_TRUE(d(Trans::T, Trans::T, 4, 3, 4).ok());
  wwr::wwrStreamSynchronize(handle->stream().get());
  wwr::wwrblasDestroy(blas);
}

} // namespace
} // namespace calaman
