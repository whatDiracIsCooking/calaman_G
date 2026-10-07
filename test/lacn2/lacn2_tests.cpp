// Oracle test for calaman.lacn2 -- the Hager/Higham 1-norm estimator by reverse
// communication. The oracle is the reference ?lacn2 in the SAME precision,
// called through its Fortran symbol (LAPACKE does not wrap it), driven by the
// same explicit matrix: the host loop answers each kase with a host product,
// the device loop with wwr::gemv on the card.
//
// Each case checks that both loops request the same kase sequence (the state
// machines walk the same path), that the estimates and final v agree to the
// shared tolerance, and that the estimate never exceeds the exact ||A||_1
// (LAPACKE_?lange '1'). On the classes where Hager's method is exact -- n = 1,
// diagonal, nonnegative, one dominant column -- it must equal ||A||_1, and the
// reference is checked to be exact there too.
//
// REQUIRES_GPU (labeled `gpu`). Built only when calaman::lapack_reference
// exists; see this directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.wrappers.blas;
import wwr.extension.memory_buffer;
import calaman.lacn2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// The reference ?lacn2, by Fortran symbol: everything by reference, no
// CHARACTER argument and so no hidden string length.
extern "C" {
void slacn2_(const lapack_int *n, float *v, float *x, lapack_int *isgn, float *est,
             lapack_int *kase, lapack_int *isave);
void dlacn2_(const lapack_int *n, double *v, double *x, lapack_int *isgn, double *est,
             lapack_int *kase, lapack_int *isave);
}

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

// A broken state machine must fail, not hang: ?lacn2 asks for at most
// 2 + 2*(ITMAX-1) + 2 products.
constexpr int kMaxCalls = 32;

template<typename T>
struct ref;

template<>
struct ref<float> {
  static void lacn2(lapack_int n, float *v, float *x, lapack_int *isgn, float *est,
                    lapack_int *kase, lapack_int *isave) {
    slacn2_(&n, v, x, isgn, est, kase, isave);
  }
  static float lange(int n, const float *a) {
    return LAPACKE_slange(LAPACK_COL_MAJOR, '1', n, n, a, n);
  }
};

template<>
struct ref<double> {
  static void lacn2(lapack_int n, double *v, double *x, lapack_int *isgn, double *est,
                    lapack_int *kase, lapack_int *isave) {
    dlacn2_(&n, v, x, isgn, est, kase, isave);
  }
  static double lange(int n, const double *a) {
    return LAPACKE_dlange(LAPACK_COL_MAJOR, '1', n, n, a, n);
  }
};

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

template<typename T>
struct Result {
  T est{};
  std::vector<int> kases;
  std::vector<T> v;
};

// x <-- A x (kase 1) or A^T x (kase 2), column-major n-by-n, accumulated in T.
template<typename T>
void host_apply(const int kase, const int n, const std::vector<T> &a, std::vector<T> &x) {
  std::vector<T> y(static_cast<std::size_t>(n), T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const T aij = a[static_cast<std::size_t>(j) * n + i];
      if (kase == 1) {
        y[i] += aij * x[j];
      } else {
        y[j] += aij * x[i];
      }
    }
  }
  x = y;
}

template<typename T>
Result<T> run_reference(const int n, const std::vector<T> &a) {
  Result<T> r;
  r.v.assign(static_cast<std::size_t>(n), T{0});
  std::vector<T> x(static_cast<std::size_t>(n), T{0});
  std::vector<lapack_int> isgn(static_cast<std::size_t>(n), 0);
  std::array<lapack_int, 3> isave{};
  lapack_int kase = 0;
  for (int call = 0; call < kMaxCalls; ++call) {
    ref<T>::lacn2(n, r.v.data(), x.data(), isgn.data(), &r.est, &kase, isave.data());
    r.kases.push_back(kase);
    if (kase == 0) {
      break;
    }
    host_apply(kase, n, a, x);
  }
  return r;
}

template<typename T>
Result<T> run_device(const int n, const std::vector<T> &a) {
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  const std::vector<T> zeros(static_cast<std::size_t>(n), T{0});
  auto d_a = to_device(a);
  auto d_x = to_device(zeros);
  auto d_y = to_device(zeros);
  auto d_v = to_device(zeros);
  const std::size_t bytes = lacn2_bufferSize<T>(n);
  DeviceBuffer<std::byte> d_work(bytes, handle);

  Result<T> r;
  int kase = 0;
  std::array<int, 3> isave{};
  const T one{1};
  const T zero{0};
  for (int call = 0; call < kMaxCalls; ++call) {
    const Status st =
        lacn2<T>(blas, n, d_v.data(), d_x.data(), d_work.data(), bytes, r.est, kase, isave);
    EXPECT_TRUE(st.ok()) << "lacn2 returned " << st.name() << ": " << st.message();
    r.kases.push_back(kase);
    if (kase == 0 || !st.ok()) {
      break;
    }
    const auto op = kase == 1 ? wwr::WWRBLAS_OP_N : wwr::WWRBLAS_OP_T;
    EXPECT_EQ((wwr::gemv<T, int>(blas, op, n, n, &one, d_a.data(), n, d_x.data(), 1, &zero,
                                 d_y.data(), 1)),
              wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ((wwr::copy<T, int>(blas, n, d_y.data(), 1, d_x.data(), 1)),
              wwr::WWRBLAS_STATUS_SUCCESS);
  }
  r.v = to_host(d_v, static_cast<std::size_t>(n));
  wwr::wwrblasDestroy(blas);
  return r;
}

template<typename T>
void check(const std::string &what, const int n, const std::vector<T> &a, const bool exact) {
  const Result<T> got = run_device(n, a);
  const Result<T> want = run_reference(n, a);
  const T norm1 = ref<T>::lange(n, a.data());
  const T tol = factorization_tol(norm1, n, n);

  EXPECT_EQ(got.kases, want.kases) << what << ": kase sequence";
  EXPECT_NEAR(got.est, want.est, tol) << what << ": est vs reference";
  EXPECT_LE(got.est, norm1 + tol) << what << ": est exceeds ||A||_1";
  if (exact) {
    EXPECT_NEAR(want.est, norm1, tol) << what << ": reference not exact";
    EXPECT_NEAR(got.est, norm1, tol) << what << ": est vs ||A||_1";
  }

  // v = A w with |w_j| <= 2, so its error is bounded by 2 * the max row sum.
  T rowmax{0};
  for (int i = 0; i < n; ++i) {
    T s{0};
    for (int j = 0; j < n; ++j) {
      s += std::abs(a[static_cast<std::size_t>(j) * n + i]);
    }
    rowmax = std::max(rowmax, s);
  }
  const T vtol = factorization_tol(T{2} * rowmax, n, n);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(got.v[i], want.v[i], vtol) << what << ": v[" << i << "]";
  }
}

enum class Kind { Random, Nonnegative, Diagonal, DominantColumn };

template<typename T>
std::vector<T> make_matrix(const Kind kind, const int n, const unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::vector<T> a(nn, T{0});
  switch (kind) {
  case Kind::Random:
    for (auto &e : a) {
      e = static_cast<T>(dist(gen));
    }
    break;
  case Kind::Nonnegative:
    for (auto &e : a) {
      e = static_cast<T>(std::abs(dist(gen)));
    }
    break;
  case Kind::Diagonal:
    for (int i = 0; i < n; ++i) {
      a[static_cast<std::size_t>(i) * n + i] = static_cast<T>(4.0 * dist(gen));
    }
    break;
  case Kind::DominantColumn: {
    for (auto &e : a) {
      e = static_cast<T>(0.01 * dist(gen));
    }
    const int col = n / 3;
    for (int i = 0; i < n; ++i) {
      a[static_cast<std::size_t>(col) * n + i] = static_cast<T>(i % 2 == 0 ? 10.0 : -10.0);
    }
    break;
  }
  }
  return a;
}

constexpr std::array<int, 7> kSizes{1, 2, 3, 7, 16, 64, 200};

template<typename TypeParam>
void random_matches_reference() {
  for (const int n : kSizes) {
    for (unsigned seed = 1; seed <= 4; ++seed) {
      check<TypeParam>("random n=" + std::to_string(n) + " seed=" + std::to_string(seed), n,
                       make_matrix<TypeParam>(Kind::Random, n, seed), n == 1);
    }
  }
}

template<typename TypeParam>
void nonnegative_is_exact() {
  for (const int n : kSizes) {
    check<TypeParam>("nonnegative n=" + std::to_string(n), n,
                     make_matrix<TypeParam>(Kind::Nonnegative, n, 11U), true);
  }
}

template<typename TypeParam>
void diagonal_is_exact() {
  for (const int n : kSizes) {
    check<TypeParam>("diagonal n=" + std::to_string(n), n,
                     make_matrix<TypeParam>(Kind::Diagonal, n, 23U), true);
  }
}

template<typename TypeParam>
void dominant_column_is_exact() {
  for (const int n : kSizes) {
    check<TypeParam>("dominant column n=" + std::to_string(n), n,
                     make_matrix<TypeParam>(Kind::DominantColumn, n, 37U), true);
  }
}

template<typename T>
void rejects_bad_arguments() {
  const auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  constexpr int n = 8;
  auto d_x = to_device(std::vector<T>(n, T{0}));
  auto d_v = to_device(std::vector<T>(n, T{0}));
  const std::size_t bytes = lacn2_bufferSize<T>(n);
  DeviceBuffer<std::byte> d_work(bytes, handle);
  T est{};
  int kase = 0;
  std::array<int, 3> isave{};
  auto call = [&](const int nn, const std::size_t b) {
    return lacn2<T>(blas, nn, d_v.data(), d_x.data(), d_work.data(), b, est, kase, isave);
  };
  EXPECT_FALSE(call(0, bytes).ok()) << "n = 0";
  EXPECT_FALSE(call(n, bytes - 1).ok()) << "undersized workspace";
  kase = 1;
  isave[0] = 6;
  EXPECT_FALSE(call(n, bytes).ok()) << "corrupt isave";
  kase = 0;
  EXPECT_TRUE(call(n, bytes).ok()) << "fresh start";
  EXPECT_EQ(kase, 1);
  wwr::wwrblasDestroy(blas);
}

TEST(Lacn2OracleTests, RandomMatchesReferenceFloat) {
  random_matches_reference<float>();
}
TEST(Lacn2OracleTests, RandomMatchesReferenceDouble) {
  random_matches_reference<double>();
}
TEST(Lacn2OracleTests, NonnegativeIsExactFloat) {
  nonnegative_is_exact<float>();
}
TEST(Lacn2OracleTests, NonnegativeIsExactDouble) {
  nonnegative_is_exact<double>();
}
TEST(Lacn2OracleTests, DiagonalIsExactFloat) {
  diagonal_is_exact<float>();
}
TEST(Lacn2OracleTests, DiagonalIsExactDouble) {
  diagonal_is_exact<double>();
}
TEST(Lacn2OracleTests, DominantColumnIsExactFloat) {
  dominant_column_is_exact<float>();
}
TEST(Lacn2OracleTests, DominantColumnIsExactDouble) {
  dominant_column_is_exact<double>();
}
TEST(Lacn2OracleTests, RejectsBadArgumentsFloat) {
  rejects_bad_arguments<float>();
}
TEST(Lacn2OracleTests, RejectsBadArgumentsDouble) {
  rejects_bad_arguments<double>();
}

} // namespace
} // namespace calaman
