// Oracle test for calaman.aberth -- every root of a batch of polynomials by the
// Aberth-Ehrlich iteration, one warp per polynomial.
//
// The oracle is the reference LAPACK: the roots of p are the eigenvalues of its
// companion matrix, computed by LAPACKE_zgeev in DOUBLE for every precision
// under test (the numpy.roots construction). Two checks per root:
//   - backward error |p(z)| / sum |a_k| |z|^k, evaluated in double against the
//     coefficients as given -- conditioning-free, so it is held to O(n eps);
//   - forward agreement with the matched geev eigenvalue, to 100 n eps: looser,
//     since both methods' forward error scales with the root's conditioning.
// Exact cases (roots of unity, a zero root) pin the forward error directly.
//
// REQUIRES_GPU (see CMakeLists.txt): every case runs the kernel.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.aberth;
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

using cd = std::complex<double>;

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

// Host conversions between each element type and std::complex<double>, plus the
// complex root type CT and real type R for a coefficient type T.
template<typename T>
struct elem;

template<>
struct elem<float> {
  using CT = wwr::wwrFloatComplex;
  using R = float;
  static constexpr bool kComplex = false;
  static float from(cd v) { return static_cast<float>(v.real()); }
  static cd to(float v) { return {v, 0.0}; }
};

template<>
struct elem<double> {
  using CT = wwr::wwrDoubleComplex;
  using R = double;
  static constexpr bool kComplex = false;
  static double from(cd v) { return v.real(); }
  static cd to(double v) { return {v, 0.0}; }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using CT = wwr::wwrFloatComplex;
  using R = float;
  static constexpr bool kComplex = true;
  static CT from(cd v) {
    return wwr::make_wwrFloatComplex(static_cast<float>(v.real()), static_cast<float>(v.imag()));
  }
  static cd to(CT v) { return {wwr::wwrCrealf(v), wwr::wwrCimagf(v)}; }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using CT = wwr::wwrDoubleComplex;
  using R = double;
  static constexpr bool kComplex = true;
  static CT from(cd v) { return wwr::make_wwrDoubleComplex(v.real(), v.imag()); }
  static cd to(CT v) { return {wwr::wwrCreal(v), wwr::wwrCimag(v)}; }
};

template<typename T>
using root_t = typename elem<T>::CT;

template<typename R>
constexpr R eps() {
  return std::numeric_limits<R>::epsilon();
}

// Eigenvalues of the companion matrix of a_0 + ... + a_n z^n, by LAPACKE_zgeev.
std::vector<cd> companion_roots(const std::vector<cd> &a) {
  const int n = static_cast<int>(a.size()) - 1;
  std::vector<cd> c(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 1; i < n; ++i) {
    c[static_cast<std::size_t>(i - 1) * n + i] = 1.0; // subdiagonal, column-major
  }
  for (int i = 0; i < n; ++i) {
    c[static_cast<std::size_t>(n - 1) * n + i] = -a[i] / a[n]; // last column
  }
  std::vector<cd> w(n);
  const int info = LAPACKE_zgeev(
      LAPACK_COL_MAJOR, 'N', 'N', n, reinterpret_cast<lapack_complex_double *>(c.data()), n,
      reinterpret_cast<lapack_complex_double *>(w.data()), nullptr, 1, nullptr, 1);
  EXPECT_EQ(info, 0) << "zgeev failed";
  return w;
}

// |p(z)| / sum |a_k| |z|^k, in double.
double backward_error(const std::vector<cd> &a, cd z) {
  cd p = 0.0;
  double bound = 0.0;
  for (std::size_t k = a.size(); k-- > 0;) {
    p = p * z + a[k];
    bound = bound * std::abs(z) + std::abs(a[k]);
  }
  return bound == 0.0 ? 0.0 : std::abs(p) / bound;
}

// Greedy nearest-unused matching; the largest distance, relative to max(1, |ref|).
double max_match_error(const std::vector<cd> &got, const std::vector<cd> &ref) {
  std::vector<bool> used(ref.size(), false);
  double worst = 0.0;
  for (const cd z : got) {
    std::size_t best = ref.size();
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < ref.size(); ++j) {
      if (!used[j] && std::abs(z - ref[j]) < best_d) {
        best_d = std::abs(z - ref[j]);
        best = j;
      }
    }
    if (best == ref.size()) {
      return std::numeric_limits<double>::infinity(); // NaN root: nothing matched
    }
    used[best] = true;
    worst = std::max(worst, best_d / std::max(1.0, std::abs(ref[best])));
  }
  return worst;
}

struct Result {
  std::vector<std::vector<cd>> roots; // per polynomial
  std::vector<int> iters;
};

// Run aberth on @p polys (each n+1 ascending coefficients, already in T).
template<typename T>
Result run(const std::vector<std::vector<T>> &polys, const typename elem<T>::R tol,
           const int max_iter) {
  using CT = root_t<T>;
  auto handle = shared_device();
  const int batch = static_cast<int>(polys.size());
  const int n = static_cast<int>(polys.front().size()) - 1;
  std::vector<T> flat;
  for (const auto &p : polys) {
    flat.insert(flat.end(), p.begin(), p.end());
  }
  auto d_coeffs = to_device(handle, flat);
  DeviceBuffer<CT> d_roots(static_cast<std::size_t>(batch) * n, handle);
  DeviceBuffer<int> d_iters(static_cast<std::size_t>(batch), handle);

  const auto status = calaman::aberth(handle->stream().get(), n, batch, d_coeffs.data(),
                                      d_roots.data(), d_iters.data(), tol, max_iter);
  EXPECT_TRUE(status.ok()) << "aberth returned status=" << status.name();
  wwr::wwrStreamSynchronize(handle->stream().get());

  const auto roots = from_device(handle, d_roots, static_cast<std::size_t>(batch) * n);
  Result r;
  r.iters = from_device(handle, d_iters, static_cast<std::size_t>(batch));
  for (int b = 0; b < batch; ++b) {
    std::vector<cd> rb;
    for (int i = 0; i < n; ++i) {
      rb.push_back(elem<CT>::to(roots[static_cast<std::size_t>(b) * n + i]));
    }
    r.roots.push_back(std::move(rb));
  }
  return r;
}

// Gaussian coefficients (complex for a complex T), rounded into T.
template<typename T>
std::vector<std::vector<T>> random_polys(const int n, const int batch, const unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> nd;
  std::vector<std::vector<T>> polys(batch, std::vector<T>(n + 1));
  for (auto &p : polys) {
    for (auto &a : p) {
      a = elem<T>::from(elem<T>::kComplex ? cd{nd(gen), nd(gen)} : cd{nd(gen), 0.0});
    }
  }
  return polys;
}

// Random batches across degrees that straddle the warp width, against geev.
template<typename T>
void matches_companion_geev() {
  using R = typename elem<T>::R;
  for (const int n : {1, 2, 3, 7, 31, 32, 33, 64, 100}) {
    // 9 polynomials: more than one block of warps, and a partial last block.
    const auto polys = random_polys<T>(n, 9, 1234u + static_cast<unsigned>(n));
    const auto got = run<T>(polys, eps<R>(), 500);
    for (std::size_t b = 0; b < polys.size(); ++b) {
      std::vector<cd> a;
      for (const T &c : polys[b]) {
        a.push_back(elem<T>::to(c));
      }
      const auto ctx = "n=" + std::to_string(n) + " b=" + std::to_string(b);
      EXPECT_GE(got.iters[b], 1) << ctx << ": did not converge";
      double be = 0.0;
      for (const cd z : got.roots[b]) {
        be = std::max(be, backward_error(a, z));
      }
      EXPECT_LE(be, R{8} * static_cast<R>(n) * eps<R>()) << ctx << ": backward error";
      const double fe = max_match_error(got.roots[b], companion_roots(a));
      EXPECT_LE(fe, 100.0 * n * eps<R>()) << ctx << ": vs geev";
    }
  }
}

// z^n - 1: the n-th roots of unity, exactly.
template<typename T>
void roots_of_unity() {
  using R = typename elem<T>::R;
  const int n = 50;
  std::vector<T> p(n + 1, elem<T>::from(0.0));
  p[0] = elem<T>::from(-1.0);
  p[n] = elem<T>::from(1.0);
  const auto got = run<T>({p}, eps<R>(), 500);
  EXPECT_GE(got.iters[0], 1);
  std::vector<cd> exact;
  for (int k = 0; k < n; ++k) {
    exact.push_back(std::polar(1.0, 2.0 * std::numbers::pi * k / n));
  }
  EXPECT_LE(max_match_error(got.roots[0], exact), 16.0 * eps<R>());
}

// z^3 - z: roots 0 and +-1, the zero root exercising the relative-correction
// test where |z| -> 0.
template<typename T>
void zero_root() {
  using R = typename elem<T>::R;
  const std::vector<T> p = {elem<T>::from(0.0), elem<T>::from(-1.0), elem<T>::from(0.0),
                            elem<T>::from(1.0)};
  const auto got = run<T>({p}, eps<R>(), 500);
  EXPECT_GE(got.iters[0], 1);
  EXPECT_LE(max_match_error(got.roots[0], {0.0, 1.0, -1.0}), 4.0 * eps<R>());
}

// max_iter too small to settle, and a NaN coefficient: both report -1.
template<typename T>
void reports_non_convergence() {
  using R = typename elem<T>::R;
  auto polys = random_polys<T>(20, 2, 99u);
  polys[1][5] = elem<T>::from(cd{std::numeric_limits<double>::quiet_NaN(), 0.0});
  const auto capped = run<T>({polys[0]}, eps<R>(), 1);
  EXPECT_EQ(capped.iters[0], -1);
  const auto nan = run<T>({polys[1]}, eps<R>(), 50);
  EXPECT_EQ(nan.iters[0], -1);
}

} // namespace

TEST(AberthOracleTests, MatchesGeevFloat) {
  matches_companion_geev<float>();
}
TEST(AberthOracleTests, MatchesGeevDouble) {
  matches_companion_geev<double>();
}
TEST(AberthOracleTests, MatchesGeevComplexFloat) {
  matches_companion_geev<wwr::wwrFloatComplex>();
}
TEST(AberthOracleTests, MatchesGeevComplexDouble) {
  matches_companion_geev<wwr::wwrDoubleComplex>();
}

TEST(AberthOracleTests, RootsOfUnity) {
  roots_of_unity<float>();
  roots_of_unity<double>();
  roots_of_unity<wwr::wwrFloatComplex>();
  roots_of_unity<wwr::wwrDoubleComplex>();
}

TEST(AberthOracleTests, ZeroRoot) {
  zero_root<float>();
  zero_root<double>();
  zero_root<wwr::wwrDoubleComplex>();
}

TEST(AberthOracleTests, ReportsNonConvergence) {
  reports_non_convergence<double>();
  reports_non_convergence<wwr::wwrFloatComplex>();
}

// n == 0 or batch == 0 enqueues nothing, and null pointers are then allowed.
TEST(AberthOracleTests, EmptyIsNoop) {
  auto handle = shared_device();
  const auto stream = handle->stream().get();
  EXPECT_TRUE(calaman::aberth(stream, 0, 4, static_cast<const double *>(nullptr),
                              static_cast<wwr::wwrDoubleComplex *>(nullptr), nullptr, 1e-14, 10)
                  .ok());
  EXPECT_TRUE(calaman::aberth(stream, 5, 0, static_cast<const double *>(nullptr),
                              static_cast<wwr::wwrDoubleComplex *>(nullptr), nullptr, 1e-14, 10)
                  .ok());
}

TEST(AberthOracleTests, RejectsInvalidArguments) {
  auto handle = shared_device();
  const auto stream = handle->stream().get();
  DeviceBuffer<double> d_c(8, handle);
  DeviceBuffer<wwr::wwrDoubleComplex> d_r(8, handle);
  DeviceBuffer<int> d_i(1, handle);
  const auto call = [&](int n, int batch, const double *c, double tol, int max_iter) {
    return calaman::aberth(stream, n, batch, c, d_r.data(), d_i.data(), tol, max_iter);
  };
  EXPECT_EQ(call(-1, 1, d_c.data(), 1e-14, 10), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(3, -1, d_c.data(), 1e-14, 10), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(3, 1, d_c.data(), -1.0, 10), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(3, 1, d_c.data(), 1e-14, -1), wwr::wwrErrorInvalidValue);
  EXPECT_EQ(call(3, 1, nullptr, 1e-14, 10), wwr::wwrErrorInvalidValue);
  // 2n * 16 bytes over the 48 KiB shared budget.
  EXPECT_EQ(call(1537, 1, d_c.data(), 1e-14, 10), wwr::wwrErrorInvalidValue);
}

} // namespace calaman
