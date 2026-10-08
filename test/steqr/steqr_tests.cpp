// Oracle test for calaman.steqr -- eigenpairs of a symmetric tridiagonal by the
// implicit QL/QR method (?steqr). The oracle is LAPACKE_?steqr in the SAME
// precision on the host, over the identical real (d, e) and, for CompZ::V, the
// identical orthogonal (s/d) or unitary (c/z) Q in Z.
//
// Per case, for all three CompZ modes and all four element types: INFO
// matches; the eigenvalues match the oracle's to the shared factorization
// tolerance; and, with vectors, each column matches the oracle's up to a
// unit-modulus phase (a sign for s/d; where the eigenvalue gap makes that
// well-posed), Z is unitary (||Z^H Z - I||), and Z diagonalises the input
// (||A Z - Z diag(w)||, A = T for CompZ::I, A = Q T Q^H for CompZ::V).
// Residual and orthogonality are formed on the host in long double.
//
// Inputs: random, graded both ways (forcing QL and QR), clustered, pre-split
// (including isolated 2x2 blocks), Wilkinson, and over/underflow-scaled
// (exercising ?lascl); n = 0, 1, 2 and orders beyond the kernel's 256-thread
// block. REQUIRES_GPU; built only when calaman::lapack_reference exists.

#include <gtest/gtest.h>

#include <lapacke.h> // lapack_complex_float, lapack_complex_double

#include <cstddef>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.steqr;
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
using test::kTolFactor;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

using ld = long double;
using cld = std::complex<ld>;

// ========================================================================
// Per-element-type traits: the real component type, the builder, the widening
// to complex long double (host wwrC* -- .x/.y is not portable to hipComplex),
// and the oracle (the wwr complex types are layout-compatible with LAPACKE's)
// ========================================================================

template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool kComplex = false;
  static float make(ld re, ld) { return static_cast<float>(re); }
  static cld wide(float v) { return {v, 0.0L}; }
  static int ref(char compz, int n, float *d, float *e, float *z, int ldz) {
    return LAPACKE_ssteqr(LAPACK_COL_MAJOR, compz, n, d, e, z, ldz);
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool kComplex = false;
  static double make(ld re, ld) { return static_cast<double>(re); }
  static cld wide(double v) { return {v, 0.0L}; }
  static int ref(char compz, int n, double *d, double *e, double *z, int ldz) {
    return LAPACKE_dsteqr(LAPACK_COL_MAJOR, compz, n, d, e, z, ldz);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  static constexpr bool kComplex = true;
  static wwr::wwrFloatComplex make(ld re, ld im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static cld wide(wwr::wwrFloatComplex v) { return {wwr::wwrCrealf(v), wwr::wwrCimagf(v)}; }
  static int ref(char compz, int n, float *d, float *e, wwr::wwrFloatComplex *z, int ldz) {
    return LAPACKE_csteqr(LAPACK_COL_MAJOR, compz, n, d, e,
                          reinterpret_cast<lapack_complex_float *>(z), ldz);
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  static constexpr bool kComplex = true;
  static wwr::wwrDoubleComplex make(ld re, ld im) {
    return wwr::make_wwrDoubleComplex(static_cast<double>(re), static_cast<double>(im));
  }
  static cld wide(wwr::wwrDoubleComplex v) { return {wwr::wwrCreal(v), wwr::wwrCimag(v)}; }
  static int ref(char compz, int n, double *d, double *e, wwr::wwrDoubleComplex *z, int ldz) {
    return LAPACKE_zsteqr(LAPACK_COL_MAJOR, compz, n, d, e,
                          reinterpret_cast<lapack_complex_double *>(z), ldz);
  }
};

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

char compz_char(CompZ c) {
  return c == CompZ::N ? 'N' : (c == CompZ::V ? 'V' : 'I');
}

enum class Kind { random, graded_down, graded_up, clustered, split, wilkinson, huge, tiny };

const char *kind_name(Kind k) {
  switch (k) {
  case Kind::random: return "random";
  case Kind::graded_down: return "graded_down";
  case Kind::graded_up: return "graded_up";
  case Kind::clustered: return "clustered";
  case Kind::split: return "split";
  case Kind::wilkinson: return "wilkinson";
  case Kind::huge: return "huge";
  case Kind::tiny: return "tiny";
  }
  return "?";
}

/// Real (d, e) of order n for the given kind; values from a seeded generator.
template<typename R>
void make_tridiagonal(Kind kind, int n, std::vector<R> &d, std::vector<R> &e) {
  std::mt19937 gen(static_cast<std::uint32_t>(977 * n + 31 * static_cast<int>(kind) + 5));
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  d.assign(static_cast<std::size_t>(n), R{0});
  e.assign(static_cast<std::size_t>(n > 1 ? n - 1 : 0), R{0});
  const double range = std::is_same_v<R, float> ? 1e-6 : 1e-12; // graded spread
  for (int i = 0; i < n; ++i) {
    double di = u(gen);
    double ei = i + 1 < n ? u(gen) : 0.0;
    const double t = n > 1 ? static_cast<double>(i) / (n - 1) : 0.0;
    switch (kind) {
    case Kind::graded_down: { // large at the top: the reference picks QR
      const double gi = std::pow(range, t);
      const double gn = std::pow(range, n > 1 ? static_cast<double>(i + 1) / (n - 1) : 0.0);
      di = (2.0 + di) * gi;
      ei *= 0.5 * std::sqrt(gi * gn);
      break;
    }
    case Kind::graded_up: { // large at the bottom: QL
      const double gi = std::pow(range, 1.0 - t);
      const double gn = std::pow(range, n > 1 ? 1.0 - static_cast<double>(i + 1) / (n - 1) : 0.0);
      di = (2.0 + di) * gi;
      ei *= 0.5 * std::sqrt(gi * gn);
      break;
    }
    case Kind::clustered:
      di = 1.0 + 1e-3 * di * (std::is_same_v<R, float> ? 1.0 : 1e-6);
      ei *= 1e-4;
      break;
    case Kind::split:
      // Zeros at both ends, around isolated 2x2 blocks, and one negligible.
      if (i == 0 || i == n - 2 || i % 5 == 2 || i % 5 == 4) {
        ei = 0.0;
      } else if (i % 7 == 3) {
        ei = 1e-30;
      }
      break;
    case Kind::wilkinson:
      di = std::abs(i - (n - 1) / 2.0);
      ei = 1.0;
      break;
    case Kind::huge: {
      const double s = std::is_same_v<R, float> ? 1e20 : 1e160;
      di *= s;
      ei *= s;
      break;
    }
    case Kind::tiny: {
      const double s = std::is_same_v<R, float> ? 1e-6 : 1e-130;
      di *= s;
      ei *= s;
      break;
    }
    case Kind::random:
      break;
    }
    d[static_cast<std::size_t>(i)] = static_cast<R>(di);
    if (i + 1 < n) {
      e[static_cast<std::size_t>(i)] = static_cast<R>(ei);
    }
  }
}

/// A random n-by-n orthogonal (real T) or unitary (complex T) Q -- a product of
/// n Householder reflectors, built in long double -- rounded to T, stored with
/// leading dimension ldq.
template<typename T>
std::vector<T> random_unitary(int n, int ldq) {
  std::mt19937 gen(static_cast<std::uint32_t>(4242 + n));
  std::normal_distribution<double> g(0.0, 1.0);
  const auto un = static_cast<std::size_t>(n);
  std::vector<cld> q(un * un, 0.0L);
  for (std::size_t i = 0; i < un; ++i) {
    q[i + i * un] = 1.0L;
  }
  for (int r = 0; r < n; ++r) {
    std::vector<cld> v(un);
    ld nrm = 0.0L;
    for (auto &x : v) {
      const ld re = g(gen);
      const ld im = elem<T>::kComplex ? static_cast<ld>(g(gen)) : 0.0L;
      x = {re, im};
      nrm += std::norm(x);
    }
    if (nrm == 0.0L) {
      continue;
    }
    // Q <- Q (I - 2 v v^H / v^H v)
    for (std::size_t i = 0; i < un; ++i) {
      cld dot = 0.0L;
      for (std::size_t k = 0; k < un; ++k) {
        dot += q[i + k * un] * v[k];
      }
      for (std::size_t k = 0; k < un; ++k) {
        q[i + k * un] -= 2.0L * dot * std::conj(v[k]) / nrm;
      }
    }
  }
  std::vector<T> out(static_cast<std::size_t>(ldq) * un, elem<T>::make(0.0L, 0.0L));
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      const cld x = q[i + j * un];
      out[i + j * static_cast<std::size_t>(ldq)] = elem<T>::make(x.real(), x.imag());
    }
  }
  return out;
}

/// One case: device steqr vs LAPACKE_?steqr, plus orthogonality and residual.
template<typename T>
void run_case(CompZ compz, Kind kind, int n) {
  using R = typename elem<T>::R;
  const std::string ctx = std::string(1, compz_char(compz)) + " " + kind_name(kind) +
                          " n=" + std::to_string(n);
  SCOPED_TRACE(ctx);
  const bool wantz = compz != CompZ::N;
  const int ldz = wantz ? std::max(1, n) + 2 : 1;
  const auto un = static_cast<std::size_t>(n);
  const auto uld = static_cast<std::size_t>(ldz);

  std::vector<R> d0, e0;
  make_tridiagonal(kind, n, d0, e0);
  std::vector<T> z0 = compz == CompZ::V
                          ? random_unitary<T>(n, ldz)
                          : std::vector<T>(uld * std::max<std::size_t>(un, 1),
                                           elem<T>::make(7.0L, elem<T>::kComplex ? -3.0L : 0.0L));

  // Oracle.
  std::vector<R> r_d = d0, r_e = e0;
  std::vector<T> r_z = z0;
  if (r_e.empty()) {
    r_e.push_back(R{0});
  }
  const int r_info = elem<T>::ref(compz_char(compz), n, r_d.data(), r_e.data(), r_z.data(), ldz);

  // Device.
  auto d_d = to_device(d0);
  auto d_e = to_device(e0);
  auto d_z = to_device(z0);
  auto d_info = to_device(std::vector<int>{-1});
  const std::size_t bytes = steqr_bufferSize<T>(compz, n);
  auto d_work = to_device(std::vector<unsigned char>(bytes, 0));
  const auto handle = shared_device();
  const auto status = steqr<T>(handle->stream().get(), compz, n, d_d.data(), d_e.data(),
                               d_z.data(), ldz, d_work.data(), bytes, d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess);
  const int g_info = to_host(d_info, 1)[0];
  ASSERT_EQ(g_info, r_info);
  ASSERT_EQ(g_info, 0);
  const auto g_d = to_host(d_d, un);
  const auto g_zt = to_host(d_z, z0.size());

  // ||T||_F, the scale every bound below is relative to.
  ld tnorm2 = 0.0L;
  for (const R v : d0) {
    tnorm2 += static_cast<ld>(v) * v;
  }
  for (const R v : e0) {
    tnorm2 += 2.0L * static_cast<ld>(v) * v;
  }
  const R tnorm = static_cast<R>(std::sqrt(tnorm2));
  const R tol = test::factorization_tol<R>(tnorm, un, un) + std::numeric_limits<R>::min();

  for (std::size_t i = 0; i < un; ++i) {
    EXPECT_NEAR(g_d[i], r_d[i], tol) << "w(" << i << ")";
    if (i > 0) {
      EXPECT_LE(g_d[i - 1], g_d[i]) << "ascending at " << i;
    }
  }
  if (!wantz) {
    return;
  }

  // Everything below in complex long double (imaginary parts 0 for s/d).
  std::vector<cld> g_z(g_zt.size()), o_z(r_z.size()), q0(z0.size());
  for (std::size_t i = 0; i < g_zt.size(); ++i) {
    g_z[i] = elem<T>::wide(g_zt[i]);
    o_z[i] = elem<T>::wide(r_z[i]);
    q0[i] = elem<T>::wide(z0[i]);
  }

  // The lda padding rows are never written.
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = un; i < uld; ++i) {
      ASSERT_EQ(g_z[i + j * uld], q0[i + j * uld]) << "padding Z(" << i << "," << j << ")";
    }
  }

  // Columns up to a unit-modulus phase (a sign when real), where the
  // eigenvalue gap makes them well-determined: g_j ~ (o_j^H g_j / |.|) o_j.
  const R unit_tol = test::factorization_tol<R>(R{1}, un, un);
  for (std::size_t j = 0; j < un; ++j) {
    R gap = std::numeric_limits<R>::max();
    if (j > 0) {
      gap = std::min(gap, r_d[j] - r_d[j - 1]);
    }
    if (j + 1 < un) {
      gap = std::min(gap, r_d[j + 1] - r_d[j]);
    }
    if (gap < R{1e-2} * tnorm) {
      continue;
    }
    cld dot = 0.0L;
    for (std::size_t i = 0; i < un; ++i) {
      dot += std::conj(o_z[i + j * uld]) * g_z[i + j * uld];
    }
    const cld phase = std::abs(dot) == 0.0L ? cld{1.0L} : dot / std::abs(dot);
    const R vtol = unit_tol * std::max(R{1}, tnorm / gap);
    for (std::size_t i = 0; i < un; ++i) {
      EXPECT_LE(static_cast<R>(std::abs(g_z[i + j * uld] - phase * o_z[i + j * uld])), vtol)
          << "Z(" << i << "," << j << ")";
    }
  }

  // ||Z^H Z - I||_F.
  ld orth2 = 0.0L;
  for (std::size_t a = 0; a < un; ++a) {
    for (std::size_t b = 0; b < un; ++b) {
      cld s = a == b ? -1.0L : 0.0L;
      for (std::size_t i = 0; i < un; ++i) {
        s += std::conj(g_z[i + a * uld]) * g_z[i + b * uld];
      }
      orth2 += std::norm(s);
    }
  }
  EXPECT_LE(static_cast<R>(std::sqrt(orth2)), test::factorization_tol<R>(R(n), un, un))
      << "orthogonality";

  // ||A Z - Z diag(w)||_F with A = T (CompZ::I) or Q T Q^H (CompZ::V), all in
  // long double from the T-precision inputs.
  std::vector<cld> A(un * un, 0.0L);
  for (std::size_t i = 0; i < un; ++i) {
    A[i + i * un] = d0[i];
    if (i + 1 < un) {
      A[i + 1 + i * un] = e0[i];
      A[i + (i + 1) * un] = e0[i];
    }
  }
  if (compz == CompZ::V) {
    std::vector<cld> QT(un * un, 0.0L); // Q * T
    for (std::size_t j = 0; j < un; ++j) {
      for (std::size_t k = 0; k < un; ++k) {
        const cld t = A[k + j * un];
        if (t == 0.0L) {
          continue;
        }
        for (std::size_t i = 0; i < un; ++i) {
          QT[i + j * un] += q0[i + k * uld] * t;
        }
      }
    }
    std::fill(A.begin(), A.end(), cld{0.0L}); // (Q T) Q^H
    for (std::size_t j = 0; j < un; ++j) {
      for (std::size_t k = 0; k < un; ++k) {
        const cld q = std::conj(q0[j + k * uld]);
        for (std::size_t i = 0; i < un; ++i) {
          A[i + j * un] += QT[i + k * un] * q;
        }
      }
    }
  }
  ld res2 = 0.0L;
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      cld s = -g_z[i + j * uld] * static_cast<ld>(g_d[j]);
      for (std::size_t k = 0; k < un; ++k) {
        s += A[i + k * un] * g_z[k + j * uld];
      }
      res2 += std::norm(s);
    }
  }
  const R rtol = kTolFactor<R> * eps<R>() * tnorm * static_cast<R>(n) *
                     std::sqrt(static_cast<R>(std::max(n, 1))) +
                 std::numeric_limits<R>::min();
  EXPECT_LE(static_cast<R>(std::sqrt(res2)), rtol) << "residual";
}

constexpr std::array kModes = {CompZ::N, CompZ::V, CompZ::I};
constexpr std::array kKinds = {Kind::random,    Kind::graded_down, Kind::graded_up,
                               Kind::clustered, Kind::split,       Kind::wilkinson,
                               Kind::huge,      Kind::tiny};
constexpr std::array kSizes = {0, 1, 2, 3, 4, 7, 16, 65, 300};

template<typename T>
void run_all() {
  for (const CompZ compz : kModes) {
    for (const Kind kind : kKinds) {
      for (const int n : kSizes) {
        run_case<T>(compz, kind, n);
      }
    }
    run_case<T>(compz, Kind::random, 513); // two strides of the 256-thread block
  }
}

TEST(SteqrOracleTests, Float) {
  run_all<float>();
}

TEST(SteqrOracleTests, Double) {
  run_all<double>();
}

TEST(SteqrOracleTests, ComplexFloat) {
  run_all<wwr::wwrFloatComplex>();
}

TEST(SteqrOracleTests, ComplexDouble) {
  run_all<wwr::wwrDoubleComplex>();
}

TEST(SteqrOracleTests, BufferSize) {
  EXPECT_EQ(steqr_bufferSize<double>(CompZ::N, 100), 0u);
  EXPECT_EQ(steqr_bufferSize<double>(CompZ::I, 1), 0u);
  EXPECT_GE(steqr_bufferSize<double>(CompZ::V, 100), 2u * 99u * sizeof(double));
  EXPECT_GE(steqr_bufferSize<float>(CompZ::I, 100), 2u * 99u * sizeof(float));
  // The rotations stay real for a complex Z: the same bytes as the real type.
  EXPECT_EQ(steqr_bufferSize<wwr::wwrDoubleComplex>(CompZ::V, 100),
            steqr_bufferSize<double>(CompZ::V, 100));
  EXPECT_EQ(steqr_bufferSize<wwr::wwrFloatComplex>(CompZ::I, 100),
            steqr_bufferSize<float>(CompZ::I, 100));
  EXPECT_EQ(steqr_bufferSize<wwr::wwrFloatComplex>(CompZ::N, 100), 0u);
}

TEST(SteqrOracleTests, RejectsBadArguments) {
  const auto handle = shared_device();
  const auto stream = handle->stream().get();
  auto d_d = to_device(std::vector<double>(4, 1.0));
  auto d_e = to_device(std::vector<double>(3, 1.0));
  auto d_z = to_device(std::vector<double>(16, 0.0));
  auto d_info = to_device(std::vector<int>{0});
  const std::size_t bytes = steqr_bufferSize<double>(CompZ::I, 4);
  auto d_work = to_device(std::vector<unsigned char>(bytes, 0));
  EXPECT_NE(steqr<double>(stream, CompZ::I, -1, d_d.data(), d_e.data(), d_z.data(), 4,
                          d_work.data(), bytes, d_info.data()),
            wwr::wwrSuccess);
  EXPECT_NE(steqr<double>(stream, CompZ::I, 4, d_d.data(), d_e.data(), d_z.data(), 3,
                          d_work.data(), bytes, d_info.data()),
            wwr::wwrSuccess);
  EXPECT_NE(steqr<double>(stream, CompZ::I, 4, d_d.data(), d_e.data(), d_z.data(), 4,
                          d_work.data(), bytes - 1, d_info.data()),
            wwr::wwrSuccess);
  // CompZ::N needs no workspace and no Z.
  EXPECT_EQ(steqr<double>(stream, CompZ::N, 4, d_d.data(), d_e.data(), nullptr, 1, nullptr, 0,
                          d_info.data()),
            wwr::wwrSuccess);
  EXPECT_EQ(steqr<wwr::wwrDoubleComplex>(stream, CompZ::N, 4, d_d.data(), d_e.data(), nullptr, 1,
                                         nullptr, 0, d_info.data()),
            wwr::wwrSuccess);
  wwr::wwrStreamSynchronize(stream);
}

} // namespace
} // namespace calaman
