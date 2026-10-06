// Oracle test for calaman.steqr -- eigenpairs of a symmetric tridiagonal by the
// implicit QL/QR method (?steqr). The oracle is LAPACKE_?steqr in the SAME
// precision on the host, over the identical (d, e) and, for CompZ::V, the
// identical orthogonal Q in Z.
//
// Per case, for all three CompZ modes: INFO matches; the eigenvalues match the
// oracle's to the shared factorization tolerance; and, with vectors, each
// column matches the oracle's up to sign (where the eigenvalue gap makes that
// well-posed), Z is orthogonal (||Z^T Z - I||), and Z diagonalises the input
// (||A Z - Z diag(w)||, A = T for CompZ::I, A = Q T Q^T for CompZ::V).
// Residual and orthogonality are formed on the host in long double.
//
// Inputs: random, graded both ways (forcing QL and QR), clustered, pre-split
// (including isolated 2x2 blocks), Wilkinson, and over/underflow-scaled
// (exercising ?lascl); n = 0, 1, 2 and orders beyond the kernel's 256-thread
// block. REQUIRES_GPU; built only when calaman::lapack_reference exists.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>

import std;

import wwr.runtime_api;
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

int ref_steqr(char compz, int n, float *d, float *e, float *z, int ldz) {
  return LAPACKE_ssteqr(LAPACK_COL_MAJOR, compz, n, d, e, z, ldz);
}
int ref_steqr(char compz, int n, double *d, double *e, double *z, int ldz) {
  return LAPACKE_dsteqr(LAPACK_COL_MAJOR, compz, n, d, e, z, ldz);
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

/// (d, e) of order n for the given kind; values from a seeded generator.
template<typename T>
void make_tridiagonal(Kind kind, int n, std::vector<T> &d, std::vector<T> &e) {
  std::mt19937 gen(static_cast<std::uint32_t>(977 * n + 31 * static_cast<int>(kind) + 5));
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  d.assign(static_cast<std::size_t>(n), T{0});
  e.assign(static_cast<std::size_t>(n > 1 ? n - 1 : 0), T{0});
  const double range = std::is_same_v<T, float> ? 1e-6 : 1e-12; // graded spread
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
      di = 1.0 + 1e-3 * di * (std::is_same_v<T, float> ? 1.0 : 1e-6);
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
      const double s = std::is_same_v<T, float> ? 1e20 : 1e160;
      di *= s;
      ei *= s;
      break;
    }
    case Kind::tiny: {
      const double s = std::is_same_v<T, float> ? 1e-6 : 1e-130;
      di *= s;
      ei *= s;
      break;
    }
    case Kind::random:
      break;
    }
    d[static_cast<std::size_t>(i)] = static_cast<T>(di);
    if (i + 1 < n) {
      e[static_cast<std::size_t>(i)] = static_cast<T>(ei);
    }
  }
}

/// A random n-by-n orthogonal Q (product of n Householder reflectors, built in
/// long double), rounded to T, stored with leading dimension ldq.
template<typename T>
std::vector<T> random_orthogonal(int n, int ldq) {
  std::mt19937 gen(static_cast<std::uint32_t>(4242 + n));
  std::normal_distribution<double> g(0.0, 1.0);
  const auto un = static_cast<std::size_t>(n);
  std::vector<ld> q(un * un, 0.0L);
  for (std::size_t i = 0; i < un; ++i) {
    q[i + i * un] = 1.0L;
  }
  for (int r = 0; r < n; ++r) {
    std::vector<ld> v(un);
    ld nrm = 0.0L;
    for (auto &x : v) {
      x = g(gen);
      nrm += x * x;
    }
    if (nrm == 0.0L) {
      continue;
    }
    // Q <- Q (I - 2 v v^T / v^T v)
    for (std::size_t i = 0; i < un; ++i) {
      ld dot = 0.0L;
      for (std::size_t k = 0; k < un; ++k) {
        dot += q[i + k * un] * v[k];
      }
      for (std::size_t k = 0; k < un; ++k) {
        q[i + k * un] -= 2.0L * dot * v[k] / nrm;
      }
    }
  }
  std::vector<T> out(static_cast<std::size_t>(ldq) * un, T{0});
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      out[i + j * static_cast<std::size_t>(ldq)] = static_cast<T>(q[i + j * un]);
    }
  }
  return out;
}

/// One case: device steqr vs LAPACKE_?steqr, plus orthogonality and residual.
template<typename T>
void run_case(CompZ compz, Kind kind, int n) {
  const std::string ctx = std::string(1, compz_char(compz)) + " " + kind_name(kind) +
                          " n=" + std::to_string(n);
  SCOPED_TRACE(ctx);
  const bool wantz = compz != CompZ::N;
  const int ldz = wantz ? std::max(1, n) + 2 : 1;
  const auto un = static_cast<std::size_t>(n);
  const auto uld = static_cast<std::size_t>(ldz);

  std::vector<T> d0, e0;
  make_tridiagonal(kind, n, d0, e0);
  std::vector<T> z0 = compz == CompZ::V ? random_orthogonal<T>(n, ldz)
                                        : std::vector<T>(uld * std::max<std::size_t>(un, 1), T{7});

  // Oracle.
  std::vector<T> r_d = d0, r_e = e0, r_z = z0;
  if (r_e.empty()) {
    r_e.push_back(T{0});
  }
  const int r_info = ref_steqr(compz_char(compz), n, r_d.data(), r_e.data(), r_z.data(), ldz);

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
  const auto g_z = to_host(d_z, z0.size());

  // ||T||_F, the scale every bound below is relative to.
  ld tnorm2 = 0.0L;
  for (const T v : d0) {
    tnorm2 += static_cast<ld>(v) * v;
  }
  for (const T v : e0) {
    tnorm2 += 2.0L * static_cast<ld>(v) * v;
  }
  const T tnorm = static_cast<T>(std::sqrt(tnorm2));
  const T tol = test::factorization_tol<T>(tnorm, un, un) + std::numeric_limits<T>::min();

  for (std::size_t i = 0; i < un; ++i) {
    EXPECT_NEAR(g_d[i], r_d[i], tol) << "w(" << i << ")";
    if (i > 0) {
      EXPECT_LE(g_d[i - 1], g_d[i]) << "ascending at " << i;
    }
  }
  if (!wantz) {
    return;
  }

  // The lda padding rows are never written.
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = un; i < uld; ++i) {
      ASSERT_EQ(g_z[i + j * uld], z0[i + j * uld]) << "padding Z(" << i << "," << j << ")";
    }
  }

  // Columns up to sign, where the eigenvalue gap makes them well-determined.
  const T unit_tol = test::factorization_tol<T>(T{1}, un, un);
  for (std::size_t j = 0; j < un; ++j) {
    T gap = std::numeric_limits<T>::max();
    if (j > 0) {
      gap = std::min(gap, r_d[j] - r_d[j - 1]);
    }
    if (j + 1 < un) {
      gap = std::min(gap, r_d[j + 1] - r_d[j]);
    }
    if (gap < T{1e-2} * tnorm) {
      continue;
    }
    ld dot = 0.0L;
    for (std::size_t i = 0; i < un; ++i) {
      dot += static_cast<ld>(g_z[i + j * uld]) * r_z[i + j * uld];
    }
    const T sign = dot < 0.0L ? T{-1} : T{1};
    const T vtol = unit_tol * std::max(T{1}, tnorm / gap);
    for (std::size_t i = 0; i < un; ++i) {
      EXPECT_NEAR(g_z[i + j * uld], sign * r_z[i + j * uld], vtol)
          << "Z(" << i << "," << j << ")";
    }
  }

  // ||Z^T Z - I||_F.
  ld orth2 = 0.0L;
  for (std::size_t a = 0; a < un; ++a) {
    for (std::size_t b = 0; b < un; ++b) {
      ld s = a == b ? -1.0L : 0.0L;
      for (std::size_t i = 0; i < un; ++i) {
        s += static_cast<ld>(g_z[i + a * uld]) * g_z[i + b * uld];
      }
      orth2 += s * s;
    }
  }
  EXPECT_LE(static_cast<T>(std::sqrt(orth2)), test::factorization_tol<T>(T(n), un, un))
      << "orthogonality";

  // ||A Z - Z diag(w)||_F with A = T (CompZ::I) or Q T Q^T (CompZ::V), all in
  // long double from the T-precision inputs.
  std::vector<ld> A(un * un, 0.0L);
  for (std::size_t i = 0; i < un; ++i) {
    A[i + i * un] = d0[i];
    if (i + 1 < un) {
      A[i + 1 + i * un] = e0[i];
      A[i + (i + 1) * un] = e0[i];
    }
  }
  if (compz == CompZ::V) {
    std::vector<ld> QT(un * un, 0.0L); // Q * T
    for (std::size_t j = 0; j < un; ++j) {
      for (std::size_t k = 0; k < un; ++k) {
        const ld t = A[k + j * un];
        if (t == 0.0L) {
          continue;
        }
        for (std::size_t i = 0; i < un; ++i) {
          QT[i + j * un] += static_cast<ld>(z0[i + k * uld]) * t;
        }
      }
    }
    std::fill(A.begin(), A.end(), 0.0L); // (Q T) Q^T
    for (std::size_t j = 0; j < un; ++j) {
      for (std::size_t k = 0; k < un; ++k) {
        const ld q = z0[j + k * uld];
        for (std::size_t i = 0; i < un; ++i) {
          A[i + j * un] += QT[i + k * un] * q;
        }
      }
    }
  }
  ld res2 = 0.0L;
  for (std::size_t j = 0; j < un; ++j) {
    for (std::size_t i = 0; i < un; ++i) {
      ld s = -static_cast<ld>(g_z[i + j * uld]) * g_d[j];
      for (std::size_t k = 0; k < un; ++k) {
        s += A[i + k * un] * g_z[k + j * uld];
      }
      res2 += s * s;
    }
  }
  const T rtol = kTolFactor<T> * eps<T>() * tnorm * static_cast<T>(n) *
                     std::sqrt(static_cast<T>(std::max(n, 1))) +
                 std::numeric_limits<T>::min();
  EXPECT_LE(static_cast<T>(std::sqrt(res2)), rtol) << "residual";
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

TEST(SteqrOracleTests, BufferSize) {
  EXPECT_EQ(steqr_bufferSize<double>(CompZ::N, 100), 0u);
  EXPECT_EQ(steqr_bufferSize<double>(CompZ::I, 1), 0u);
  EXPECT_GE(steqr_bufferSize<double>(CompZ::V, 100), 2u * 99u * sizeof(double));
  EXPECT_GE(steqr_bufferSize<float>(CompZ::I, 100), 2u * 99u * sizeof(float));
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
  wwr::wwrStreamSynchronize(stream);
}

} // namespace
} // namespace calaman
