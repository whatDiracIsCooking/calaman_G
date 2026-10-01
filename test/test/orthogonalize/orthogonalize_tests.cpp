// Oracle test for calaman.orthogonalize -- the QR-based in-place orthonormalizer
// (geqrf, then orgqr/ungqr to expand the thin Q). Unlike the linalg suites, the
// oracle is NOT the reference LAPACK: a QR factorization is unique only up to a
// per-column sign (real) or unit-modulus phase (complex), so LAPACK's Q and the
// device's Q may legitimately differ element-by-element. The robust, sign-free
// invariants are the two defining properties of a correct orthonormal basis for
// range(A), and both are plain matrix products -- BLAS-level, not LAPACK:
//
//   1. Orthonormality: Q^H Q = I_n.
//   2. Span preservation: Q(Q^H A) = A. Because Q's columns are an orthonormal
//      basis for range(A), Q Q^H is the orthogonal projector onto range(A), and
//      every column of A already lies in range(A) -- so reprojecting A through Q
//      returns A. This pins that Q spans the SAME space as A, not merely some
//      orthonormal set.
//
// The products are evaluated on the host in std::complex after Q is copied back,
// so the suite needs neither device BLAS nor the CPU reference LAPACK -- it has
// no calaman::lapack_reference guard and runs wherever a card exists. All four
// instantiated types are exercised, so BOTH the orgqr (real) and ungqr (complex)
// branches of the if constexpr are covered.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/work on the device,
// creates a solver handle and runs geqrf + orgqr/ungqr, so `ctest -LE gpu`
// excludes it.

#include <gtest/gtest.h>

import std;

import wwr.solver;
import wwr.complex;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.orthogonalize;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceHandle, DeviceAbort>;

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

// --- Element-type plumbing: map each device element type T to a host scalar and
// convert at the buffer boundary, so the verification math is one std::complex
// code path for all four types. For a real T the imaginary part is simply zero.

template<typename T>
inline constexpr bool is_complex_v =
    std::is_same_v<T, wwr::wwrFloatComplex> || std::is_same_v<T, wwr::wwrDoubleComplex>;

// The real scalar underlying T: float for float/wwrFloatComplex, double otherwise.
template<typename T>
using real_t =
    std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, wwr::wwrFloatComplex>, float,
                       double>;

template<typename T>
T from_std(std::complex<real_t<T>> z) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return z.real();
  } else if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return wwr::make_gpuFloatComplex(z.real(), z.imag());
  } else {
    return wwr::make_gpuDoubleComplex(z.real(), z.imag());
  }
}

template<typename T>
std::complex<real_t<T>> to_std(T v) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return {v, real_t<T>{0}};
  } else if constexpr (std::is_same_v<T, wwr::wwrFloatComplex>) {
    return {wwr::wwrCrealf(v), wwr::wwrCimagf(v)};
  } else {
    return {wwr::wwrCreal(v), wwr::wwrCimag(v)};
  }
}

// Frobenius norm of a host complex matrix: sqrt(sum |z|^2). std::norm(z) is |z|^2.
template<typename Real>
Real frobenius(const std::vector<std::complex<Real>> &v) {
  Real sum{};
  for (const auto z : v) {
    sum += std::norm(z);
  }
  return std::sqrt(sum);
}

/// @brief orthogonalize's thin Q must have orthonormal columns that span range(A)
///
/// Runs orthogonalize on a random m-by-n (m >= n) input, copies Q back, and
/// checks the two sign-free invariants on the host: ||Q^H Q - I|| and the
/// reprojection residual ||Q(Q^H A) - A||. The orthonormality bound drops the
/// ||A|| factor (Q's columns are unit-norm regardless of the input's scale); the
/// reprojection residual keeps it, since that error does scale with ||A||.
template<typename T>
void expect_orthonormal(int m, int n, unsigned seed) {
  using Real = real_t<T>;
  using Cplx = std::complex<Real>;
  ASSERT_GE(m, n);
  const int lda = m;
  const std::size_t mn = static_cast<std::size_t>(lda) * n;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);

  // Random A in host complex form (imag part 0 for a real T), column-major.
  std::vector<Cplx> a(mn);
  for (auto &z : a) {
    const Real re = static_cast<Real>(dist(rng));
    const Real im = is_complex_v<T> ? static_cast<Real>(dist(rng)) : Real{0};
    z = Cplx(re, im);
  }

  std::vector<T> a_dev(mn);
  for (std::size_t i = 0; i < mn; ++i) {
    a_dev[i] = from_std<T>(a[i]);
  }

  auto handle = std::make_shared<DeviceHandle>(0);
  wwr::wwrsolverDnHandle_t solver{};
  ASSERT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrsolverDnSetStream(solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);

  auto d_q = to_device(handle, a_dev);

  int lwork = 0;
  ASSERT_EQ(orthogonalize_bufferSize<T>(solver, m, n, &lwork), wwr::WWRSOLVER_STATUS_SUCCESS);
  ASSERT_GT(lwork, 0);

  std::vector<T> work_init(static_cast<std::size_t>(lwork)); // value-initialised scratch
  auto d_work = to_device(handle, work_init);
  std::vector<int> info_init(1, 0);
  auto d_info_geqrf = to_device(handle, info_init);
  auto d_info_gqr = to_device(handle, info_init);

  orthogonalize<T>(solver, m, n, d_q.data(), d_work.data(), lwork, d_info_geqrf.data(),
                   d_info_gqr.data());

  const auto got = from_device(handle, d_q, mn);
  const auto info_geqrf = from_device(handle, d_info_geqrf, 1);
  const auto info_gqr = from_device(handle, d_info_gqr, 1);
  wwr::wwrsolverDnDestroy(solver);

  EXPECT_EQ(info_geqrf[0], 0) << "geqrf info m=" << m << " n=" << n << " seed=" << seed;
  EXPECT_EQ(info_gqr[0], 0) << "orgqr/ungqr info m=" << m << " n=" << n << " seed=" << seed;

  std::vector<Cplx> q(mn);
  for (std::size_t i = 0; i < mn; ++i) {
    q[i] = to_std<T>(got[i]);
  }

  const Real norm_a = frobenius(a);
  const Real ortho_tol =
      factorization_tol<Real>(Real{1}, static_cast<std::size_t>(m), static_cast<std::size_t>(n));
  const Real recon_tol =
      factorization_tol<Real>(norm_a, static_cast<std::size_t>(m), static_cast<std::size_t>(n));

  // 1. Orthonormality: G = Q^H Q should be I_n. Accumulate ||G - I||_F.
  std::vector<Cplx> gram_minus_i(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      Cplx g{};
      for (int i = 0; i < m; ++i) {
        g += std::conj(q[static_cast<std::size_t>(j) * lda + i]) *
             q[static_cast<std::size_t>(k) * lda + i];
      }
      gram_minus_i[static_cast<std::size_t>(k) * n + j] = g - (j == k ? Cplx{1} : Cplx{0});
    }
  }
  EXPECT_LE(frobenius(gram_minus_i), ortho_tol)
      << "||Q^H Q - I|| m=" << m << " n=" << n << " seed=" << seed;

  // 2. Span preservation: Q(Q^H A) should equal A. First M = Q^H A (n x n),
  // then reproject P = Q M (m x n) and accumulate ||P - A||_F.
  std::vector<Cplx> coeff(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int c = 0; c < n; ++c) {
      Cplx s{};
      for (int i = 0; i < m; ++i) {
        s += std::conj(q[static_cast<std::size_t>(j) * lda + i]) *
             a[static_cast<std::size_t>(c) * lda + i];
      }
      coeff[static_cast<std::size_t>(c) * n + j] = s;
    }
  }
  std::vector<Cplx> residual(mn);
  for (int c = 0; c < n; ++c) {
    for (int i = 0; i < m; ++i) {
      Cplx p{};
      for (int j = 0; j < n; ++j) {
        p += q[static_cast<std::size_t>(j) * lda + i] * coeff[static_cast<std::size_t>(c) * n + j];
      }
      const std::size_t idx = static_cast<std::size_t>(c) * lda + i;
      residual[idx] = p - a[idx];
    }
  }
  EXPECT_LE(frobenius(residual), recon_tol)
      << "||Q(Q^H A) - A|| m=" << m << " n=" << n << " seed=" << seed;
}

// Tall (m > n), the common case, across all four types.
TEST(OrthogonalizeOracleTests, TallFloat) {
  expect_orthonormal<float>(8, 4, 1);
  expect_orthonormal<float>(12, 5, 2);
}
TEST(OrthogonalizeOracleTests, TallDouble) {
  expect_orthonormal<double>(8, 4, 3);
  expect_orthonormal<double>(12, 5, 4);
}
TEST(OrthogonalizeOracleTests, TallComplexFloat) {
  expect_orthonormal<wwr::wwrFloatComplex>(8, 4, 5);
  expect_orthonormal<wwr::wwrFloatComplex>(12, 5, 6);
}
TEST(OrthogonalizeOracleTests, TallComplexDouble) {
  expect_orthonormal<wwr::wwrDoubleComplex>(8, 4, 7);
  expect_orthonormal<wwr::wwrDoubleComplex>(12, 5, 8);
}

// Square (m == n): the thin Q is a full orthonormal basis of all of C^m.
TEST(OrthogonalizeOracleTests, SquareFloat) {
  expect_orthonormal<float>(6, 6, 9);
}
TEST(OrthogonalizeOracleTests, SquareDouble) {
  expect_orthonormal<double>(6, 6, 10);
  expect_orthonormal<double>(16, 16, 11);
}
TEST(OrthogonalizeOracleTests, SquareComplexFloat) {
  expect_orthonormal<wwr::wwrFloatComplex>(6, 6, 12);
}
TEST(OrthogonalizeOracleTests, SquareComplexDouble) {
  expect_orthonormal<wwr::wwrDoubleComplex>(6, 6, 13);
  expect_orthonormal<wwr::wwrDoubleComplex>(16, 16, 14);
}

// Single column (n == 1): orthonormalization reduces to normalisation, and the
// tau_size = align_up(1, 256) padding dominates the workspace layout.
TEST(OrthogonalizeOracleTests, SingleColumn) {
  expect_orthonormal<double>(10, 1, 15);
  expect_orthonormal<wwr::wwrDoubleComplex>(10, 1, 16);
}

// Larger matrices, past a single warp of columns, real and complex.
TEST(OrthogonalizeOracleTests, Larger) {
  expect_orthonormal<double>(64, 32, 17);
  expect_orthonormal<wwr::wwrDoubleComplex>(48, 24, 18);
}

} // namespace
} // namespace calaman
