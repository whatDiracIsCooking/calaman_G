// Oracle test for calaman.inv_sqrt -- the symmetric (Loewdin) inverse square
// root X = S^{-1/2} of a real symmetric positive-(semi)definite matrix. Like
// the orthogonalize suite, the oracle is NOT the reference LAPACK: the symmetric
// PD root is unique and the per-eigenvector sign cancels in U diag U^T, so the
// robust checks are the two defining, sign-free invariants, both plain matrix
// products evaluated on the host after X is copied back:
//
//   1. Reconstruction: X^T S X = I_n. For the unique symmetric PD root this pins
//      X to S^{-1/2} (with X symmetric). The residual is measured in double.
//   2. Symmetry: X = X^T, which the U diag(Lambda^{-1/2}) U^T assembly produces
//      by construction.
//
// A separate case feeds an INDEFINITE symmetric S (negative eigenvalues) and
// asserts X is finite -- the floor in the inverse_sqrt kernel drops the
// non-positive modes to 0 before the square root, so the result never goes
// NaN/Inf even though X^T S X is then only a rank-deficient projector.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages S/scratch on the device,
// creates the BLAS + solver handles and runs syevd + the kernel + dgmm + gemm,
// so `ctest -LE gpu` excludes it.

#include <gtest/gtest.h>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.inv_sqrt;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;
import calaman.test.shared.tolerance;

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

// Create the BLAS + solver handles inv_sqrt needs, both bound to the shared
// stream. Destroyed by the caller.
struct Handles {
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
};

Handles make_handles(const std::shared_ptr<DeviceHandle> &handle) {
  Handles h{};
  EXPECT_EQ(wwr::wwrblasCreate(&h.blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(h.blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnCreate(&h.solver), wwr::WWRSOLVER_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrsolverDnSetStream(h.solver, handle->stream().get()),
            wwr::WWRSOLVER_STATUS_SUCCESS);
  return h;
}

void destroy_handles(Handles &h) {
  wwr::wwrblasDestroy(h.blas);
  wwr::wwrsolverDnDestroy(h.solver);
}

// Run inv_sqrt on a device copy of the n-by-n column-major symmetric S and
// return X = S^{-1/2}. The caller supplies S already staged in host form.
template<typename T>
std::vector<T> run_inv_sqrt(const std::vector<T> &s, int n) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);

  auto handle = shared_device();
  auto h = make_handles(handle);

  auto d_s = to_device(handle, s);
  const std::vector<T> zeros_nn(nn, T{0});
  const std::vector<T> zeros_n(static_cast<std::size_t>(n), T{0});
  auto d_x = to_device(handle, zeros_nn);
  auto d_u = to_device(handle, zeros_nn);
  auto d_m = to_device(handle, zeros_nn);
  auto d_eig = to_device(handle, zeros_n);

  int lwork = 0;
  EXPECT_TRUE(inv_sqrt_bufferSize<T>(h.solver, n, &lwork).ok());
  EXPECT_GE(lwork, 0);
  std::vector<T> work_init(static_cast<std::size_t>(std::max(lwork, 1)), T{0});
  auto d_work = to_device(handle, work_init);

  const std::vector<int> info_init(1, 0);
  auto d_info = to_device(handle, info_init);

  const auto status =
      inv_sqrt<T>(h.blas, h.solver, handle->stream().get(), n, d_s.data(), d_x.data(), d_u.data(),
                  d_m.data(), d_eig.data(), d_work.data(), lwork, d_info.data());
  EXPECT_TRUE(status.ok()) << "inv_sqrt status n=" << n << " status=" << status.name();

  const auto x = from_device(handle, d_x, nn);
  const auto info = from_device(handle, d_info, 1);
  const auto s_back = from_device(handle, d_s, nn);
  destroy_handles(h);

  EXPECT_EQ(info[0], 0) << "syevd info n=" << n;
  // The operand must be left untouched.
  for (std::size_t i = 0; i < nn; ++i) {
    EXPECT_EQ(s_back[i], s[i]) << "s_dev modified at " << i << " n=" << n;
  }
  return x;
}

// A random symmetric positive-definite S = (B+B^T) spectrum shifted onto the
// positive axis: S = B B^T + n I, with entries O(1), so cond(S) stays modest
// and S^{-1/2} is the exact symmetric root (no mode is floored).
template<typename T>
std::vector<T> make_spd(int n, unsigned seed) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> b(nn);
  for (auto &v : b) {
    v = dist(rng);
  }
  // S = B B^T + n I (symmetric PD), column-major (i + j*n).
  std::vector<T> s(nn, T{0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      double acc = 0.0;
      for (int k = 0; k < n; ++k) {
        acc += b[static_cast<std::size_t>(k) * n + i] * b[static_cast<std::size_t>(k) * n + j];
      }
      if (i == j) {
        acc += static_cast<double>(n);
      }
      s[static_cast<std::size_t>(j) * n + i] = static_cast<T>(acc);
    }
  }
  return s;
}

/// @brief X = S^{-1/2} must satisfy X^T S X = I and be symmetric
template<typename T>
void expect_inverse_sqrt(int n, unsigned seed) {
  const auto s = make_spd<T>(n, seed);
  const auto x = run_inv_sqrt(s, n);

  const auto idx = [n](int i, int j) { return static_cast<std::size_t>(j) * n + i; };

  // Promote to double and form Z = X^T S X. X is symmetric to roundoff, so
  // X^T == X; use the returned entries directly.
  std::vector<double> sd(s.size()), xd(x.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    sd[i] = static_cast<double>(s[i]);
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    xd[i] = static_cast<double>(x[i]);
  }

  // Y = S X  (n x n)
  std::vector<double> y(static_cast<std::size_t>(n) * n, 0.0);
  for (int c = 0; c < n; ++c) {
    for (int i = 0; i < n; ++i) {
      double acc = 0.0;
      for (int k = 0; k < n; ++k) {
        acc += sd[idx(i, k)] * xd[idx(k, c)];
      }
      y[idx(i, c)] = acc;
    }
  }
  // Z = X^T Y = X S X; accumulate ||Z - I||_F and ||X - X^T||_F.
  double recon = 0.0;
  double sym = 0.0;
  for (int c = 0; c < n; ++c) {
    for (int r = 0; r < n; ++r) {
      double acc = 0.0;
      for (int i = 0; i < n; ++i) {
        acc += xd[idx(i, r)] * y[idx(i, c)]; // X^T[r,i] = X[i,r]
      }
      const double target = (r == c) ? 1.0 : 0.0;
      recon += (acc - target) * (acc - target);
      const double d = xd[idx(r, c)] - xd[idx(c, r)];
      sym += d * d;
    }
  }
  recon = std::sqrt(recon);
  sym = std::sqrt(sym);

  // The reconstruction error scales with cond(S); S here is well conditioned, so
  // ||S||_F is a safe, generous proxy for the span of the bound. The device-type
  // epsilon drives it, measured in double.
  double norm_s = 0.0;
  for (const T v : s) {
    norm_s += static_cast<double>(v) * static_cast<double>(v);
  }
  norm_s = std::sqrt(norm_s);
  const double unit = static_cast<double>(eps<T>());
  const double recon_tol = static_cast<double>(kTolFactor<T>) * unit * norm_s * n;
  const double sym_tol = static_cast<double>(kTolFactor<T>) * unit * n;

  EXPECT_LE(recon, recon_tol) << "||X^T S X - I|| n=" << n << " seed=" << seed;
  EXPECT_LE(sym, sym_tol) << "||X - X^T|| n=" << n << " seed=" << seed;
}

TEST(InvSqrtOracleTests, SpdFloat) {
  expect_inverse_sqrt<float>(4, 1);
  expect_inverse_sqrt<float>(16, 2);
}
TEST(InvSqrtOracleTests, SpdDouble) {
  expect_inverse_sqrt<double>(4, 3);
  expect_inverse_sqrt<double>(16, 4);
}
TEST(InvSqrtOracleTests, SpdSingleton) {
  expect_inverse_sqrt<double>(1, 5);
}
TEST(InvSqrtOracleTests, SpdLarger) {
  expect_inverse_sqrt<double>(64, 6);
  expect_inverse_sqrt<float>(48, 7);
}

// n <= 0 is a no-op success -- nothing to diagonalize.
TEST(InvSqrtOracleTests, Empty) {
  const std::vector<double> s(1, 0.0);
  auto handle = shared_device();
  auto h = make_handles(handle);
  auto d = to_device(handle, s);
  const auto status = inv_sqrt<double>(h.blas, h.solver, handle->stream().get(), 0, d.data(),
                                       d.data(), d.data(), d.data(), d.data(), d.data(), 0,
                                       nullptr);
  destroy_handles(h);
  EXPECT_TRUE(status.ok());
}

// An INDEFINITE symmetric S (negative eigenvalues): the floor drops the
// non-positive modes before the square root, so X must be finite (no NaN/Inf)
// rather than X^T S X = I, which cannot hold for a non-PD S.
TEST(InvSqrtOracleTests, IndefiniteStaysFinite) {
  const int n = 8;
  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // S = (R + R^T)/2, symmetric and generically indefinite, column-major.
  std::vector<double> r(nn);
  for (auto &v : r) {
    v = dist(rng);
  }
  std::vector<double> s(nn, 0.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      s[static_cast<std::size_t>(j) * n + i] =
          0.5 * (r[static_cast<std::size_t>(j) * n + i] + r[static_cast<std::size_t>(i) * n + j]);
    }
  }

  const auto x = run_inv_sqrt(s, n);
  for (std::size_t i = 0; i < nn; ++i) {
    EXPECT_TRUE(std::isfinite(x[i])) << "X not finite at " << i;
  }
}

} // namespace
} // namespace calaman
