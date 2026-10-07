// Reference suite for calaman.ritz's ritz_rotate: C = B S(:, 0:count) on the
// device against V Z_count, where V is an orthonormal basis (LAPACKE ?geqrf +
// ?orgqr) and Z the eigenvectors of a random symmetric matrix (LAPACKE
// ?syevd). Every operand carries a leading dimension larger than its rows, the
// padding must survive, C must be orthonormal, and the handle's pointer mode
// (host or device on entry) must be restored. REQUIRES_GPU; float and double.

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.ritz;
import calaman.error_handling;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;
using test::frobenius_norm;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// ── LAPACKE overloads ────────────────────────────────────────────────────────

lapack_int ref_syevd(lapack_int n, float *a, float *w) {
  return LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}
lapack_int ref_syevd(lapack_int n, double *a, double *w) {
  return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}
lapack_int ref_geqrf(lapack_int m, lapack_int n, float *a, float *tau) {
  return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, m, tau);
}
lapack_int ref_geqrf(lapack_int m, lapack_int n, double *a, double *tau) {
  return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, m, tau);
}
lapack_int ref_orgqr(lapack_int m, lapack_int n, float *a, const float *tau) {
  return LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, n, n, a, m, tau);
}
lapack_int ref_orgqr(lapack_int m, lapack_int n, double *a, const double *tau) {
  return LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, a, m, tau);
}

/// rows x cols column-major, leading dimension rows.
template<typename T>
std::vector<T> random_matrix(int rows, int cols, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  std::vector<T> a(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (T &x : a) {
    x = dist(gen);
  }
  return a;
}

/// Copy a rows x cols, ld = rows matrix into ld @p ld storage padded with @p pad.
template<typename T>
std::vector<T> padded(const std::vector<T> &a, int rows, int cols, int ld, T pad) {
  std::vector<T> out(static_cast<std::size_t>(ld) * static_cast<std::size_t>(cols), pad);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      out[static_cast<std::size_t>(i + j * ld)] = a[static_cast<std::size_t>(i + j * rows)];
    }
  }
  return out;
}

template<typename T>
std::vector<T> download(wwr::wwrStream_t stream, const T *src, std::size_t count) {
  std::vector<T> out(count);
  EXPECT_EQ(
      wwr::wwrMemcpyAsync(out.data(), src, sizeof(T) * count, wwr::wwrMemcpyDeviceToHost, stream),
      wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return out;
}

template<typename T>
DeviceBuffer<T> upload(const std::shared_ptr<DeviceHandle> &handle, const std::vector<T> &src) {
  DeviceBuffer<T> d(src.size(), handle);
  const wwr::wwrStream_t stream = handle->stream().get();
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), src.data(), sizeof(T) * src.size(),
                                wwr::wwrMemcpyHostToDevice, stream),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  return d;
}

/// C = V(:, 0:k) Z(:, 0:count) on the device, V orthonormal n x k, Z the
/// eigenvectors of a symmetric k x k; checked against the host product, for
/// orthonormality, for untouched padding and for the restored pointer mode.
template<typename T>
void check_rotate(int n, int k, int count, wwr::wwrblasPointerMode_t caller_mode) {
  const std::shared_ptr<DeviceHandle> handle = shared_device();
  const wwr::wwrStream_t stream = handle->stream().get();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, stream), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetPointerMode(blas, caller_mode), wwr::WWRBLAS_STATUS_SUCCESS);

  // The basis: an orthonormal n x k V.
  std::vector<T> v = random_matrix<T>(n, k, 31);
  std::vector<T> tau(static_cast<std::size_t>(k));
  ASSERT_EQ(ref_geqrf(n, k, v.data(), tau.data()), 0);
  ASSERT_EQ(ref_orgqr(n, k, v.data(), tau.data()), 0);

  // The projected eigenvectors: Z of a random symmetric k x k (lower read).
  std::vector<T> z = random_matrix<T>(k, k, 37);
  std::vector<T> w(static_cast<std::size_t>(k));
  ASSERT_EQ(ref_syevd(k, z.data(), w.data()), 0);

  const int ldb = n + 3;
  const int lds = k + 2; // a projected matrix stored at a larger capacity
  const int ldc = n + 1;
  const T pad = T(-777);
  const auto d_b = upload(handle, padded(v, n, k, ldb, pad));
  const auto d_s = upload(handle, padded(z, k, k, lds, pad));
  auto d_c = upload(handle, std::vector<T>(static_cast<std::size_t>(ldc) * count, pad));

  EXPECT_TRUE(
      ritz_rotate<T>(blas, n, k, count, d_b.data(), ldb, d_s.data(), lds, d_c.data(), ldc).ok());
  const std::vector<T> c = download(stream, d_c.data(), static_cast<std::size_t>(ldc) * count);

  wwr::wwrblasPointerMode_t mode{};
  ASSERT_EQ(wwr::wwrblasGetPointerMode(blas, &mode), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(mode, caller_mode) << "pointer mode must be restored";
  wwr::wwrblasDestroy(blas);

  const auto at = [](const std::vector<T> &a, int ld, int i, int j) {
    return a[static_cast<std::size_t>(i + j * ld)];
  };
  const T tol = factorization_tol<T>(frobenius_norm(z), static_cast<std::size_t>(n),
                                     static_cast<std::size_t>(k));
  for (int j = 0; j < count; ++j) {
    for (int i = 0; i < n; ++i) {
      T want = T(0);
      for (int q = 0; q < k; ++q) {
        want += at(v, n, i, q) * at(z, k, q, j);
      }
      EXPECT_NEAR(at(c, ldc, i, j), want, tol) << i << "," << j;
    }
    EXPECT_EQ(at(c, ldc, n, j), pad) << "padding row of column " << j;
    for (int l = 0; l <= j; ++l) {
      T g = T(0);
      for (int i = 0; i < n; ++i) {
        g += at(c, ldc, i, j) * at(c, ldc, i, l);
      }
      EXPECT_NEAR(g, l == j ? T(1) : T(0), tol) << j << "," << l;
    }
  }
}

TEST(RitzRotateReferenceTests, LeadingColumnsFloat) {
  check_rotate<float>(40, 12, 5, wwr::WWRBLAS_POINTER_MODE_HOST);
}
TEST(RitzRotateReferenceTests, LeadingColumnsDouble) {
  check_rotate<double>(40, 12, 5, wwr::WWRBLAS_POINTER_MODE_HOST);
}
TEST(RitzRotateReferenceTests, AllColumnsFloat) {
  check_rotate<float>(33, 9, 9, wwr::WWRBLAS_POINTER_MODE_HOST);
}
TEST(RitzRotateReferenceTests, AllColumnsDouble) {
  check_rotate<double>(33, 9, 9, wwr::WWRBLAS_POINTER_MODE_HOST);
}
TEST(RitzRotateReferenceTests, RestoresDeviceModeFloat) {
  check_rotate<float>(40, 12, 3, wwr::WWRBLAS_POINTER_MODE_DEVICE);
}
TEST(RitzRotateReferenceTests, RestoresDeviceModeDouble) {
  check_rotate<double>(40, 12, 3, wwr::WWRBLAS_POINTER_MODE_DEVICE);
}

} // namespace
} // namespace calaman
