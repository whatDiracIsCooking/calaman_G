// Oracle test for calaman.gehrd: the device blocked reduction of a general
// matrix to upper Hessenberg form by an orthogonal similarity Q^T*A*Q = H must
// agree with reference LAPACK -- ?gehrd -- computed in the SAME precision on the
// host. The device path must produce the same packed A (H plus the reflector
// tails), the same tau, and the same INFO (0 for every well-posed case here).
//
// ?gehrd has a LAPACKE C binding, but this suite calls the Fortran symbol
// sgehrd_ / dgehrd_ directly, as the gehd2 / lahr2 suites call ?gehd2_ / ?lahr2_
// -- a uniform oracle shape and no LAPACKE workspace-query indirection. Every
// argument is passed by reference, matching the gfortran ABI; a host workspace
// query (LWORK = -1) sizes the oracle's WORK the LAPACK way.
//
// Both paths run mathematically the same reduction; only the block size differs
// (calaman.gehrd's internal nb vs the reference's ilaenv choice), which changes
// only the floating-point accumulation order, not the unique Hessenberg factor.
// A direct element-wise comparison of A and tau pins the device output to the
// reference's representation to the shared factorization tolerance; an
// independent reconstruction ||A - Q*H*Q^T|| / ||A|| then re-derives Q from the
// DEVICE's own reflectors + tau and checks it is a genuine similarity reduction.
//
// Cases cover square float/double across several n and block boundaries: n well
// below the internal nb (the whole-window unblocked gehd2 path), n straddling it
// (the blocked loop plus the unblocked tail), a windowed [ilo,ihi] sub-block (the
// gebal -> gehrd path), the n <= 1 quick return, and argument validation.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages A/tau/work on the device
// and runs the reflector kernels, so `ctest -LE gpu` excludes it. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise, so
// its absence is a missing tier, not a silent pass (docs/architecture.md §3).

#include <gtest/gtest.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.gehrd;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?gehrd through its Fortran ABI. Every argument is passed by
// reference (the gfortran ABI); LWORK = -1 is the workspace query.
extern "C" {
void sgehrd_(const int *n, const int *ilo, const int *ihi, float *a, const int *lda, float *tau,
             float *work, const int *lwork, int *info);
void dgehrd_(const int *n, const int *ilo, const int *ihi, double *a, const int *lda, double *tau,
             double *work, const int *lwork, int *info);
}

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;
using test::frobenius_norm;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// The internal block size calaman.gehrd uses; the device workspace must hold the
// Y panel (n x nb) plus the T factor (nb x nb), i.e. n*nb + nb*nb elements.
constexpr int kNb = 32;

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

// Reference dispatch, same precision as the device path.
void ref_gehrd(int n, int ilo, int ihi, float *a, int lda, float *tau, float *work, int lwork,
               int *info) {
  sgehrd_(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, info);
}
void ref_gehrd(int n, int ilo, int ihi, double *a, int lda, double *tau, double *work, int lwork,
               int *info) {
  dgehrd_(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, info);
}

// Residual ||A_orig - Q*H*Q^T||_F, re-deriving Q and H from the device's own
// packed output. H is `packed` with the reflector tails zeroed (rows r > c+1 of
// columns c in [ilo-1, ihi-2]); Q = H(ilo)..H(ihi-1), each H_c = I - tau_c v v^T
// with v(c+1)=1, v(c+2:ihi-1) from `packed` below the subdiagonal. All matrices
// are n-by-n column-major with leading dimension n. (The gehd2 suite's invariant,
// applied to gehrd's output -- the packed layout and reflector convention match.)
template<typename T>
T reduction_residual(const std::vector<T> &a_orig, const std::vector<T> &packed,
                     const std::vector<T> &tau, int n, int ilo, int ihi) {
  const auto ld = static_cast<std::size_t>(n);
  const int ilo0 = ilo - 1;
  const int ihi0 = ihi - 1;

  // H: packed with the stored reflector tails cleared.
  std::vector<T> b = packed;
  for (int c = ilo0; c <= ihi0 - 1; ++c) {
    for (int r = c + 2; r <= ihi0; ++r) {
      b[static_cast<std::size_t>(c) * ld + static_cast<std::size_t>(r)] = T{0};
    }
  }

  // Each reflector's full length-n vector v (zeros outside [c+1, ihi-1]).
  auto make_v = [&](int c) {
    std::vector<T> v(static_cast<std::size_t>(n), T{0});
    v[static_cast<std::size_t>(c + 1)] = T{1};
    for (int r = c + 2; r <= ihi0; ++r) {
      v[static_cast<std::size_t>(r)] =
          packed[static_cast<std::size_t>(c) * ld + static_cast<std::size_t>(r)];
    }
    return v;
  };

  // B := Q*B = H(ilo)..H(ihi-1) * B. Apply the reflectors right-to-left so the
  // leftmost factor H(ilo) lands last. H_c*B = B - tau_c v (v^T B).
  for (int c = ihi0 - 1; c >= ilo0; --c) {
    const std::vector<T> v = make_v(c);
    const T t = tau[static_cast<std::size_t>(c)];
    for (int j = 0; j < n; ++j) {
      T w{};
      for (int r = 0; r < n; ++r) {
        w += v[static_cast<std::size_t>(r)] *
             b[static_cast<std::size_t>(j) * ld + static_cast<std::size_t>(r)];
      }
      const T s = t * w;
      for (int r = 0; r < n; ++r) {
        b[static_cast<std::size_t>(j) * ld + static_cast<std::size_t>(r)] -=
            s * v[static_cast<std::size_t>(r)];
      }
    }
  }

  // B := B*Q^T = B * H(ihi-1)..H(ilo). B*H_c = B - tau_c (B v) v^T.
  for (int c = ihi0 - 1; c >= ilo0; --c) {
    const std::vector<T> v = make_v(c);
    const T t = tau[static_cast<std::size_t>(c)];
    std::vector<T> p(static_cast<std::size_t>(n), T{0}); // p = B v
    for (int k = 0; k < n; ++k) {
      const T vk = v[static_cast<std::size_t>(k)];
      if (vk == T{0}) {
        continue;
      }
      for (int i = 0; i < n; ++i) {
        p[static_cast<std::size_t>(i)] +=
            b[static_cast<std::size_t>(k) * ld + static_cast<std::size_t>(i)] * vk;
      }
    }
    for (int j = 0; j < n; ++j) {
      const T s = t * v[static_cast<std::size_t>(j)];
      if (s == T{0}) {
        continue;
      }
      for (int i = 0; i < n; ++i) {
        b[static_cast<std::size_t>(j) * ld + static_cast<std::size_t>(i)] -=
            s * p[static_cast<std::size_t>(i)];
      }
    }
  }

  std::vector<T> residual(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual[i] = a_orig[i] - b[i];
  }
  return frobenius_norm(residual);
}

// One case: gehrd on the device must match reference ?gehrd for a random n-by-n A
// over the window [ilo, ihi].
template<typename T>
void expect_matches_reference(int n, int ilo, int ihi, unsigned seed) {
  const int lda = n;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-3.0, 3.0);

  std::vector<T> a_orig(static_cast<std::size_t>(lda) * n);
  for (auto &x : a_orig) {
    x = static_cast<T>(dist(rng));
  }

  // ?gehrd assumes A is already upper triangular in rows/columns 1:ilo-1 and
  // ihi+1:n -- the block structure calaman.gebal leaves. Zero the strictly
  // below-diagonal entries there so the reduction is a genuine similarity on the
  // WHOLE matrix, which is what makes the ||A - Q*H*Q^T|| invariant hold. For the
  // full window (ilo=1, ihi=n) this clears nothing.
  for (int j = 0; j < n; ++j) {
    for (int i = j + 1; i < n; ++i) {
      if (j < ilo - 1 || i > ihi - 1) {
        a_orig[static_cast<std::size_t>(j) * lda + i] = T{0};
      }
    }
  }

  // Reference ?gehrd on a copy: query the optimal LWORK, then reduce. tau length
  // n-1, zeroed outside [ilo-1, ihi-1] by the reference itself.
  std::vector<T> ref_a = a_orig;
  std::vector<T> ref_tau(static_cast<std::size_t>(std::max(n - 1, 1)), T{0});
  int ref_info = -1;
  T query{};
  ref_gehrd(n, ilo, ihi, ref_a.data(), lda, ref_tau.data(), &query, -1, &ref_info);
  ASSERT_EQ(ref_info, 0) << "lwork query n=" << n;
  const int lwork = std::max(1, static_cast<int>(query));
  std::vector<T> ref_work(static_cast<std::size_t>(lwork), T{0});
  ref_info = -1;
  ref_gehrd(n, ilo, ihi, ref_a.data(), lda, ref_tau.data(), ref_work.data(), lwork, &ref_info);
  ASSERT_EQ(ref_info, 0) << "n=" << n << " ilo=" << ilo << " ihi=" << ihi;

  // Device gehrd. work holds the Y panel (n x nb) + T factor (nb x nb).
  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_a = to_device(handle, a_orig);
  std::vector<T> tau_init(static_cast<std::size_t>(std::max(n - 1, 1)), T{0});
  auto d_tau = to_device(handle, tau_init);
  const std::size_t worklen =
      static_cast<std::size_t>(std::max(n, 1)) * kNb + static_cast<std::size_t>(kNb) * kNb;
  std::vector<T> work_init(worklen, T{0});
  auto d_work = to_device(handle, work_init);

  const auto status = gehrd<T>(blas, n, ilo, ihi, d_a.data(), lda, d_tau.data(), d_work.data());
  ASSERT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS) << "n=" << n << " ilo=" << ilo << " ihi=" << ihi;

  const auto got_a = from_device(handle, d_a, static_cast<std::size_t>(lda) * n);
  const auto got_tau = from_device(handle, d_tau, static_cast<std::size_t>(std::max(n - 1, 1)));
  wwr::wwrblasDestroy(blas);

  const T norm_a = frobenius_norm(a_orig);
  const T tol =
      factorization_tol<T>(norm_a, static_cast<std::size_t>(n), static_cast<std::size_t>(n));

  // 1. Packed A matches the reference element-wise. Both reduce the same A by the
  // same reflector convention; only the block size differs (rounding), absorbed
  // by the factorization tolerance.
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t k = static_cast<std::size_t>(j) * lda + static_cast<std::size_t>(i);
      EXPECT_NEAR(got_a[k], ref_a[k], tol)
          << "A(" << i << "," << j << ") n=" << n << " ilo=" << ilo << " ihi=" << ihi;
    }
  }

  // 2. The reflector scalars tau[ilo-1 .. ihi-2] match the reference; the rest
  // are zero in both (?gehrd's INFO-contract clear, which calaman.gehrd mirrors).
  for (int c = 0; c < std::max(n - 1, 0); ++c) {
    EXPECT_NEAR(got_tau[static_cast<std::size_t>(c)], ref_tau[static_cast<std::size_t>(c)], tol)
        << "tau[" << c << "] n=" << n << " ilo=" << ilo << " ihi=" << ihi;
  }

  // 3. Independent invariant: the device output is a genuine similarity reduction.
  EXPECT_LE(reduction_residual(a_orig, got_a, got_tau, n, ilo, ihi), tol)
      << "residual n=" << n << " ilo=" << ilo << " ihi=" << ihi;
}

} // namespace

// Small n, well below the internal block size: the whole-window unblocked gehd2
// path. Mirrors the gehd2 suite's square cases, now driven through gehrd.
TEST(GehrdOracleTests, SmallSquareFloat) {
  expect_matches_reference<float>(4, 1, 4, 1);
  expect_matches_reference<float>(7, 1, 7, 2);
}
TEST(GehrdOracleTests, SmallSquareDouble) {
  expect_matches_reference<double>(4, 1, 4, 3);
  expect_matches_reference<double>(7, 1, 7, 4);
  expect_matches_reference<double>(16, 1, 16, 5);
  expect_matches_reference<double>(31, 1, 31, 6); // n == nb-1: still unblocked
}

// n straddling the internal nb == 32: the blocked loop runs, then the unblocked
// tail. These exercise the lahr2 -> Y gemm -> larfb panel path and a non-multiple
// of nb (the block-boundary bookkeeping).
TEST(GehrdOracleTests, BlockedDouble) {
  expect_matches_reference<double>(33, 1, 33, 7);  // one block + tiny tail
  expect_matches_reference<double>(48, 1, 48, 8);  // one block + tail
  expect_matches_reference<double>(64, 1, 64, 9);  // two blocks
  expect_matches_reference<double>(80, 1, 80, 10); // two blocks + tail
}
TEST(GehrdOracleTests, BlockedFloat) {
  expect_matches_reference<float>(40, 1, 40, 11);
  expect_matches_reference<float>(70, 1, 70, 12);
}

// A windowed sub-block: ilo > 1 and ihi < n, the gebal -> gehrd path. Both a
// small (unblocked) and a large (blocked) window; the rows/columns outside
// [ilo,ihi] must come back untouched.
TEST(GehrdOracleTests, WindowedDouble) {
  expect_matches_reference<double>(12, 3, 10, 20); // small window, unblocked
  expect_matches_reference<double>(50, 5, 45, 21); // large window, blocked
}
TEST(GehrdOracleTests, WindowedFloat) {
  expect_matches_reference<float>(14, 2, 11, 22);
}

// n <= 1 (and ilo == ihi) is a quick return with no reflectors: A unchanged, no
// tau written, launch succeeds.
TEST(GehrdOracleTests, NoReflectorQuickReturn) {
  expect_matches_reference<double>(1, 1, 1, 30);
  expect_matches_reference<double>(5, 3, 3, 31); // ihi == ilo: empty window
}

// Argument validation mirrors LAPACK's INFO < 0 contract: an illegal ilo/ihi/lda
// returns wwrErrorInvalidValue and touches nothing.
TEST(GehrdOracleTests, RejectsIllegalArguments) {
  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  const int n = 5;
  auto d_a = to_device(handle, std::vector<double>(static_cast<std::size_t>(n) * n, 1.0));
  auto d_tau = to_device(handle, std::vector<double>(static_cast<std::size_t>(n - 1), 0.0));
  const std::size_t worklen = static_cast<std::size_t>(n) * kNb + static_cast<std::size_t>(kNb) * kNb;
  auto d_work = to_device(handle, std::vector<double>(worklen, 0.0));

  // ilo = 0 (< 1), ihi = n+1 (> n), lda < n.
  EXPECT_EQ(gehrd<double>(blas, n, 0, n, d_a.data(), n, d_tau.data(), d_work.data()),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(gehrd<double>(blas, n, 1, n + 1, d_a.data(), n, d_tau.data(), d_work.data()),
            wwr::wwrErrorInvalidValue);
  EXPECT_EQ(gehrd<double>(blas, n, 1, n, d_a.data(), n - 1, d_tau.data(), d_work.data()),
            wwr::wwrErrorInvalidValue);

  wwr::wwrblasDestroy(blas);
}

} // namespace calaman
