// Oracle test for calaman.lasy2: the device solve of the order-(1..2) Sylvester
// equation op(TL)*X + ISGN*X*op(TR) = SCALE*B must agree with reference LAPACK --
// ?lasy2 -- computed in the SAME precision on the host. The device path must
// produce the same X, the same SCALE, the same XNORM, and the same INFO.
//
// ?lasy2 is an auxiliary routine with no LAPACKE C binding, so the oracle calls
// the Fortran symbol slasy2_ / dlasy2_ directly, exactly as the larf suite calls
// slarf_ / dlarf_. Fortran LOGICALs arrive as 4-byte ints (0 / 1); every other
// argument is passed by reference, matching the gfortran ABI.
//
// Both paths run the identical algorithm (same complete-pivoting order, same SMIN
// perturbation), so for well-conditioned inputs INFO is 0 on both and only
// rounding (device FMA contraction) separates the numbers -- a tight relative
// tolerance. Two edge cases pin the INFO==1 perturbation branch with exact
// cancellation, where the path is identical on host and device.
//
// REQUIRES_GPU (see CMakeLists.txt): every case stages the matrices on the device
// and runs the kernel, so the suite is excluded by `ctest -LE gpu`. Built only
// when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

#include <array>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lasy2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?lasy2: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; LOGICAL is a 4-byte int (0 / 1).
extern "C" {
void slasy2_(const int *ltranl, const int *ltranr, const int *isgn, const int *n1, const int *n2,
             const float *tl, const int *ldtl, const float *tr, const int *ldtr, const float *b,
             const int *ldb, float *scale, float *x, const int *ldx, float *xnorm, int *info);
void dlasy2_(const int *ltranl, const int *ltranr, const int *isgn, const int *n1, const int *n2,
             const double *tl, const int *ldtl, const double *tr, const int *ldtr, const double *b,
             const int *ldb, double *scale, double *x, const int *ldx, double *xnorm, int *info);
}

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::eps;
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

// Reference dispatch, same precision as the device path.
void ref_lasy2(int ltranl, int ltranr, int isgn, int n1, int n2, float *tl, int ldtl, float *tr,
               int ldtr, float *b, int ldb, float *scale, float *x, int ldx, float *xnorm,
               int *info) {
  slasy2_(&ltranl, &ltranr, &isgn, &n1, &n2, tl, &ldtl, tr, &ldtr, b, &ldb, scale, x, &ldx, xnorm,
          info);
}
void ref_lasy2(int ltranl, int ltranr, int isgn, int n1, int n2, double *tl, int ldtl, double *tr,
               int ldtr, double *b, int ldb, double *scale, double *x, int ldx, double *xnorm,
               int *info) {
  dlasy2_(&ltranl, &ltranr, &isgn, &n1, &n2, tl, &ldtl, tr, &ldtr, b, &ldb, scale, x, &ldx, xnorm,
          info);
}

// A relative tolerance tight enough to catch a wrong sign, pivot or scale yet
// generous for the handful of flops each path's solve does (device FMA
// contraction is the only divergence for the well-conditioned cases).
template<typename T>
T tol(T ref) {
  const T a = ref < T{0} ? -ref : ref;
  return T{256} * eps<T>() * (a + T{1});
}

// One case: lasy2 on the device must match the reference. All matrices are stored
// column-major with leading dimension 2 (a 2x2 slab), so the same arrays serve
// every (n1, n2) shape -- the solve reads only the leading n1 / n2 rows/cols.
template<typename T>
void run_case(bool ltranl, bool ltranr, int isgn, int n1, int n2, std::array<T, 4> tl,
              std::array<T, 4> tr, std::array<T, 4> b, const char *ctx) {
  constexpr int ld = 2;

  // Reference overwrites its own copies; x is sentinel-filled so an unwritten
  // entry would show, though only the leading n1 x n2 block is compared.
  std::array<T, 4> r_tl = tl, r_tr = tr, r_b = b, r_x;
  r_x.fill(static_cast<T>(-999));
  T r_scale = static_cast<T>(-1), r_xnorm = static_cast<T>(-1);
  int r_info = -1;
  ref_lasy2(ltranl ? 1 : 0, ltranr ? 1 : 0, isgn, n1, n2, r_tl.data(), ld, r_tr.data(), ld,
            r_b.data(), ld, &r_scale, r_x.data(), ld, &r_xnorm, &r_info);

  auto handle = shared_device();
  auto d_tl = to_device(handle, std::vector<T>(tl.begin(), tl.end()));
  auto d_tr = to_device(handle, std::vector<T>(tr.begin(), tr.end()));
  auto d_b = to_device(handle, std::vector<T>(b.begin(), b.end()));
  auto d_x = to_device(handle, std::vector<T>(4, static_cast<T>(-999)));
  auto d_scale = to_device(handle, std::vector<T>{static_cast<T>(-1)});
  auto d_xnorm = to_device(handle, std::vector<T>{static_cast<T>(-1)});
  auto d_info = to_device(handle, std::vector<int>{-1});

  const auto status =
      lasy2<T>(handle->stream().get(), ltranl, ltranr, isgn, n1, n2, d_tl.data(), ld, d_tr.data(),
               ld, d_b.data(), ld, d_scale.data(), d_x.data(), ld, d_xnorm.data(), d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const auto g_x = from_device(handle, d_x, 4);
  const T g_scale = from_device(handle, d_scale, 1)[0];
  const T g_xnorm = from_device(handle, d_xnorm, 1)[0];
  const int g_info = from_device(handle, d_info, 1)[0];

  EXPECT_EQ(g_info, r_info) << ctx << " info";
  EXPECT_NEAR(g_scale, r_scale, tol(r_scale)) << ctx << " scale";
  EXPECT_NEAR(g_xnorm, r_xnorm, tol(r_xnorm)) << ctx << " xnorm";
  for (int j = 1; j <= n2; ++j) {
    for (int i = 1; i <= n1; ++i) {
      const int idx = (i - 1) + (j - 1) * ld;
      EXPECT_NEAR(g_x[idx], r_x[idx], tol(r_x[idx])) << ctx << " x(" << i << "," << j << ")";
    }
  }
}

// Well-conditioned slabs (distinct, well-separated diagonals so op(TL) + isgn *
// op(TR) is safely nonsingular and info is 0 for every sign and transpose combo).
// Column-major: {a11, a21, a12, a22}.
template<typename T>
std::array<T, 4> tl_slab() {
  return {static_cast<T>(2), static_cast<T>(-0.3), static_cast<T>(0.5), static_cast<T>(5)};
}
template<typename T>
std::array<T, 4> tr_slab() {
  return {static_cast<T>(-11), static_cast<T>(0.2), static_cast<T>(0.7), static_cast<T>(-19)};
}
template<typename T>
std::array<T, 4> b_slab() {
  return {static_cast<T>(1), static_cast<T>(3), static_cast<T>(-2), static_cast<T>(0.5)};
}

// Every (ltranl, ltranr, isgn) combination for one (n1, n2) shape.
template<typename T>
void run_shape(int n1, int n2) {
  for (const bool ltranl : {false, true}) {
    for (const bool ltranr : {false, true}) {
      for (const int isgn : {1, -1}) {
        char ctx[64];
        std::snprintf(ctx, sizeof(ctx), "n1=%d n2=%d tl=%d tr=%d isgn=%d", n1, n2,
                      static_cast<int>(ltranl), static_cast<int>(ltranr), isgn);
        run_case<T>(ltranl, ltranr, isgn, n1, n2, tl_slab<T>(), tr_slab<T>(), b_slab<T>(), ctx);
      }
    }
  }
}

template<typename T>
void run_all_shapes() {
  run_shape<T>(1, 1);
  run_shape<T>(1, 2);
  run_shape<T>(2, 1);
  run_shape<T>(2, 2);
}

} // namespace

TEST(Lasy2OracleTests, MatchesReferenceFloat) {
  run_all_shapes<float>();
}

TEST(Lasy2OracleTests, MatchesReferenceDouble) {
  run_all_shapes<double>();
}

// 1x1 exact cancellation: TL11 + ISGN*TR11 == 0, so tau1 underflows to SMLNUM
// and the reference perturbs it (info == 1). The device path takes the same
// branch on the same exact zero.
TEST(Lasy2OracleTests, SingularOneByOnePerturbs) {
  const std::array<float, 4> tl{4, 0, 0, 0};
  const std::array<float, 4> tr{4, 0, 0, 0};
  const std::array<float, 4> b{2, 0, 0, 0};
  run_case<float>(false, false, -1, 1, 1, tl, tr, b, "singular 1x1 (float)");
  const std::array<double, 4> dtl{4, 0, 0, 0};
  const std::array<double, 4> dtr{4, 0, 0, 0};
  const std::array<double, 4> db{2, 0, 0, 0};
  run_case<double>(false, false, -1, 1, 1, dtl, dtr, db, "singular 1x1 (double)");
}

// 2x2 with TL == TR and ISGN == -1: every diagonal of the 4x4 system cancels to
// zero and every off-diagonal is zero, so all four pivots are perturbed to SMIN
// (info == 1). The elimination/back-substitution path is identical on both.
TEST(Lasy2OracleTests, SingularTwoByTwoPerturbs) {
  const std::array<float, 4> m{3, 0, 0, 3};
  const std::array<float, 4> b{1, -2, 0.5F, 4};
  run_case<float>(false, false, -1, 2, 2, m, m, b, "singular 2x2 (float)");
  const std::array<double, 4> dm{3, 0, 0, 3};
  const std::array<double, 4> db{1, -2, 0.5, 4};
  run_case<double>(false, false, -1, 2, 2, dm, dm, db, "singular 2x2 (double)");
}

// A zero order is a quick return: the reference writes only info = 0 and touches
// nothing else. lasy2 must agree -- info 0, and the launch must succeed.
TEST(Lasy2OracleTests, ZeroOrderQuickReturn) {
  auto handle = shared_device();
  auto d_tl = to_device(handle, std::vector<double>(4, 1.0));
  auto d_tr = to_device(handle, std::vector<double>(4, 1.0));
  auto d_b = to_device(handle, std::vector<double>(4, 1.0));
  auto d_x = to_device(handle, std::vector<double>(4, -999.0));
  auto d_scale = to_device(handle, std::vector<double>{-1.0});
  auto d_xnorm = to_device(handle, std::vector<double>{-1.0});
  auto d_info = to_device(handle, std::vector<int>{-1});

  const auto status =
      lasy2<double>(handle->stream().get(), false, false, 1, 0, 2, d_tl.data(), 2, d_tr.data(), 2,
                    d_b.data(), 2, d_scale.data(), d_x.data(), 2, d_xnorm.data(), d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess);
  EXPECT_EQ(from_device(handle, d_info, 1)[0], 0);
}

} // namespace calaman
