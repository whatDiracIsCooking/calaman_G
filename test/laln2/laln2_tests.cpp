// Oracle test for calaman.laln2: the device solve of the scaled 1x1 / 2x2 real
// system (ca*op(A) - w*D) X = SCALE*B with a real or complex right-hand side must
// agree with reference LAPACK -- ?laln2 -- computed in the SAME precision on the
// host. The device path must produce the same X, the same SCALE, the same XNORM,
// and the same INFO.
//
// ?laln2 is an auxiliary routine with no LAPACKE C binding, so the oracle calls
// the Fortran symbol slaln2_ / dlaln2_ directly, exactly as the lasy2 suite calls
// slasy2_ / dlasy2_. The one LOGICAL (LTRANS) arrives as a 4-byte int (0 / 1);
// every other argument is passed by reference, matching the gfortran ABI.
//
// Both paths run the identical algorithm (same complete-pivoting order, same
// SMINI perturbation, same DLADIV for the complex divisions), so for
// well-conditioned inputs INFO is 0 on both and only rounding (device FMA
// contraction) separates the numbers -- a tight relative tolerance. One edge case
// pins the INFO==1 perturbation branch, where the path is identical on host and
// device.
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
import calaman.laln2;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

// Reference ?laln2: no LAPACKE binding, so the Fortran symbols directly. All
// arguments by reference; LOGICAL is a 4-byte int (0 / 1).
extern "C" {
void slaln2_(const int *ltrans, const int *na, const int *nw, const float *smin, const float *ca,
             const float *a, const int *lda, const float *d1, const float *d2, const float *b,
             const int *ldb, const float *wr, const float *wi, float *x, const int *ldx,
             float *scale, float *xnorm, int *info);
void dlaln2_(const int *ltrans, const int *na, const int *nw, const double *smin, const double *ca,
             const double *a, const int *lda, const double *d1, const double *d2, const double *b,
             const int *ldb, const double *wr, const double *wi, double *x, const int *ldx,
             double *scale, double *xnorm, int *info);
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
void ref_laln2(int ltrans, int na, int nw, float smin, float ca, float *a, int lda, float d1,
               float d2, float *b, int ldb, float wr, float wi, float *x, int ldx, float *scale,
               float *xnorm, int *info) {
  slaln2_(&ltrans, &na, &nw, &smin, &ca, a, &lda, &d1, &d2, b, &ldb, &wr, &wi, x, &ldx, scale,
          xnorm, info);
}
void ref_laln2(int ltrans, int na, int nw, double smin, double ca, double *a, int lda, double d1,
               double d2, double *b, int ldb, double wr, double wi, double *x, int ldx,
               double *scale, double *xnorm, int *info) {
  dlaln2_(&ltrans, &na, &nw, &smin, &ca, a, &lda, &d1, &d2, b, &ldb, &wr, &wi, x, &ldx, scale,
          xnorm, info);
}

// A relative tolerance tight enough to catch a wrong sign, pivot or scale yet
// generous for the handful of flops each path's solve does (device FMA
// contraction is the only divergence for the well-conditioned cases).
template<typename T>
T tol(T ref) {
  const T a = ref < T{0} ? -ref : ref;
  return T{256} * eps<T>() * (a + T{1});
}

// One case: laln2 on the device must match the reference. A and B are stored
// column-major with leading dimension 2 (a 2x2 slab), so the same arrays serve
// every (na, nw) shape -- the solve reads only the leading na rows / nw cols.
template<typename T>
void run_case(bool ltrans, int na, int nw, T smin, T ca, std::array<T, 4> a, T d1, T d2,
              std::array<T, 4> b, T wr, T wi, const char *ctx) {
  constexpr int ld = 2;

  // Reference overwrites its own copies; x is sentinel-filled so an unwritten
  // entry would show, though only the leading na x nw block is compared.
  std::array<T, 4> r_a = a, r_b = b, r_x;
  r_x.fill(static_cast<T>(-999));
  T r_scale = static_cast<T>(-1), r_xnorm = static_cast<T>(-1);
  int r_info = -1;
  ref_laln2(ltrans ? 1 : 0, na, nw, smin, ca, r_a.data(), ld, d1, d2, r_b.data(), ld, wr, wi,
            r_x.data(), ld, &r_scale, &r_xnorm, &r_info);

  auto handle = shared_device();
  auto d_a = to_device(handle, std::vector<T>(a.begin(), a.end()));
  auto d_b = to_device(handle, std::vector<T>(b.begin(), b.end()));
  auto d_x = to_device(handle, std::vector<T>(4, static_cast<T>(-999)));
  auto d_scale = to_device(handle, std::vector<T>{static_cast<T>(-1)});
  auto d_xnorm = to_device(handle, std::vector<T>{static_cast<T>(-1)});
  auto d_info = to_device(handle, std::vector<int>{-1});

  const auto status =
      laln2<T>(handle->stream().get(), ltrans, na, nw, smin, ca, d_a.data(), ld, d1, d2,
               d_b.data(), ld, wr, wi, d_x.data(), ld, d_scale.data(), d_xnorm.data(),
               d_info.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << ctx;

  const auto g_x = from_device(handle, d_x, 4);
  const T g_scale = from_device(handle, d_scale, 1)[0];
  const T g_xnorm = from_device(handle, d_xnorm, 1)[0];
  const int g_info = from_device(handle, d_info, 1)[0];

  EXPECT_EQ(g_info, r_info) << ctx << " info";
  EXPECT_NEAR(g_scale, r_scale, tol(r_scale)) << ctx << " scale";
  EXPECT_NEAR(g_xnorm, r_xnorm, tol(r_xnorm)) << ctx << " xnorm";
  for (int j = 1; j <= nw; ++j) {
    for (int i = 1; i <= na; ++i) {
      const int idx = (i - 1) + (j - 1) * ld;
      EXPECT_NEAR(g_x[idx], r_x[idx], tol(r_x[idx])) << ctx << " x(" << i << "," << j << ")";
    }
  }
}

// Well-conditioned slabs: A diagonally dominant and ca*A - w*D safely nonsingular
// for the (wr, wi) and (d1, d2) below, so info is 0 for both ltrans values and
// both nw. Column-major: {a11, a21, a12, a22}.
template<typename T>
std::array<T, 4> a_slab() {
  return {static_cast<T>(6), static_cast<T>(0.4), static_cast<T>(-0.7), static_cast<T>(9)};
}
template<typename T>
std::array<T, 4> b_slab() {
  return {static_cast<T>(1), static_cast<T>(-3), static_cast<T>(2), static_cast<T>(0.5)};
}

// Every (ltrans) x (na, nw) combination with the well-conditioned slab.
template<typename T>
void run_shape(int na, int nw) {
  const T smin = static_cast<T>(1e-3);
  const T ca = static_cast<T>(1.25);
  const T d1 = static_cast<T>(1);
  const T d2 = static_cast<T>(1);
  const T wr = static_cast<T>(0.3);
  const T wi = nw == 2 ? static_cast<T>(0.7) : static_cast<T>(0);
  for (const bool ltrans : {false, true}) {
    char ctx[64];
    std::snprintf(ctx, sizeof(ctx), "na=%d nw=%d ltrans=%d", na, nw, static_cast<int>(ltrans));
    run_case<T>(ltrans, na, nw, smin, ca, a_slab<T>(), d1, d2, b_slab<T>(), wr, wi, ctx);
  }
}

template<typename T>
void run_all_shapes() {
  run_shape<T>(1, 1); // real 1x1
  run_shape<T>(1, 2); // complex 1x1
  run_shape<T>(2, 1); // real 2x2
  run_shape<T>(2, 2); // complex 2x2
}

} // namespace

TEST(Laln2OracleTests, MatchesReferenceFloat) {
  run_all_shapes<float>();
}

TEST(Laln2OracleTests, MatchesReferenceDouble) {
  run_all_shapes<double>();
}

// Near-singular C forces the SMINI perturbation (info == 1). With ca*A == w*D on
// the diagonal and tiny off-diagonals, every singular value falls below smin, so
// the reference perturbs C up to SMINI*identity. The perturbation path is
// identical on host and device, so X / scale / xnorm / info all match.
TEST(Laln2OracleTests, SingularPerturbs) {
  // na=1 real: ca*A11 - wr*D1 == 0 exactly -> C perturbed to SMINI.
  const std::array<float, 4> a1{2, 0, 0, 0};
  const std::array<float, 4> b1{1, 0, 0, 0};
  run_case<float>(false, 1, 1, 1e-2F, 1.0F, a1, 2.0F, 0.0F, b1, 1.0F, 0.0F,
                  "singular 1x1 (float)");
  const std::array<double, 4> da1{2, 0, 0, 0};
  const std::array<double, 4> db1{1, 0, 0, 0};
  run_case<double>(false, 1, 1, 1e-2, 1.0, da1, 2.0, 0.0, db1, 1.0, 0.0, "singular 1x1 (double)");

  // na=2: tiny C with smin large, so norm(C) < smini -> SMINI*identity branch.
  const std::array<double, 4> a2{1e-8, 0, 0, 1e-8};
  const std::array<double, 4> b2{1, -2, 0.5, 4};
  run_case<double>(false, 2, 1, 1e-2, 1.0, a2, 0.0, 0.0, b2, 0.0, 0.0, "singular 2x2 real");
  run_case<double>(false, 2, 2, 1e-2, 1.0, a2, 0.0, 0.0, b2, 0.0, 0.0, "singular 2x2 complex");
}

} // namespace calaman
