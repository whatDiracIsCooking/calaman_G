// Oracle test for calaman.lanv2: the Schur-standardised 2-by-2 block must agree
// with the reference LAPACK -- slanv2 / dlanv2 -- computed in the SAME precision
// on the host. lanv2 rewrites [a b; c d] into standard form and emits the two
// eigenvalues (rt1r,rt1i),(rt2r,rt2i) and the rotation (cs,sn); the device path
// must reproduce all ten outputs.
//
// lanv2 OVERWRITES a, b, c, d, so every case stages a fresh block on the device
// and reads all four back. It takes a stream (not a BLAS handle): ?lanv2 is pure
// scalar arithmetic with no BLAS, so the handle it needs is only the one that
// orders its device reads/writes.
//
// The cases walk ?lanv2's branches, each its own call below:
//   - c == 0                      -> already triangular, identity rotation
//   - b == 0                      -> swap rows and columns
//   - a == d, sign(b) != sign(c)  -> the special equal-diagonal case
//   - real distinct eigenvalues   -> the z >= 4*eps branch
//   - complex-conjugate pair      -> the normalise-and-rotate branch (rt1i > 0)
//   - near-equal real eigenvalues -> the reduce-to-triangular sub-branch
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and
// copies the block across, so the suite is excluded by `ctest -LE gpu`. Built
// only when calaman::lapack_reference exists; its CMakeLists.txt returns early
// otherwise, so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

// Netlib ?lanv2 through its Fortran ABI. gfortran passes every scalar by
// reference; there are no CHARACTER arguments, so no hidden length is appended.
// The symbols live in LAPACK::LAPACK, linked by calaman::lapack_reference; this
// build ships no LAPACKE_?lanv2 C binding (?lanv2 is a LAPACK auxiliary routine).
extern "C" {
void slanv2_(float *a, float *b, float *c, float *d, float *rt1r, float *rt1i, float *rt2r,
             float *rt2i, float *cs, float *sn);
void dlanv2_(double *a, double *b, double *c, double *d, double *rt1r, double *rt1i, double *rt2r,
             double *rt2i, double *cs, double *sn);
}

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lanv2;
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
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// The reference oracle: reference LAPACK's ?lanv2, dispatched by type, same
// precision as the device path. a, b, c, d are overwritten with the standard form.
void ref_lanv2(float *a, float *b, float *c, float *d, float *rt1r, float *rt1i, float *rt2r,
               float *rt2i, float *cs, float *sn) {
  slanv2_(a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn);
}
void ref_lanv2(double *a, double *b, double *c, double *d, double *rt1r, double *rt1i, double *rt2r,
               double *rt2i, double *cs, double *sn) {
  dlanv2_(a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn);
}

/// @brief A relative tolerance tight enough to catch a wrong entry, eigenvalue or
///        rotation yet generous for the sqrt/hypot/divide each path performs
template<typename T>
T lanv2_tol(T a, T b, T c, T d) {
  const T m = std::max(std::max(std::abs(a), std::abs(b)), std::max(std::abs(c), std::abs(d)));
  return (T{64} * eps<T>()) * (m + T{1});
}

/// @brief One 2-by-2 case: lanv2 on the device must match the reference
///
/// Stages [a b; c d] on the device, runs the host oracle and the device path on
/// identical inputs, then checks the standardised block (read back) and all six
/// eigenvalue/rotation scalars agree.
template<typename T>
void expect_matches_reference(T a0, T b0, T c0, T d0) {
  // Reference overwrites its block in place.
  T ra = a0, rb = b0, rc = c0, rd = d0;
  T r_rt1r{}, r_rt1i{}, r_rt2r{}, r_rt2i{}, r_cs{}, r_sn{};
  ref_lanv2(&ra, &rb, &rc, &rd, &r_rt1r, &r_rt1i, &r_rt2r, &r_rt2i, &r_cs, &r_sn);

  auto handle = shared_device();

  // The block staged contiguously as [a, b, c, d]; lanv2 takes the four element
  // pointers, which need not be contiguous in general (here they are).
  HostBuffer<T> host_block(4);
  host_block.data()[0] = a0;
  host_block.data()[1] = b0;
  host_block.data()[2] = c0;
  host_block.data()[3] = d0;
  DeviceBuffer<T> d_block(4, handle);
  wwr::extension::copy(d_block, host_block, handle->stream().get());

  T *da = d_block.data();
  T got_rt1r = T{-12345}, got_rt1i = T{-12345}, got_rt2r = T{-12345}, got_rt2i = T{-12345};
  T got_cs = T{-12345}, got_sn = T{-12345};
  const auto status = lanv2<T>(handle->stream().get(), da, da + 1, da + 2, da + 3, &got_rt1r,
                               &got_rt1i, &got_rt2r, &got_rt2i, &got_cs, &got_sn);
  ASSERT_EQ(status, wwr::wwrSuccess) << "a=" << a0 << " b=" << b0 << " c=" << c0 << " d=" << d0;

  HostBuffer<T> out_block(4);
  wwr::extension::copy(out_block, d_block, handle->stream().get());
  ASSERT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);

  const T tol = lanv2_tol(a0, b0, c0, d0);
  const char *msg = "a=";
  EXPECT_NEAR(out_block.data()[0], ra, tol) << msg << a0;
  EXPECT_NEAR(out_block.data()[1], rb, tol) << msg << a0;
  EXPECT_NEAR(out_block.data()[2], rc, tol) << msg << a0;
  EXPECT_NEAR(out_block.data()[3], rd, tol) << msg << a0;
  EXPECT_NEAR(got_rt1r, r_rt1r, tol) << msg << a0;
  EXPECT_NEAR(got_rt1i, r_rt1i, tol) << msg << a0;
  EXPECT_NEAR(got_rt2r, r_rt2r, tol) << msg << a0;
  EXPECT_NEAR(got_rt2i, r_rt2i, tol) << msg << a0;
  EXPECT_NEAR(got_cs, r_cs, tol) << msg << a0;
  EXPECT_NEAR(got_sn, r_sn, tol) << msg << a0;
}

template<typename T>
void run_all_branches() {
  expect_matches_reference<T>(T{2}, T{1}, T{0}, T{3});   // c == 0: already triangular
  expect_matches_reference<T>(T{2}, T{0}, T{5}, T{3});   // b == 0: swap rows/cols
  expect_matches_reference<T>(T{2}, T{1}, T{-1}, T{2});  // a==d, sign(b)!=sign(c)
  expect_matches_reference<T>(T{4}, T{1}, T{2}, T{1});   // real distinct eigenvalues
  expect_matches_reference<T>(T{1}, T{-5}, T{2}, T{1});  // complex pair (1 +/- i*sqrt(10))
  expect_matches_reference<T>(T{3}, T{4}, T{1}, T{3});   // near-equal -> triangular reduce
  expect_matches_reference<T>(T{-2}, T{7}, T{-3}, T{5}); // mixed signs, c != 0
  expect_matches_reference<T>(T{0}, T{1}, T{-1}, T{0});  // pure rotation, complex pair
}

TEST(Lanv2OracleTests, MatchesReferenceFloat) {
  run_all_branches<float>();
}

TEST(Lanv2OracleTests, MatchesReferenceDouble) {
  run_all_branches<double>();
}

// The standard form guarantees c == 0 for real eigenvalues; for a complex pair,
// a == d and b*c < 0. Check the structural invariant directly on the device
// output, independently of the oracle, for one real and one complex case.
template<typename T>
void expect_standard_form(T a0, T b0, T c0, T d0, bool complex_pair) {
  auto handle = shared_device();
  HostBuffer<T> host_block(4);
  host_block.data()[0] = a0;
  host_block.data()[1] = b0;
  host_block.data()[2] = c0;
  host_block.data()[3] = d0;
  DeviceBuffer<T> d_block(4, handle);
  wwr::extension::copy(d_block, host_block, handle->stream().get());

  T *da = d_block.data();
  T rt1r{}, rt1i{}, rt2r{}, rt2i{}, cs{}, sn{};
  ASSERT_EQ(lanv2<T>(handle->stream().get(), da, da + 1, da + 2, da + 3, &rt1r, &rt1i, &rt2r, &rt2i,
                     &cs, &sn),
            wwr::wwrSuccess);

  HostBuffer<T> out(4);
  wwr::extension::copy(out, d_block, handle->stream().get());
  ASSERT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);

  const T tol =
      T{64} * eps<T>() * (std::abs(a0) + std::abs(b0) + std::abs(c0) + std::abs(d0) + T{1});
  if (complex_pair) {
    EXPECT_NEAR(out.data()[0], out.data()[3], tol); // aa == dd
    EXPECT_LT(out.data()[1] * out.data()[2], T{0}); // bb*cc < 0
    EXPECT_GT(rt1i, T{0});                          // rt1i > 0 for a complex pair
    EXPECT_NEAR(rt2i, -rt1i, tol);
  } else {
    EXPECT_NEAR(out.data()[2], T{0}, tol); // cc == 0
    EXPECT_EQ(rt1i, T{0});
    EXPECT_EQ(rt2i, T{0});
  }
  // The rotation is orthonormal: cs^2 + sn^2 == 1.
  EXPECT_NEAR(cs * cs + sn * sn, T{1}, tol);
}

TEST(Lanv2OracleTests, StandardFormInvariant) {
  expect_standard_form<double>(4.0, 1.0, 2.0, 1.0, /*complex_pair=*/false);
  expect_standard_form<double>(1.0, -5.0, 2.0, 1.0, /*complex_pair=*/true);
  expect_standard_form<float>(4.0F, 1.0F, 2.0F, 1.0F, /*complex_pair=*/false);
  expect_standard_form<float>(1.0F, -5.0F, 2.0F, 1.0F, /*complex_pair=*/true);
}

} // namespace
} // namespace calaman
