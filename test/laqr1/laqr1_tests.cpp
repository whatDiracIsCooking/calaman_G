// Oracle test for calaman.laqr1: the scaled first column of (H-s1*I)(H-s2*I) must
// agree with the reference LAPACK -- slaqr1 / dlaqr1 -- computed in the SAME
// precision on the host. laqr1 reads the upper-Hessenberg H from the device and
// writes the length-N vector V back; the device path must reproduce every entry.
//
// laqr1 takes a stream (not a BLAS handle): ?laqr1 is pure scalar arithmetic with
// no BLAS, so the handle it needs is only the one that orders its device
// reads/writes. H is staged on the device, V read back and compared.
//
// The cases walk ?laqr1's branches for N = 2 and N = 3, each its own call below:
//   - real shift pair              -> SI1 = SI2 = 0
//   - complex-conjugate shift pair -> SI2 = -SI1 != 0
//   - zero first column (s == 0)   -> V == 0 early return
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and
// copies H across, so the suite is excluded by `ctest -LE gpu`. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass.

#include <gtest/gtest.h>

// Netlib ?laqr1 through its Fortran ABI. gfortran passes every scalar by
// reference; there are no CHARACTER arguments, so no hidden length is appended.
// The symbols live in LAPACK::LAPACK, linked by calaman::lapack_reference; this
// build ships no LAPACKE_?laqr1 C binding (?laqr1 is a LAPACK auxiliary routine).
extern "C" {
void slaqr1_(const int *n, const float *h, const int *ldh, const float *sr1, const float *si1,
             const float *sr2, const float *si2, float *v);
void dlaqr1_(const int *n, const double *h, const int *ldh, const double *sr1, const double *si1,
             const double *sr2, const double *si2, double *v);
}

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.laqr1;
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

// The reference oracle: reference LAPACK's ?laqr1, dispatched by type, same
// precision as the device path. V is overwritten with the scaled first column.
void ref_laqr1(const int *n, const float *h, const int *ldh, const float *sr1, const float *si1,
               const float *sr2, const float *si2, float *v) {
  slaqr1_(n, h, ldh, sr1, si1, sr2, si2, v);
}
void ref_laqr1(const int *n, const double *h, const int *ldh, const double *sr1, const double *si1,
               const double *sr2, const double *si2, double *v) {
  dlaqr1_(n, h, ldh, sr1, si1, sr2, si2, v);
}

/// @brief One case: laqr1 on the device must match the reference
///
/// Stages the n-by-n upper-Hessenberg @p hcol (column-major, ldh == n) on the
/// device, runs the host oracle and the device path on identical inputs, then
/// checks all n entries of V agree.
template<typename T>
void expect_matches_reference(int n, const std::vector<T> &hcol, T sr1, T si1, T sr2, T si2) {
  const int ldh = n;
  std::vector<T> ref_v(static_cast<std::size_t>(n), T{-12345});
  ref_laqr1(&n, hcol.data(), &ldh, &sr1, &si1, &sr2, &si2, ref_v.data());

  auto handle = shared_device();

  HostBuffer<T> host_h(hcol.size());
  for (std::size_t i = 0; i < hcol.size(); ++i) {
    host_h.data()[i] = hcol[i];
  }
  DeviceBuffer<T> d_h(hcol.size(), handle);
  wwr::extension::copy(d_h, host_h, handle->stream().get());

  DeviceBuffer<T> d_v(static_cast<std::size_t>(n), handle);
  const auto status =
      laqr1<T>(handle->stream().get(), n, d_h.data(), ldh, sr1, si1, sr2, si2, d_v.data());
  ASSERT_EQ(status, wwr::wwrSuccess) << "n=" << n;

  HostBuffer<T> out_v(static_cast<std::size_t>(n));
  wwr::extension::copy(out_v, d_v, handle->stream().get());
  ASSERT_EQ(wwr::wwrStreamSynchronize(handle->stream().get()), wwr::wwrSuccess);

  // Relative tolerance generous for the divide / shift products each entry forms.
  T hmax = T{1};
  for (const T e : hcol) {
    hmax = std::max(hmax, std::abs(e));
  }
  const T tol = T{64} * eps<T>() * (hmax + std::abs(sr1) + std::abs(sr2) + T{1});
  for (int i = 0; i < n; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    EXPECT_NEAR(out_v.data()[i], ref_v[ui], tol) << "n=" << n << " i=" << i;
  }
}

template<typename T>
void run_all_branches() {
  // N = 2 upper-Hessenberg, column-major [H11,H21, H12,H22].
  const std::vector<T> h2{T{4}, T{2}, T{1}, T{3}};
  expect_matches_reference<T>(2, h2, T{1}, T{0}, T{2}, T{0});    // real shift pair
  expect_matches_reference<T>(2, h2, T{3}, T{1}, T{3}, T{-1});   // complex-conjugate pair
  // Zero first column: H11 == SR2, H21 == 0, SI2 == 0 -> s == 0 -> V == 0.
  const std::vector<T> h2z{T{5}, T{0}, T{1}, T{3}};
  expect_matches_reference<T>(2, h2z, T{5}, T{0}, T{5}, T{0});

  // N = 3 upper-Hessenberg, column-major [H11,H21,H31, H12,H22,H32, H13,H23,H33].
  const std::vector<T> h3{T{6}, T{2}, T{0}, T{1}, T{5}, T{3}, T{7}, T{4}, T{2}};
  expect_matches_reference<T>(3, h3, T{1}, T{0}, T{2}, T{0});    // real shift pair
  expect_matches_reference<T>(3, h3, T{2}, T{3}, T{2}, T{-3});   // complex-conjugate pair
  // Zero first column: H11 == SR2, H21 == H31 == 0, SI2 == 0 -> V == 0.
  const std::vector<T> h3z{T{6}, T{0}, T{0}, T{1}, T{5}, T{3}, T{7}, T{4}, T{2}};
  expect_matches_reference<T>(3, h3z, T{6}, T{0}, T{6}, T{0});
}

TEST(Laqr1OracleTests, MatchesReferenceFloat) {
  run_all_branches<float>();
}

TEST(Laqr1OracleTests, MatchesReferenceDouble) {
  run_all_branches<double>();
}

} // namespace
} // namespace calaman
