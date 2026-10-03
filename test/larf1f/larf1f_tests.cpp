// Oracle test for calaman.larf1f: applying the elementary reflector
// H = I - tau*v*v^T with an IMPLICIT unit v(1) must agree with the reference
// LAPACK for both sides, computed in the SAME precision on the host.
//
// The oracle is netlib's plain ?larf (dlarf_ / slarf_) through its Fortran ABI.
// This LAPACK build (3.12) ships no ?larf1f, but ?larf1f(v) is by definition
// ?larf applied to the full vector whose first element is 1: so the oracle calls
// ?larf with a reference copy of v whose head has been forced to 1. The device
// input keeps a NON-1 sentinel in v(1) (the ramp value, deliberately not 1), so
// a run that matches proves larf1f ignores the stored head rather than reading
// it. The reference matrix is column-major, matching C's storage.
//
// C is m x n, non-square in every case so a side-L/side-R mix-up (which swaps
// the roles of the m- and n-length vectors) cannot pass by coincidence. v has
// length m for side L and n for side R; the device workspace w has the
// complementary length. The m == 1 (side L) and n == 1 (side R) cases exercise
// the scal fast-path, where the whole reflector is the bare implicit 1.
//
// Inputs are small, so H*C stays well inside the mantissa and the only spread
// between the GPU and the reference is BLAS summation order, absorbed by the
// shared factorization_tol. tau == 0 is checked separately: H is the identity,
// so C must come back bitwise-unchanged.
//
// REQUIRES_GPU (see CMakeLists.txt): every case allocates device memory and runs
// the gemv/ger/axpy/scal kernels, so `ctest -LE gpu` excludes it. Built only when
// calaman::lapack_reference exists; its CMakeLists.txt returns early otherwise,
// so its absence is a missing tier, not a silent pass (docs/architecture.md §3).

#include <gtest/gtest.h>

#include <cstddef> // std::size_t for the Fortran prototypes below

// Netlib ?larf through its Fortran ABI. gfortran passes scalars/arrays by
// reference and appends a hidden length for each CHARACTER argument (here the
// trailing size_t for `side`). The symbols live in LAPACK::LAPACK, linked by
// calaman::lapack_reference; this build ships no LAPACKE_?larf C binding.
extern "C" {
void slarf_(const char *side, const int *m, const int *n, const float *v, const int *incv,
            const float *tau, float *c, const int *ldc, float *work, std::size_t side_len);
void dlarf_(const char *side, const int *m, const int *n, const double *v, const int *incv,
            const double *tau, double *c, const int *ldc, double *work, std::size_t side_len);
}

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.larf1f;
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
using test::shared_device;
using test::factorization_tol;
using test::frobenius_norm;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

/// @brief Upload `host` to a fresh device buffer on `handle`'s stream
template<typename T>
DeviceBuffer<T> to_device(std::shared_ptr<DeviceHandle> handle, const HostBuffer<T> &host,
                          std::size_t n) {
  DeviceBuffer<T> device(n, handle);
  wwr::extension::copy(device, host, handle->stream().get());
  return device;
}

/// @brief Download `device` (length n) back to a host vector
template<typename T>
std::vector<T> from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<T> &device,
                           std::size_t n) {
  HostBuffer<T> host(n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  std::vector<T> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = host.data()[i];
  }
  return out;
}

// The reference oracle: netlib ?larf (Fortran ABI), same precision as the
// device path, column-major to match C's storage. Caller forces v(1) = 1 so this
// plain ?larf computes exactly what ?larf1f would.
void ref_larf(char side, int m, int n, const float *v, int incv, float tau, float *c, int ldc,
              float *work) {
  slarf_(&side, &m, &n, v, &incv, &tau, c, &ldc, work, 1);
}
void ref_larf(char side, int m, int n, const double *v, int incv, double tau, double *c, int ldc,
              double *work) {
  dlarf_(&side, &m, &n, v, &incv, &tau, c, &ldc, work, 1);
}

// Small exactly-representable ramps of mixed sign for C, v and the scalar tau.
template<typename T>
T c_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 11) - 5);
}
template<typename T>
T v_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 7) - 3);
}

/// @brief calaman::larf1f on the device must match the reference for one case
///
/// C is m x n column-major with ldc == m. v has length m (side L) or n (side R);
/// the device input stores a non-1 sentinel in v(1), while the reference copy
/// forces v(1) = 1. The final C is compared element-by-element against LAPACK's,
/// to the shared factorization tolerance scaled by ||C_out||.
template<typename T>
void expect_matches_reference(Side side, int m, int n, T tau) {
  const bool left = side == Side::L;
  const int vlen = left ? m : n;
  const int wlen = left ? n : m;
  const std::size_t cn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  auto handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  HostBuffer<T> host_c(cn);
  HostBuffer<T> host_v(static_cast<std::size_t>(vlen));
  std::vector<T> ref_c(cn);
  std::vector<T> ref_v(static_cast<std::size_t>(vlen));
  for (std::size_t i = 0; i < cn; ++i) {
    host_c.data()[i] = c_at<T>(i);
    ref_c[i] = host_c.data()[i];
  }
  for (int i = 0; i < vlen; ++i) {
    host_v.data()[static_cast<std::size_t>(i)] = v_at<T>(static_cast<std::size_t>(i));
    ref_v[static_cast<std::size_t>(i)] = host_v.data()[static_cast<std::size_t>(i)];
  }
  // The implicit unit head: the device keeps host_v[0]'s sentinel, the reference
  // gets v(1) = 1. v_at(0) == -3, so the sentinel is not accidentally 1.
  ref_v[0] = static_cast<T>(1);

  // Reference: overwrite ref_c with H applied, in host memory.
  std::vector<T> ref_work(static_cast<std::size_t>(wlen));
  const char side_char = left ? 'L' : 'R';
  ref_larf(side_char, m, n, ref_v.data(), 1, tau, ref_c.data(), m, ref_work.data());

  auto d_c = to_device(handle, host_c, cn);
  auto d_v = to_device(handle, host_v, static_cast<std::size_t>(vlen));
  DeviceBuffer<T> d_w(static_cast<std::size_t>(wlen), handle);

  const auto status = larf1f<T>(blas, side, m, n, d_v.data(), 1, tau, d_c.data(), m, d_w.data());
  const auto got = from_device(handle, d_c, cn);
  wwr::wwrblasDestroy(blas);

  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS)
      << "side=" << side_char << " m=" << m << " n=" << n;
  const T tol = factorization_tol<T>(frobenius_norm<T>(ref_c), static_cast<std::size_t>(m),
                                     static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < cn; ++i) {
    EXPECT_NEAR(got[i], ref_c[i], tol)
        << "side=" << side_char << " m=" << m << " n=" << n << " i=" << i;
  }
}

TEST(Larf1fOracleTests, MatchesReferenceLeftFloat) {
  expect_matches_reference<float>(Side::L, 5, 3, 0.7F);
  expect_matches_reference<float>(Side::L, 7, 4, 1.3F);
  expect_matches_reference<float>(Side::L, 3, 8, 0.4F);
}

TEST(Larf1fOracleTests, MatchesReferenceLeftDouble) {
  expect_matches_reference<double>(Side::L, 5, 3, 0.7);
  expect_matches_reference<double>(Side::L, 7, 4, 1.3);
  expect_matches_reference<double>(Side::L, 3, 8, 0.4);
}

TEST(Larf1fOracleTests, MatchesReferenceRightFloat) {
  expect_matches_reference<float>(Side::R, 5, 3, 0.7F);
  expect_matches_reference<float>(Side::R, 7, 4, 1.3F);
  expect_matches_reference<float>(Side::R, 8, 3, 0.4F);
}

TEST(Larf1fOracleTests, MatchesReferenceRightDouble) {
  expect_matches_reference<double>(Side::R, 5, 3, 0.7);
  expect_matches_reference<double>(Side::R, 7, 4, 1.3);
  expect_matches_reference<double>(Side::R, 8, 3, 0.4);
}

TEST(Larf1fOracleTests, UnitReflectorScalFastPath) {
  // The whole vector is the implicit 1: side L with m == 1, side R with n == 1.
  // H = (1 - tau) and the update is a single scal on C's one row/column -- the
  // branch that also avoids a degenerate 0-length tail gemv.
  expect_matches_reference<float>(Side::L, 1, 6, 0.7F);
  expect_matches_reference<double>(Side::L, 1, 6, 1.3);
  expect_matches_reference<float>(Side::R, 6, 1, 0.4F);
  expect_matches_reference<double>(Side::R, 6, 1, 1.3);
}

TEST(Larf1fOracleTests, TauZeroIsNoop) {
  // tau == 0: H is the identity, so C must come back exactly as uploaded, for
  // both sides. The stronger check here is bitwise against the ORIGINAL C -- a
  // stray ger/axpy with a nonzero alpha would perturb it below any oracle
  // tolerance and still fail.
  for (const Side side : {Side::L, Side::R}) {
    const int m = 6;
    const int n = 4;
    const std::size_t cn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const int vlen = side == Side::L ? m : n;
    const int wlen = side == Side::L ? n : m;

    auto handle = shared_device();
    wwr::wwrblasHandle_t blas{};
    ASSERT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
    ASSERT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

    HostBuffer<double> host_c(cn);
    HostBuffer<double> host_v(static_cast<std::size_t>(vlen));
    for (std::size_t i = 0; i < cn; ++i) {
      host_c.data()[i] = c_at<double>(i);
    }
    for (int i = 0; i < vlen; ++i) {
      host_v.data()[static_cast<std::size_t>(i)] = v_at<double>(static_cast<std::size_t>(i));
    }

    auto d_c = to_device(handle, host_c, cn);
    auto d_v = to_device(handle, host_v, static_cast<std::size_t>(vlen));
    DeviceBuffer<double> d_w(static_cast<std::size_t>(wlen), handle);

    const auto status = larf1f<double>(blas, side, m, n, d_v.data(), 1, 0.0, d_c.data(), m,
                                       d_w.data());
    const auto got = from_device(handle, d_c, cn);
    wwr::wwrblasDestroy(blas);

    EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
    for (std::size_t i = 0; i < cn; ++i) {
      EXPECT_DOUBLE_EQ(got[i], c_at<double>(i)) << "side=" << (side == Side::L ? 'L' : 'R')
                                                << " i=" << i;
    }
  }
}

} // namespace
} // namespace calaman
