// Oracle test for calaman.lange -- the ?lange matrix norm (max-abs, 1-norm,
// infinity-norm, Frobenius) of a column-major matrix. The oracle is LAPACKE_?lange
// itself (lapacke.h arrives with calaman::lapack_reference), computed in the SAME
// precision on the host. The device path is a two-stage block reduction, a
// different summation order from the reference, so agreement is real evidence.
//
// Inputs are small mixed-sign integer ramps, so every entry is exact and float and
// double behave identically; the only GPU-vs-reference difference is summation
// order, absorbed by a relative tolerance.
//
// The numerical suites stage A on the device and run the kernels, so they are
// REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`) -- including the
// empty-matrix case, which writes 0 with a device memset rather than returning
// early. Built only when calaman::lapack_reference exists (docs/architecture.md
// §3); its CMakeLists.txt returns early otherwise, so its absence is a missing
// tier, not a silent pass.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lange;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;
using wwr::extension::HostBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
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
T scalar_from_device(std::shared_ptr<DeviceHandle> handle, const DeviceBuffer<T> &device) {
  HostBuffer<T> host(1);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return host.data()[0];
}

// The oracle: LAPACKE_?lange over the same column-major storage and lda.
char norm_char(MatrixNorm which) {
  switch (which) {
  case MatrixNorm::max_abs:
    return 'M';
  case MatrixNorm::one:
    return '1';
  case MatrixNorm::inf:
    return 'I';
  case MatrixNorm::frobenius:
    return 'F';
  }
  return 'M';
}

float ref_lange(char norm, int m, int n, const float *a, int lda) {
  return LAPACKE_slange(LAPACK_COL_MAJOR, norm, m, n, a, lda);
}
double ref_lange(char norm, int m, int n, const double *a, int lda) {
  return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, m, n, a, lda);
}

// A relative tolerance generous enough for device-vs-LAPACKE summation order,
// tight enough to catch a wrong norm. Same shape as columnwise_ell1's oracle tol.
template<typename T>
T norm_tol(T ref) {
  const T rel = std::is_same_v<T, float> ? T(1e-4) : T(1e-11);
  const T atol = std::is_same_v<T, float> ? T(1e-4) : T(1e-11);
  const T a = ref < T{0} ? -ref : ref;
  return rel * a + atol;
}

// Mixed-sign ramp so |.| is non-trivial; small so the sums stay well inside the
// mantissa and float and double behave identically.
template<typename T>
T a_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 19) - 9);
}

// Stage a rows-by-cols matrix with leading dimension lda (lda >= rows), run
// lange on the device for one norm, and check against LAPACKE. Padding rows
// between rows and lda hold a huge sentinel the kernels must ignore.
template<typename T>
void expect_matches_reference(MatrixNorm which, std::size_t rows, std::size_t cols,
                              std::size_t lda) {
  ASSERT_GE(lda, rows);
  const std::size_t total = lda * cols;

  const T sentinel = static_cast<T>(1e6); // in the padding rows; must not be read
  std::vector<T> a(total, sentinel);
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < rows; ++i) {
      a[j * lda + i] = a_at<T>(j * rows + i);
    }
  }

  const T oracle = ref_lange(norm_char(which), static_cast<int>(rows), static_cast<int>(cols),
                             a.data(), static_cast<int>(lda));

  auto d_a = to_device(shared_device(), a);
  DeviceBuffer<T> d_result(1, shared_device());

  const Status s = lange<T>(shared_device()->stream().get(), which, rows, cols, d_a.data(), lda,
                            d_result.data());
  ASSERT_TRUE(s.ok()) << "lange returned " << s.name() << ": " << s.message();
  const T got = scalar_from_device(shared_device(), d_result);

  EXPECT_NEAR(got, oracle, norm_tol(oracle))
      << "norm=" << norm_char(which) << " rows=" << rows << " cols=" << cols << " lda=" << lda;
}

template<typename T>
void run_shapes(MatrixNorm which) {
  expect_matches_reference<T>(which, 1, 1, 1);       // single element
  expect_matches_reference<T>(which, 1, 8, 1);       // single row
  expect_matches_reference<T>(which, 8, 1, 8);       // single column
  expect_matches_reference<T>(which, 7, 5, 7);       // small rectangle, contiguous
  expect_matches_reference<T>(which, 16, 16, 16);    // square
  expect_matches_reference<T>(which, 5, 4, 9);       // lda > rows: padded submatrix view
  expect_matches_reference<T>(which, 1000, 3, 1000); // rows > block size: strided fold
  expect_matches_reference<T>(which, 1000, 3, 1040); // and with padding
  expect_matches_reference<T>(which, 3, 1000, 3);    // many columns / wide rows
}

// ========================================================================
// Device: numerical agreement with LAPACKE_?lange, per norm
// ========================================================================

TEST(LangeOracleTests, MaxAbsFloat) {
  run_shapes<float>(MatrixNorm::max_abs);
}
TEST(LangeOracleTests, MaxAbsDouble) {
  run_shapes<double>(MatrixNorm::max_abs);
}
TEST(LangeOracleTests, OneNormFloat) {
  run_shapes<float>(MatrixNorm::one);
}
TEST(LangeOracleTests, OneNormDouble) {
  run_shapes<double>(MatrixNorm::one);
}
TEST(LangeOracleTests, InfNormFloat) {
  run_shapes<float>(MatrixNorm::inf);
}
TEST(LangeOracleTests, InfNormDouble) {
  run_shapes<double>(MatrixNorm::inf);
}
TEST(LangeOracleTests, FrobeniusFloat) {
  run_shapes<float>(MatrixNorm::frobenius);
}
TEST(LangeOracleTests, FrobeniusDouble) {
  run_shapes<double>(MatrixNorm::frobenius);
}

// The 1-norm (max column sum) and infinity-norm (max row sum) must differ on a
// non-symmetric matrix -- a guard that the row/column directions are not swapped.
TEST(LangeOracleTests, OneAndInfDifferOnAsymmetric) {
  // Column 0 is heavy; row 0 is light: max col sum != max row sum.
  const std::size_t rows = 3;
  const std::size_t cols = 3;
  std::vector<double> a = {
      // column-major: col0 = {10,10,10}, col1 = {1,1,1}, col2 = {1,1,1}
      10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  };
  const double one = ref_lange('1', rows, cols, a.data(), rows);
  const double inf = ref_lange('I', rows, cols, a.data(), rows);
  ASSERT_NE(one, inf) << "test matrix must distinguish the two norms";

  auto d_a = to_device(shared_device(), a);
  DeviceBuffer<double> d_result(1, shared_device());

  ASSERT_TRUE(lange<double>(shared_device()->stream().get(), MatrixNorm::one, rows, cols,
                            d_a.data(), rows, d_result.data())
                  .ok());
  EXPECT_NEAR(scalar_from_device(shared_device(), d_result), one, norm_tol(one));

  ASSERT_TRUE(lange<double>(shared_device()->stream().get(), MatrixNorm::inf, rows, cols,
                            d_a.data(), rows, d_result.data())
                  .ok());
  EXPECT_NEAR(scalar_from_device(shared_device(), d_result), inf, norm_tol(inf));
}

// An empty matrix writes 0, matching DLANGE. A device memset, so this is a GPU
// test: the result scalar is pre-seeded to a sentinel that the call must clear.
TEST(LangeOracleTests, EmptyMatrixWritesZero) {
  std::vector<double> dummy = {42.0};
  auto d_a = to_device(shared_device(), dummy);
  DeviceBuffer<double> d_result(1, shared_device());
  const std::vector<double> sentinel = {-12345.0};
  wwr::extension::copy(d_result, to_device(shared_device(), sentinel),
                       shared_device()->stream().get());
  wwr::wwrStreamSynchronize(shared_device()->stream().get());

  ASSERT_TRUE(lange<double>(shared_device()->stream().get(), MatrixNorm::frobenius, 0, 4,
                            d_a.data(), 1, d_result.data())
                  .ok());
  EXPECT_DOUBLE_EQ(scalar_from_device(shared_device(), d_result), 0.0) << "m == 0 must write 0";

  wwr::extension::copy(d_result, to_device(shared_device(), sentinel),
                       shared_device()->stream().get());
  wwr::wwrStreamSynchronize(shared_device()->stream().get());
  ASSERT_TRUE(lange<double>(shared_device()->stream().get(), MatrixNorm::one, 4, 0, d_a.data(), 4,
                            d_result.data())
                  .ok());
  EXPECT_DOUBLE_EQ(scalar_from_device(shared_device(), d_result), 0.0) << "n == 0 must write 0";
}

} // namespace
} // namespace calaman
