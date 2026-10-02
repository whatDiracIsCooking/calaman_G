// Oracle test for calaman.columnwise_ell2 -- the per-column L2 (Euclidean) norm
// out[j] = sqrt(sum_i A(i, j)^2) of a column-major matrix. There is no LAPACKE
// binding for a batched per-column L2 norm; the oracle is the reference CBLAS
// cblas_?nrm2 applied to each column independently, in the SAME precision on the
// host. nrm2 is a different implementation (and summation order -- it scales to
// avoid overflow) from the device's square-then-tree-reduce-then-sqrt, so
// agreement is real evidence.
//
// The numerical suites stage A on the device and run the two kernels, so they
// are REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`). The
// argument/no-op contract (ColumnwiseEll2ArgCheckTests) is a pure host call --
// columnwise_ell2 returns before touching the device when rows or cols is 0 --
// so it passes a null stream and host dummy pointers and runs on a card-less
// runner.
//
// Built only when calaman::lapack_reference exists (docs/architecture.md §3);
// its CMakeLists.txt returns early otherwise, so its absence is a missing tier,
// not a silent pass.

#include <gtest/gtest.h>

#include <cblas.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.columnwise_ell2;
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

// The oracle: cblas_?nrm2 over one column (rows elements, stride 1 within the
// column-major storage). A column of the staged matrix starts at a + j*lda.
float ref_nrm2(int rows, const float *col) { return cblas_snrm2(rows, col, 1); }
double ref_nrm2(int rows, const double *col) { return cblas_dnrm2(rows, col, 1); }

// A relative tolerance generous enough for device-vs-CBLAS summation order (and
// nrm2's scaling), tight enough to catch a wrong norm or a missing square. Same
// shape as calaman.diff_norm's and columnwise_ell1's oracle tolerance.
template<typename T>
T norm_tol(T ref) {
  const T rel = std::is_same_v<T, float> ? T(1e-4) : T(1e-11);
  const T atol = std::is_same_v<T, float> ? T(1e-4) : T(1e-11);
  const T a = ref < T{0} ? -ref : ref;
  return rel * a + atol;
}

// Mixed-sign ramp so the square is non-trivial and sign must drop out; small so
// the per-column sums of squares stay well inside the mantissa and float and
// double behave identically.
template<typename T>
T a_at(std::size_t i) {
  return static_cast<T>(static_cast<int>(i % 19) - 9);
}

// Stage a rows-by-cols matrix with leading dimension lda (lda >= rows), run
// columnwise_ell2 on the device, and check every column norm against the
// reference nrm2. Padding rows between rows and lda are filled with a huge
// sentinel the kernel must ignore.
template<typename T>
void expect_matches_reference(std::size_t rows, std::size_t cols, std::size_t lda) {
  ASSERT_GE(lda, rows);
  const std::size_t total = lda * cols;

  const T sentinel = static_cast<T>(1e6); // in the padding rows; must not be summed
  std::vector<T> a(total, sentinel);
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < rows; ++i) {
      a[j * lda + i] = a_at<T>(j * rows + i);
    }
  }

  std::vector<T> oracle(cols);
  for (std::size_t j = 0; j < cols; ++j) {
    oracle[j] = ref_nrm2(static_cast<int>(rows), a.data() + j * lda);
  }

  auto d_a = to_device(shared_device(), a);
  DeviceBuffer<T> d_result(cols, shared_device());

  columnwise_ell2<T>(shared_device()->stream().get(), rows, cols, d_a.data(), lda, d_result.data());
  const auto got = from_device(shared_device(), d_result, cols);

  for (std::size_t j = 0; j < cols; ++j) {
    EXPECT_NEAR(got[j], oracle[j], norm_tol(oracle[j]))
        << "col " << j << " rows=" << rows << " cols=" << cols << " lda=" << lda;
  }
}

template<typename T>
void run_shapes() {
  expect_matches_reference<T>(1, 1, 1);     // single element: sqrt(a^2) == |a|
  expect_matches_reference<T>(1, 8, 1);     // single row, many columns
  expect_matches_reference<T>(8, 1, 8);     // single column
  expect_matches_reference<T>(7, 5, 7);     // small rectangle, contiguous
  expect_matches_reference<T>(16, 16, 16);  // square
  expect_matches_reference<T>(5, 4, 9);     // lda > rows: padded submatrix view
  expect_matches_reference<T>(1000, 3, 1000); // rows > block size: strided per-thread fold
  expect_matches_reference<T>(1000, 3, 1040); // and with padding
}

// ========================================================================
// Device: numerical agreement with the reference nrm2
// ========================================================================

TEST(ColumnwiseEll2OracleTests, MatchesReferenceFloat) {
  run_shapes<float>();
}

TEST(ColumnwiseEll2OracleTests, MatchesReferenceDouble) {
  run_shapes<double>();
}

TEST(ColumnwiseEll2OracleTests, SingleRowIsMagnitude) {
  // A one-row matrix: out[j] = sqrt(A(0, j)^2) = |A(0, j)|. With negative
  // entries this catches a square missing from the fold's seed element -- the
  // seed would otherwise carry the raw (negative) value under the sqrt.
  const std::size_t cols = 6;
  std::vector<double> a(cols);
  for (std::size_t j = 0; j < cols; ++j) {
    a[j] = -static_cast<double>(j + 1); // -1, -2, ... all negative
  }

  auto d_a = to_device(shared_device(), a);
  DeviceBuffer<double> d_result(cols, shared_device());
  columnwise_ell2<double>(shared_device()->stream().get(), 1, cols, d_a.data(), 1, d_result.data());
  const auto got = from_device(shared_device(), d_result, cols);

  for (std::size_t j = 0; j < cols; ++j) {
    const double want = static_cast<double>(j + 1); // |-(j+1)|
    EXPECT_NEAR(got[j], want, norm_tol(want)) << "col " << j;
    EXPECT_GT(got[j], 0.0) << "col " << j << " must be a positive magnitude";
  }
}

// ========================================================================
// Host-only: empty input leaves the result untouched
// ========================================================================

TEST(ColumnwiseEll2ArgCheckTests, EmptyIsNoopBeforeTouchingTheDevice) {
  // rows == 0 or cols == 0 returns before any device work, so a null stream and
  // host dummy pointers are enough and the output sentinel must survive.
  float sentinel = -12345.0f;
  float a_dummy = 1.0f;
  wwr::wwrStream_t null_stream{};

  columnwise_ell2<float>(null_stream, 0, 4, &a_dummy, 1, &sentinel);
  EXPECT_FLOAT_EQ(sentinel, -12345.0f) << "rows == 0 must not write";

  columnwise_ell2<float>(null_stream, 4, 0, &a_dummy, 4, &sentinel);
  EXPECT_FLOAT_EQ(sentinel, -12345.0f) << "cols == 0 must not write";
}

} // namespace
} // namespace calaman
