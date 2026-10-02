// Oracle test for calaman.horner -- matrix polynomial evaluation p(A) =
// sum_{k=0}^{degree} c_k A^k by Horner's method. Horner evaluation is not a
// LAPACK routine, so there is no LAPACKE_?horner binding; the oracle is an
// INDEPENDENT host accumulation of the power series -- form each power A^k by a
// reference CBLAS gemm (cblas_?gemm) and sum c_k A^k -- which is a different
// algorithm from the nested Horner recurrence under test, so agreement is real
// evidence rather than the same arithmetic computed twice.
//
// The numerical suites (HornerOracleTests, HornerPaddingTests) stage A, the
// coefficients and P on the device and run the gemm + diagonal kernels, so they
// are REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`). The workspace
// query (HornerBufferSizeTests) and the argument-checking contract
// (HornerArgCheckTests) are pure host calls -- horner()'s argument validation
// returns before it ever reads the handle or a device pointer, so those cases
// pass a null handle and host dummy pointers and run on a card-less runner.
//
// Built only when calaman::lapack_reference exists; its CMakeLists.txt returns
// early otherwise.

#include <gtest/gtest.h>

#include <cblas.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.horner;
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
using test::shared_device;
using test::eps;
using test::frobenius_norm;
using test::kTolFactor;

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

// C := A * B, all n-by-n column-major with leading dimension n, via reference
// CBLAS. The oracle's one matrix product.
void ref_matmul(int n, const float *a, const float *b, float *c) {
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
}
void ref_matmul(int n, const double *a, const double *b, double *c) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
}

// Independent oracle: p(A) = sum_{k=0}^{degree} c[k] * A^k, n-by-n column-major.
// Accumulates the powers explicitly (the "naive form" the module's README
// contrasts with Horner), so the summation order differs from the device's
// nested recurrence.
template<typename T>
std::vector<T> poly_reference(const std::vector<T> &a, int n, const std::vector<T> &c, int degree) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<T> result(nn, T{0});
  std::vector<T> apow(nn, T{0}); // running A^k, seeded at A^0 = I
  for (int i = 0; i < n; ++i) {
    result[static_cast<std::size_t>(i) * n + i] = c[0];
    apow[static_cast<std::size_t>(i) * n + i] = T{1};
  }
  std::vector<T> tmp(nn);
  for (int k = 1; k <= degree; ++k) {
    ref_matmul(n, apow.data(), a.data(), tmp.data()); // A^k = A^{k-1} * A
    apow.swap(tmp);
    const T ck = c[static_cast<std::size_t>(k)];
    for (std::size_t idx = 0; idx < nn; ++idx) {
      result[idx] += ck * apow[idx];
    }
  }
  return result;
}

// A first-order bound on the evaluation: sum_k |c_k| * ||A||^k is the worst-case
// magnitude the recurrence builds up (||A||_F overestimates the spectral norm),
// scaled by eps, the inner dimension n and the big-O factor the other suites use.
template<typename T>
T poly_tol(const std::vector<T> &a, const std::vector<T> &c, int degree, int n) {
  const T anorm = frobenius_norm(a);
  T bound{};
  T apow = T{1};
  for (int k = 0; k <= degree; ++k) {
    bound += std::abs(c[static_cast<std::size_t>(k)]) * apow;
    apow *= anorm;
  }
  return kTolFactor<T> * eps<T>() * bound * static_cast<T>(n);
}

// Run horner() on the device: stage coeffs/A/P (P seeded with @p p_inout so a
// padded leading dimension can be checked afterward), allocate the workspace the
// query asks for (null when degree is 0, exercising that accepted case), read P
// back into @p p_inout. Returns horner()'s status.
template<typename T>
Status run_horner(std::shared_ptr<DeviceHandle> handle, int n, const std::vector<T> &c, int degree,
                  const std::vector<T> &a, int lda, int ldp, std::vector<T> &p_inout) {
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_c = to_device(handle, c);
  auto d_a = to_device(handle, a);
  auto d_p = to_device(handle, p_inout);

  const std::size_t wbytes = (degree > 0) ? horner_bufferSize<T>(n) : 0;
  const std::size_t wcount = (degree > 0) ? wbytes / sizeof(T) : 1;
  DeviceBuffer<T> d_work(wcount, handle);
  void *work_ptr = (degree > 0) ? static_cast<void *>(d_work.data()) : nullptr;

  const auto status = horner<T>(blas, n, d_c.data(), degree, d_a.data(), lda, d_p.data(), ldp,
                                work_ptr, wbytes);

  p_inout = from_device(handle, d_p, static_cast<std::size_t>(ldp) * n);
  wwr::wwrblasDestroy(blas);
  return status;
}

// Square n-by-n polynomial agreement: random modest-norm A and coefficients,
// device Horner vs the explicit power-series oracle, compared elementwise.
template<typename T>
void expect_poly_matches(int n, int degree, unsigned seed) {
  const int lda = n;
  const int ldp = n;
  std::mt19937 rng(seed);
  // Entries in (-0.5, 0.5) keep ||A|| below 1 for the small n here, so the high
  // powers the oracle forms stay well inside range for float as well as double.
  std::uniform_real_distribution<double> dist(-0.5, 0.5);

  std::vector<T> a(static_cast<std::size_t>(lda) * n);
  for (auto &x : a) {
    x = static_cast<T>(dist(rng));
  }
  std::vector<T> c(static_cast<std::size_t>(degree) + 1);
  for (auto &x : c) {
    x = static_cast<T>(dist(rng) * 2.0); // coefficients in (-1, 1)
  }

  const auto oracle = poly_reference(a, n, c, degree);

  std::vector<T> got(static_cast<std::size_t>(ldp) * n, T{0});
  ASSERT_EQ(run_horner<T>(shared_device(), n, c, degree, a, lda, ldp, got),
            wwr::WWRBLAS_STATUS_SUCCESS);

  const T tol = poly_tol(a, c, degree, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(got[static_cast<std::size_t>(j) * ldp + i],
                  oracle[static_cast<std::size_t>(j) * n + i], tol)
          << "(" << i << "," << j << ") n=" << n << " degree=" << degree << " seed=" << seed;
    }
  }
}

// ========================================================================
// Host-only: workspace query
// ========================================================================

TEST(HornerBufferSizeTests, OneBlockAlignedAndDegreeIndependent) {
  // One n-by-n block of T, rounded up to 256 bytes, independent of degree.
  for (const int n : {1, 2, 5, 7, 16, 33}) {
    const std::size_t want_floor =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(double);
    const std::size_t bytes = horner_bufferSize<double>(n);
    EXPECT_GE(bytes, want_floor) << "n=" << n;
    EXPECT_EQ(bytes % 256, 0u) << "n=" << n << " must be 256-aligned";
    EXPECT_LT(bytes, want_floor + 256) << "n=" << n << " only one block, no slack";
  }
  // Separate float check: smaller element, same block count.
  EXPECT_GE(horner_bufferSize<float>(10), 10u * 10u * sizeof(float));
  // n < 1 is clamped to one element, never zero bytes.
  EXPECT_GE(horner_bufferSize<double>(0), sizeof(double));
}

// ========================================================================
// Host-only: argument-checking contract
// ========================================================================
//
// Every rejected argument is caught before horner() touches the handle or a
// device pointer, so a null handle and host dummy pointers are enough: the
// return must be WWRBLAS_STATUS_NOT_INITIALIZED without any device work.

TEST(HornerArgCheckTests, RejectsBadArgumentsBeforeTouchingTheDevice) {
  const int n = 4;
  const int degree = 3;
  std::vector<double> c(static_cast<std::size_t>(degree) + 1, 1.0);
  std::vector<double> a(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<double> p(static_cast<std::size_t>(n) * n, 0.0);
  const std::size_t wbytes = horner_bufferSize<double>(n);
  std::vector<double> work(wbytes / sizeof(double), 0.0);
  wwr::wwrblasHandle_t null_blas{}; // never dereferenced on the rejection path

  auto call = [&](int nn, int deg, const double *dc, const double *da, int lda, double *dp, int ldp,
                  void *dw, std::size_t wb) {
    return horner<double>(null_blas, nn, dc, deg, da, lda, dp, ldp, dw, wb);
  };
  const auto ok_lda = n;
  const auto ok_ldp = n;

  EXPECT_EQ(call(0, degree, c.data(), a.data(), ok_lda, p.data(), ok_ldp, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "n < 1";
  EXPECT_EQ(call(n, -1, c.data(), a.data(), ok_lda, p.data(), ok_ldp, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "degree < 0";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), n - 1, p.data(), ok_ldp, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "lda < n";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), ok_lda, p.data(), n - 1, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "ldp < n";
  EXPECT_EQ(call(n, degree, nullptr, a.data(), ok_lda, p.data(), ok_ldp, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "null coeffs";
  EXPECT_EQ(call(n, degree, c.data(), nullptr, ok_lda, p.data(), ok_ldp, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "null A";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), ok_lda, nullptr, ok_ldp, work.data(), wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "null P";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), ok_lda, p.data(), ok_ldp, nullptr, wbytes),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "degree > 0 with null workspace";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), ok_lda, p.data(), ok_ldp, work.data(), wbytes - 1),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "undersized workspace";
}

// ========================================================================
// Device: numerical agreement with the oracle
// ========================================================================

TEST(HornerOracleTests, DegreeZeroIsScaledIdentity) {
  // p(A) = c_0 I for any A, and the call accepts a null workspace (run_horner
  // passes one for degree 0). Checked for both precisions.
  expect_poly_matches<float>(5, 0, 1);
  expect_poly_matches<double>(6, 0, 2);
}

TEST(HornerOracleTests, DegreeOneClosedForm) {
  expect_poly_matches<float>(4, 1, 3);
  expect_poly_matches<double>(6, 1, 4);
}

TEST(HornerOracleTests, EvenDegreeFloat) {
  // Even degree: the recurrence starts in P (parity lands the last step there).
  expect_poly_matches<float>(4, 2, 10);
  expect_poly_matches<float>(6, 4, 11);
}
TEST(HornerOracleTests, EvenDegreeDouble) {
  expect_poly_matches<double>(5, 2, 12);
  expect_poly_matches<double>(6, 4, 13);
}

TEST(HornerOracleTests, OddDegreeFloat) {
  // Odd degree: the recurrence starts in the workspace -- the other ping-pong
  // parity, so the last step still lands in P with no extra copy.
  expect_poly_matches<float>(4, 3, 20);
  expect_poly_matches<float>(6, 5, 21);
}
TEST(HornerOracleTests, OddDegreeDouble) {
  expect_poly_matches<double>(5, 3, 22);
  expect_poly_matches<double>(6, 5, 23);
}

TEST(HornerOracleTests, SquareLargerDouble) {
  expect_poly_matches<double>(16, 4, 30);
  expect_poly_matches<double>(16, 5, 31);
}

TEST(HornerOracleTests, ZeroPolynomial) {
  // All coefficients zero -> p(A) = 0 exactly (the device multiplies and adds
  // exact zeros), so the comparison is exact.
  const int n = 6;
  const int degree = 4;
  std::mt19937 rng(40);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<double> a(static_cast<std::size_t>(n) * n);
  for (auto &x : a) {
    x = dist(rng);
  }
  std::vector<double> c(static_cast<std::size_t>(degree) + 1, 0.0);
  std::vector<double> got(static_cast<std::size_t>(n) * n, 7.0); // sentinel, must be overwritten
  ASSERT_EQ(run_horner<double>(shared_device(), n, c, degree, a, n, n, got),
            wwr::WWRBLAS_STATUS_SUCCESS);
  for (const double v : got) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST(HornerOracleTests, NilpotentSeriesTerminates) {
  // A strictly upper-triangular N is nilpotent: N^k = 0 for k >= n, so a
  // degree-(2n) polynomial has the same value as its degree-(n-1) truncation.
  // The oracle forms the (vanishing) high powers too, so this checks the device
  // and oracle agree that those terms contribute nothing.
  const int n = 5;
  const int degree = 9;
  std::vector<double> a(static_cast<std::size_t>(n) * n, 0.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) { // strictly above the diagonal
      a[static_cast<std::size_t>(j) * n + i] = 0.3 * static_cast<double>(i + j + 1);
    }
  }
  std::vector<double> c(static_cast<std::size_t>(degree) + 1);
  std::mt19937 rng(50);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto &x : c) {
    x = dist(rng);
  }

  const auto oracle = poly_reference(a, n, c, degree);
  std::vector<double> got(static_cast<std::size_t>(n) * n, 0.0);
  ASSERT_EQ(run_horner<double>(shared_device(), n, c, degree, a, n, n, got),
            wwr::WWRBLAS_STATUS_SUCCESS);
  const double tol = poly_tol(a, c, degree, n);
  for (std::size_t idx = 0; idx < got.size(); ++idx) {
    EXPECT_NEAR(got[idx], oracle[idx], tol) << "idx=" << idx;
  }
}

// ========================================================================
// Device: leading-dimension padding
// ========================================================================

TEST(HornerPaddingTests, PaddingRowsBetweenNAndLdpUntouched) {
  // P may be a view into a taller allocation: only the n-by-n block is written,
  // the rows between n and ldp keep their prior contents. Seed the whole buffer
  // with a sentinel and check the padding rows survive, the block is correct.
  const int n = 5;
  const int ldp = n + 3;
  const int lda = n;
  for (const int degree : {2, 3}) { // one of each ping-pong parity
    std::mt19937 rng(60u + static_cast<unsigned>(degree));
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> a(static_cast<std::size_t>(lda) * n);
    for (auto &x : a) {
      x = dist(rng);
    }
    std::vector<double> c(static_cast<std::size_t>(degree) + 1);
    for (auto &x : c) {
      x = dist(rng) * 2.0;
    }

    const double sentinel = -123.5;
    std::vector<double> buf(static_cast<std::size_t>(ldp) * n, sentinel);
    ASSERT_EQ(run_horner<double>(shared_device(), n, c, degree, a, lda, ldp, buf),
              wwr::WWRBLAS_STATUS_SUCCESS);

    const auto oracle = poly_reference(a, n, c, degree);
    const double tol = poly_tol(a, c, degree, n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(buf[static_cast<std::size_t>(j) * ldp + i],
                    oracle[static_cast<std::size_t>(j) * n + i], tol)
            << "block (" << i << "," << j << ") degree=" << degree;
      }
      for (int i = n; i < ldp; ++i) {
        EXPECT_DOUBLE_EQ(buf[static_cast<std::size_t>(j) * ldp + i], sentinel)
            << "padding row " << i << " col " << j << " degree=" << degree;
      }
    }
  }
}

} // namespace
} // namespace calaman
