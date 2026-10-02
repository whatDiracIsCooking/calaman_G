// Oracle test for calaman.paterson_stockmeyer -- matrix polynomial evaluation
// p(A) = sum_{k=0}^{degree} c_k A^k by the Paterson-Stockmeyer scheme. Like
// calaman.horner, this is not a LAPACK routine, so there is no LAPACKE binding;
// the oracle is an INDEPENDENT host accumulation of the power series -- form
// each power A^k by a reference CBLAS gemm (cblas_?gemm) and sum c_k A^k --
// which is a different algorithm from the blocked scheme under test, so
// agreement is real evidence rather than the same arithmetic computed twice. A
// second cross-check runs calaman.horner on the same input: the two schemes must
// agree to rounding (PatersonStockmeyerOracleTests.MatchesHorner).
//
// The numerical suites (PatersonStockmeyerOracleTests, PatersonStockmeyerPadding
// Tests) stage A, the coefficients and P on the device and run the gemm + block
// kernels, so they are REQUIRES_GPU (labeled `gpu`, excluded by `ctest -LE gpu`).
// The block-size/plan logic (PatersonStockmeyerPlanTests), the workspace query
// (PatersonStockmeyerBufferSizeTests) and the argument-checking contract
// (PatersonStockmeyerArgCheckTests) are pure host calls -- the plan helpers are
// constexpr and the argument validation returns before paterson_stockmeyer()
// ever reads the handle or a device pointer, so those cases need no card.
//
// Built only when calaman::lapack_reference exists; its CMakeLists.txt returns
// early otherwise.

#include <gtest/gtest.h>

#include <cblas.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.paterson_stockmeyer;
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
// Accumulates the powers explicitly, so neither the summation order nor the
// blocking matches the device scheme under test.
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
// magnitude the scheme builds up (||A||_F overestimates the spectral norm),
// scaled by eps, the inner dimension n and the big-O factor the other suites
// use. Paterson-Stockmeyer forms the explicit powers A^k directly, so it is
// slightly less accurate than Horner in general -- the same bound still covers
// it for the modest-norm inputs here.
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

// Run paterson_stockmeyer() on the device: stage coeffs/A/P (P seeded with
// @p p_inout so a padded leading dimension can be checked afterward), allocate
// the workspace the query asks for (null when the plan needs no blocks, which
// covers the degree-0 and degree-1 cases), read P back into @p p_inout. Returns
// the call's status. @p s_requested is forwarded to both the size query and the
// evaluation, as the contract requires.
template<typename T>
Status run_ps(std::shared_ptr<DeviceHandle> handle, int n, const std::vector<T> &c, int degree,
              const std::vector<T> &a, int lda, int ldp, std::vector<T> &p_inout,
              int s_requested = 0) {
  wwr::wwrblasHandle_t blas{};
  EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);

  auto d_c = to_device(handle, c);
  auto d_a = to_device(handle, a);
  auto d_p = to_device(handle, p_inout);

  const std::size_t wbytes = paterson_stockmeyer_bufferSize<T>(n, degree, s_requested);
  const std::size_t wcount = (wbytes > 0) ? wbytes / sizeof(T) : 1;
  DeviceBuffer<T> d_work(wcount, handle);
  void *work_ptr = (wbytes > 0) ? static_cast<void *>(d_work.data()) : nullptr;

  const auto status = paterson_stockmeyer<T>(blas, n, d_c.data(), degree, d_a.data(), lda,
                                             d_p.data(), ldp, work_ptr, wbytes, s_requested);

  p_inout = from_device(handle, d_p, static_cast<std::size_t>(ldp) * n);
  wwr::wwrblasDestroy(blas);
  return status;
}

// Run calaman.horner() on the device -- the cross-check oracle in MatchesHorner.
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
  const std::size_t wcount = (wbytes > 0) ? wbytes / sizeof(T) : 1;
  DeviceBuffer<T> d_work(wcount, handle);
  void *work_ptr = (wbytes > 0) ? static_cast<void *>(d_work.data()) : nullptr;

  const auto status = horner<T>(blas, n, d_c.data(), degree, d_a.data(), lda, d_p.data(), ldp,
                                work_ptr, wbytes);

  p_inout = from_device(handle, d_p, static_cast<std::size_t>(ldp) * n);
  wwr::wwrblasDestroy(blas);
  return status;
}

// Square n-by-n polynomial agreement: random modest-norm A and coefficients,
// the device scheme at block size @p s_requested vs the explicit power-series
// oracle, compared elementwise.
template<typename T>
void expect_poly_matches(int n, int degree, unsigned seed, int s_requested = 0) {
  const int lda = n;
  const int ldp = n;
  std::mt19937 rng(seed);
  // Entries in (-0.5, 0.5) keep ||A|| below 1 for the small n here, so the high
  // powers the oracle and the bank both form stay well inside range for float.
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
  ASSERT_EQ(run_ps<T>(shared_device(), n, c, degree, a, lda, ldp, got, s_requested),
            wwr::WWRBLAS_STATUS_SUCCESS);

  const T tol = poly_tol(a, c, degree, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(got[static_cast<std::size_t>(j) * ldp + i],
                  oracle[static_cast<std::size_t>(j) * n + i], tol)
          << "(" << i << "," << j << ") n=" << n << " degree=" << degree << " s=" << s_requested
          << " seed=" << seed;
    }
  }
}

// ========================================================================
// Host-only: the block-size / plan logic
// ========================================================================

TEST(PatersonStockmeyerPlanTests, CostTableMatchesReadme) {
  // The product counts the README advertises against Horner.
  EXPECT_EQ(paterson_stockmeyer_plan(4).num_gemms, 3);
  EXPECT_EQ(paterson_stockmeyer_plan(8).num_gemms, 4);
  EXPECT_EQ(paterson_stockmeyer_plan(16).num_gemms, 7);
  EXPECT_EQ(paterson_stockmeyer_plan(30).num_gemms, 10);

  // The README's worked example: degree 12 -> s = 3, 6 products, 3 blocks.
  const auto p12 = paterson_stockmeyer_plan(12);
  EXPECT_EQ(p12.s, 3);
  EXPECT_EQ(p12.num_gemms, 6);
  EXPECT_EQ(p12.num_blocks, 3);
}

TEST(PatersonStockmeyerPlanTests, EdgeCasesDegreeZeroOneTwo) {
  // Degree 0: nothing to evaluate -- no product, no workspace.
  const auto p0 = paterson_stockmeyer_plan(0);
  EXPECT_EQ(p0.num_gemms, 0);
  EXPECT_EQ(p0.num_blocks, 0);

  // Degree 1: c_0 I + c_1 A is a single fused block, so s = 2 and NO product.
  const auto p1 = paterson_stockmeyer_plan(1);
  EXPECT_EQ(p1.s, 2);
  EXPECT_EQ(p1.num_gemms, 0);
  EXPECT_EQ(p1.num_blocks, 0);

  // Degree 2: the s > d regime wins -- form A^2, one fused block, ONE product,
  // against Horner's two.
  const auto p2 = paterson_stockmeyer_plan(2);
  EXPECT_EQ(p2.s, 3);
  EXPECT_EQ(p2.num_gemms, 1);
}

TEST(PatersonStockmeyerPlanTests, BlockSizeIsExactArgminThroughDegree32) {
  // For every degree up to 32 the chosen block size is a true argmin of the
  // product count over s in [1, d+1] -- the exhaustive check the README promises
  // -- and the scheme never costs more products than Horner's d.
  for (int d = 1; d <= 32; ++d) {
    const int best = paterson_stockmeyer_plan(d).num_gemms;
    EXPECT_LE(best, d) << "degree " << d << " must not lose to Horner";
    for (int s = 1; s <= d + 1; ++s) {
      EXPECT_LE(best, paterson_stockmeyer_cost(d, s))
          << "degree " << d << " not argmin at s=" << s;
    }
    // The chosen s actually achieves that minimum.
    EXPECT_EQ(best, paterson_stockmeyer_cost(d, paterson_stockmeyer_plan(d).s)) << "degree " << d;
  }
}

TEST(PatersonStockmeyerPlanTests, SRequestedOverrideClampAndDegenerate) {
  // s = 1 degenerates exactly to Horner: d products.
  EXPECT_EQ(paterson_stockmeyer_plan(10, 1).num_gemms, 10);
  // A request above d + 1 is clamped to d + 1 (no polynomial left to block).
  EXPECT_EQ(paterson_stockmeyer_plan(5, 999).s, 6);
  // A forced mid-range s is honoured.
  EXPECT_EQ(paterson_stockmeyer_plan(9, 4).s, 4);
}

// ========================================================================
// Host-only: workspace query
// ========================================================================

TEST(PatersonStockmeyerBufferSizeTests, HoldsPlanBlocksAligned) {
  // The buffer is exactly plan.num_blocks blocks, each an n-by-n T span rounded
  // up to 256 bytes -- the uniform stride the power bank and the accumulator
  // share, so every block lands aligned.
  for (const int n : {1, 2, 5, 7, 16}) {
    for (const int degree : {2, 4, 8, 16}) {
      const auto plan = paterson_stockmeyer_plan(degree);
      const std::size_t nn =
          static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(double);
      const std::size_t block = ((nn + 255) / 256) * 256;
      const std::size_t want = static_cast<std::size_t>(plan.num_blocks) * block;
      const std::size_t bytes = paterson_stockmeyer_bufferSize<double>(n, degree);
      EXPECT_EQ(bytes, want) << "n=" << n << " degree=" << degree;
      EXPECT_EQ(bytes % 256, 0u) << "n=" << n << " degree=" << degree;
    }
  }
  // A forced block size must size the SAME layout the evaluation will use.
  EXPECT_EQ(paterson_stockmeyer_bufferSize<float>(8, 10, 1),
            static_cast<std::size_t>(paterson_stockmeyer_plan(10, 1).num_blocks) *
                (((8u * 8u * sizeof(float)) + 255) / 256) * 256);
}

TEST(PatersonStockmeyerBufferSizeTests, DegreeZeroAndOneNeedNoWorkspace) {
  // Both evaluate as a single fused block with no product, so no scratch block.
  EXPECT_EQ(paterson_stockmeyer_bufferSize<double>(10, 0), 0u);
  EXPECT_EQ(paterson_stockmeyer_bufferSize<double>(10, 1), 0u);
  EXPECT_EQ(paterson_stockmeyer_bufferSize<float>(10, 1), 0u);
}

// ========================================================================
// Host-only: argument-checking contract
// ========================================================================
//
// Every rejected argument is caught before paterson_stockmeyer() touches the
// handle or a device pointer, so a null handle and host dummy pointers are
// enough: the return must be WWRBLAS_STATUS_NOT_INITIALIZED without device work.
// n=4, degree=4 gives a plan with blocks (s=2, 2 blocks), so the null/undersized
// workspace rejections actually fire.

TEST(PatersonStockmeyerArgCheckTests, RejectsBadArgumentsBeforeTouchingTheDevice) {
  const int n = 4;
  const int degree = 4;
  std::vector<double> c(static_cast<std::size_t>(degree) + 1, 1.0);
  std::vector<double> a(static_cast<std::size_t>(n) * n, 1.0);
  std::vector<double> p(static_cast<std::size_t>(n) * n, 0.0);
  const std::size_t wbytes = paterson_stockmeyer_bufferSize<double>(n, degree);
  ASSERT_GT(wbytes, 0u); // the fixture relies on this plan needing a workspace
  std::vector<double> work(wbytes / sizeof(double), 0.0);
  wwr::wwrblasHandle_t null_blas{}; // never dereferenced on the rejection path

  auto call = [&](int nn, int deg, const double *dc, const double *da, int lda, double *dp, int ldp,
                  void *dw, std::size_t wb, int s) {
    return paterson_stockmeyer<double>(null_blas, nn, dc, deg, da, lda, dp, ldp, dw, wb, s);
  };
  const auto L = n; // ok leading dimensions

  EXPECT_EQ(call(0, degree, c.data(), a.data(), L, p.data(), L, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "n < 1";
  EXPECT_EQ(call(n, -1, c.data(), a.data(), L, p.data(), L, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "degree < 0";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), n - 1, p.data(), L, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "lda < n";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), L, p.data(), n - 1, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "ldp < n";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), L, p.data(), L, work.data(), wbytes, -1),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "s_requested < 0";
  EXPECT_EQ(call(n, degree, nullptr, a.data(), L, p.data(), L, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "null coeffs";
  EXPECT_EQ(call(n, degree, c.data(), nullptr, L, p.data(), L, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "null A";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), L, nullptr, L, work.data(), wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "null P";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), L, p.data(), L, nullptr, wbytes, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "plan needs blocks but workspace is null";
  EXPECT_EQ(call(n, degree, c.data(), a.data(), L, p.data(), L, work.data(), wbytes - 1, 0),
            wwr::WWRBLAS_STATUS_NOT_INITIALIZED)
      << "undersized workspace";
}

// ========================================================================
// Device: numerical agreement with the oracle
// ========================================================================

TEST(PatersonStockmeyerOracleTests, DegreeZeroIsScaledIdentity) {
  // p(A) = c_0 I for any A; accepts a null workspace (run_ps passes one).
  expect_poly_matches<float>(5, 0, 1);
  expect_poly_matches<double>(6, 0, 2);
}

TEST(PatersonStockmeyerOracleTests, DegreeOneSingleFusedBlock) {
  // c_0 I + c_1 A with s = 2: one fused block, no product, no workspace.
  expect_poly_matches<float>(4, 1, 3);
  expect_poly_matches<double>(6, 1, 4);
}

TEST(PatersonStockmeyerOracleTests, DegreeTwoSingleProduct) {
  // The s > d winner: form A^2, one fused block.
  expect_poly_matches<float>(5, 2, 5);
  expect_poly_matches<double>(7, 2, 6);
}

TEST(PatersonStockmeyerOracleTests, EvenRemainderFloat) {
  // Degrees whose auto plan lands an even r (the ping-pong parity that starts
  // the outer Horner in P).
  expect_poly_matches<float>(4, 4, 10);
  expect_poly_matches<float>(6, 8, 11);
}
TEST(PatersonStockmeyerOracleTests, EvenRemainderDouble) {
  expect_poly_matches<double>(5, 4, 12);
  expect_poly_matches<double>(6, 8, 13);
}

TEST(PatersonStockmeyerOracleTests, OddRemainderFloat) {
  expect_poly_matches<float>(4, 6, 20);
  expect_poly_matches<float>(6, 10, 21);
}
TEST(PatersonStockmeyerOracleTests, OddRemainderDouble) {
  expect_poly_matches<double>(5, 6, 22);
  expect_poly_matches<double>(6, 10, 23);
}

TEST(PatersonStockmeyerOracleTests, SquareLargerDouble) {
  expect_poly_matches<double>(16, 12, 30);
  expect_poly_matches<double>(16, 16, 31);
}

TEST(PatersonStockmeyerOracleTests, AllBlockSizesAgree) {
  // The sharpest case: one polynomial, every block size from 1 (degenerate
  // Horner) to d+1 (fully explicit, no Horner step). All must agree with the
  // oracle -- exercising both ends and every blocking in between on one input.
  const int n = 5;
  const int degree = 7;
  std::mt19937 rng(77);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<double> a(static_cast<std::size_t>(n) * n);
  for (auto &x : a) {
    x = dist(rng);
  }
  std::vector<double> c(static_cast<std::size_t>(degree) + 1);
  for (auto &x : c) {
    x = dist(rng) * 2.0;
  }

  const auto oracle = poly_reference(a, n, c, degree);
  const double tol = poly_tol(a, c, degree, n);

  for (int s = 1; s <= degree + 1; ++s) {
    std::vector<double> got(static_cast<std::size_t>(n) * n, 0.0);
    ASSERT_EQ(run_ps<double>(shared_device(), n, c, degree, a, n, n, got, s),
              wwr::WWRBLAS_STATUS_SUCCESS)
        << "s=" << s;
    for (std::size_t idx = 0; idx < got.size(); ++idx) {
      EXPECT_NEAR(got[idx], oracle[idx], tol) << "s=" << s << " idx=" << idx;
    }
  }
}

TEST(PatersonStockmeyerOracleTests, MatchesHorner) {
  // The two schemes compute the same polynomial; cross-check against
  // calaman.horner directly, not just the shared power-series oracle.
  const int n = 8;
  for (const int degree : {3, 6, 9}) {
    std::mt19937 rng(90u + static_cast<unsigned>(degree));
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    std::vector<double> a(static_cast<std::size_t>(n) * n);
    for (auto &x : a) {
      x = dist(rng);
    }
    std::vector<double> c(static_cast<std::size_t>(degree) + 1);
    for (auto &x : c) {
      x = dist(rng) * 2.0;
    }

    std::vector<double> ps(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> hn(static_cast<std::size_t>(n) * n, 0.0);
    ASSERT_EQ(run_ps<double>(shared_device(), n, c, degree, a, n, n, ps),
              wwr::WWRBLAS_STATUS_SUCCESS);
    ASSERT_EQ(run_horner<double>(shared_device(), n, c, degree, a, n, n, hn),
              wwr::WWRBLAS_STATUS_SUCCESS);

    const double tol = poly_tol(a, c, degree, n);
    for (std::size_t idx = 0; idx < ps.size(); ++idx) {
      EXPECT_NEAR(ps[idx], hn[idx], tol) << "degree=" << degree << " idx=" << idx;
    }
  }
}

TEST(PatersonStockmeyerOracleTests, ZeroPolynomial) {
  // All coefficients zero -> p(A) = 0 exactly (the device multiplies and adds
  // exact zeros), so the comparison is exact.
  const int n = 6;
  const int degree = 8;
  std::mt19937 rng(40);
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  std::vector<double> a(static_cast<std::size_t>(n) * n);
  for (auto &x : a) {
    x = dist(rng);
  }
  std::vector<double> c(static_cast<std::size_t>(degree) + 1, 0.0);
  std::vector<double> got(static_cast<std::size_t>(n) * n, 7.0); // sentinel, must be overwritten
  ASSERT_EQ(run_ps<double>(shared_device(), n, c, degree, a, n, n, got),
            wwr::WWRBLAS_STATUS_SUCCESS);
  for (const double v : got) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST(PatersonStockmeyerOracleTests, NilpotentSeriesTerminates) {
  // A strictly upper-triangular N is nilpotent: N^k = 0 for k >= n, so a
  // degree-(2n) polynomial equals its degree-(n-1) truncation. The bank forms
  // the (vanishing) high powers too, so this checks device and oracle agree
  // those terms contribute nothing.
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
  ASSERT_EQ(run_ps<double>(shared_device(), n, c, degree, a, n, n, got),
            wwr::WWRBLAS_STATUS_SUCCESS);
  const double tol = poly_tol(a, c, degree, n);
  for (std::size_t idx = 0; idx < got.size(); ++idx) {
    EXPECT_NEAR(got[idx], oracle[idx], tol) << "idx=" << idx;
  }
}

// ========================================================================
// Device: leading-dimension padding
// ========================================================================

TEST(PatersonStockmeyerPaddingTests, PaddingRowsBetweenNAndLdpUntouched) {
  // P may be a view into a taller allocation: only the n-by-n block is written,
  // the rows between n and ldp keep their prior contents. Seed the whole buffer
  // with a sentinel and check the padding rows survive, the block is correct.
  // Degrees 7 and 8 give opposite outer-Horner parities.
  const int n = 5;
  const int ldp = n + 3;
  const int lda = n;
  for (const int degree : {7, 8}) {
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
    ASSERT_EQ(run_ps<double>(shared_device(), n, c, degree, a, lda, ldp, buf),
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
