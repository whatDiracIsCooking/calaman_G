// Oracle test for calaman.lansy -- the ?lansy norm of a real symmetric matrix
// read from one triangle. The oracle is the reference ?lansy (LAPACKE_?lansy)
// in the SAME precision on the host.
//
// Every case fills the triangle `uplo` does NOT name, and the lda > n padding
// rows, with garbage (NaN and the largest finite value): a single read of either
// would turn the result into NaN or inf and fail the comparison. All suites
// stage data on the device, so they are REQUIRES_GPU (labeled `gpu`). Built only
// when calaman::lapack_reference exists; see this directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lansy;
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
using test::factorization_tol;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

constexpr MatrixNorm kNorms[] = {MatrixNorm::max_abs, MatrixNorm::one, MatrixNorm::inf,
                                 MatrixNorm::frobenius};
constexpr Uplo kUplos[] = {Uplo::U, Uplo::L};

template<typename T>
DeviceBuffer<T> to_device(const std::vector<T> &host) {
  const auto handle = shared_device();
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
T scalar_from_device(const DeviceBuffer<T> &device) {
  const auto handle = shared_device();
  HostBuffer<T> host(1);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return host.data()[0];
}

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

char uplo_char(Uplo uplo) {
  return uplo == Uplo::U ? 'U' : 'L';
}

float ref_lansy(char norm, char uplo, std::size_t n, const float *a, std::size_t lda) {
  return LAPACKE_slansy(LAPACK_COL_MAJOR, norm, uplo, static_cast<lapack_int>(n), a,
                        static_cast<lapack_int>(lda));
}
double ref_lansy(char norm, char uplo, std::size_t n, const double *a, std::size_t lda) {
  return LAPACKE_dlansy(LAPACK_COL_MAJOR, norm, uplo, static_cast<lapack_int>(n), a,
                        static_cast<lapack_int>(lda));
}

bool stored(Uplo uplo, std::size_t i, std::size_t j) {
  return uplo == Uplo::U ? i <= j : i >= j;
}

// The stored triangle holds mixed-sign values over a few binades; every other
// slot of the lda-by-n array (the unstored triangle and the padding rows)
// alternates NaN and the largest finite T.
template<typename T>
std::vector<T> make_matrix(Uplo uplo, std::size_t n, std::size_t lda, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-4.0, 4.0);
  std::vector<T> a(lda * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < lda; ++i) {
      const bool in = i < n && stored(uplo, i, j);
      a[i + j * lda] = in                ? static_cast<T>(dist(gen))
                       : (i + j) % 2 == 0 ? std::numeric_limits<T>::quiet_NaN()
                                          : std::numeric_limits<T>::max();
    }
  }
  return a;
}

// The max norm compares and copies only, so it must match exactly; the sums
// differ from the reference in order (and, for 'F', by ?lassq's scaling), so
// they get the shared eps-relative tolerance, scaled by n terms per row/column.
template<typename T>
T norm_tol(MatrixNorm which, T ref, std::size_t n) {
  return which == MatrixNorm::max_abs ? T{0} : factorization_tol<T>(ref, n, n);
}

template<typename T>
void expect_matches_reference(MatrixNorm which, Uplo uplo, std::size_t n, std::size_t lda) {
  const auto a =
      make_matrix<T>(uplo, n, lda, static_cast<std::uint32_t>(31 * n + 7 * lda + 1));
  const T oracle = ref_lansy(norm_char(which), uplo_char(uplo), n, a.data(), lda);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";

  auto d_A = to_device(a);
  DeviceBuffer<T> d_result(1, shared_device());
  const Status s = lansy<T>(shared_device()->stream().get(), which, uplo, n, d_A.data(), lda,
                            d_result.data());
  ASSERT_TRUE(s.ok()) << "lansy returned " << s.name() << ": " << s.message();
  EXPECT_NEAR(scalar_from_device(d_result), oracle, norm_tol(which, oracle, n))
      << "norm=" << norm_char(which) << " uplo=" << uplo_char(uplo) << " n=" << n
      << " lda=" << lda;
}

template<typename T>
void run_sizes(MatrixNorm which) {
  // 1..3: the edge columns; 255..257: around the block's thread count, so a
  // column part straddles one stride; 1000: many strides, lda == n and lda > n.
  for (const Uplo uplo : kUplos) {
    for (const std::size_t n : {1u, 2u, 3u, 17u, 64u, 255u, 256u, 257u, 1000u}) {
      expect_matches_reference<T>(which, uplo, n, n);
      expect_matches_reference<T>(which, uplo, n, n + 5);
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?lansy, per norm and type
// ========================================================================

TEST(LansyOracleTests, MaxAbsFloat) {
  run_sizes<float>(MatrixNorm::max_abs);
}
TEST(LansyOracleTests, MaxAbsDouble) {
  run_sizes<double>(MatrixNorm::max_abs);
}
TEST(LansyOracleTests, OneNormFloat) {
  run_sizes<float>(MatrixNorm::one);
}
TEST(LansyOracleTests, OneNormDouble) {
  run_sizes<double>(MatrixNorm::one);
}
TEST(LansyOracleTests, InfNormFloat) {
  run_sizes<float>(MatrixNorm::inf);
}
TEST(LansyOracleTests, InfNormDouble) {
  run_sizes<double>(MatrixNorm::inf);
}
TEST(LansyOracleTests, FrobeniusFloat) {
  run_sizes<float>(MatrixNorm::frobenius);
}
TEST(LansyOracleTests, FrobeniusDouble) {
  run_sizes<double>(MatrixNorm::frobenius);
}

// n == 0 writes 0 (DLANSY returns 0) over a pre-seeded sentinel, for every norm
// and both triangles.
TEST(LansyOracleTests, EmptyWritesZero) {
  for (const MatrixNorm which : kNorms) {
    for (const Uplo uplo : kUplos) {
      auto d_result = to_device(std::vector<float>{-12345.0f});
      auto dummy = to_device(std::vector<float>{42.0f});
      ASSERT_TRUE(lansy<float>(shared_device()->stream().get(), which, uplo, 0, dummy.data(), 1,
                               d_result.data())
                      .ok());
      EXPECT_EQ(scalar_from_device(d_result), 0.0f)
          << "norm=" << norm_char(which) << " uplo=" << uplo_char(uplo);
    }
  }
}

// A NaN in the STORED triangle -- on the diagonal or off it -- propagates
// through every norm, as DLANSY's DISNAN guard does for the max folds.
TEST(LansyOracleTests, NaNInStoredTrianglePropagates) {
  const std::size_t n = 300;
  for (const Uplo uplo : kUplos) {
    const std::size_t off = uplo == Uplo::U ? 2 + 150 * n : 150 + 2 * n; // (2,150) / (150,2)
    for (const std::size_t pos : {std::size_t{77 + 77 * n}, off}) {
      auto a = make_matrix<double>(uplo, n, n, 9);
      a[pos] = std::numeric_limits<double>::quiet_NaN();
      auto d_A = to_device(a);
      DeviceBuffer<double> d_result(1, shared_device());
      for (const MatrixNorm which : kNorms) {
        ASSERT_TRUE(lansy<double>(shared_device()->stream().get(), which, uplo, n, d_A.data(), n,
                                  d_result.data())
                        .ok());
        EXPECT_TRUE(std::isnan(scalar_from_device(d_result)))
            << "norm=" << norm_char(which) << " uplo=" << uplo_char(uplo) << " pos=" << pos;
      }
    }
  }
}

} // namespace
} // namespace calaman
