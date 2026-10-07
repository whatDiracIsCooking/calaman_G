// Oracle test for calaman.langt -- the ?langt norm of a general tridiagonal
// (dl, d, du). The oracle is the reference ?langt (LAPACK_?langt, s/d/c/z) in
// the SAME precision on the host, over the same buffers (the wwr complex types
// are layout-compatible with lapack_complex_*).
//
// ?langt has no leading dimension; its analogue here is a sub-range: dl, d and
// du are views at an offset into larger buffers whose slots outside the view
// hold garbage (NaN and the largest finite value), so a read outside [0, n) of
// d or [0, n-1) of dl/du turns the result into NaN or inf and fails the
// comparison. All suites stage data on the device, so they are REQUIRES_GPU
// (labeled `gpu`). Built only when calaman::lapack_reference exists; see this
// directory's CMakeLists.txt.

#include <gtest/gtest.h>

#include <lapacke.h>

import std;

import wwr.runtime_api;
import wwr.complex;
import wwr.extension.memory_buffer;
import calaman.langt;
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

// Every LAPACK NORM char langt accepts, with the MatrixNorm it maps to ('O' is
// the 1-norm's synonym, 'E' the Frobenius norm's).
struct NormCase {
  char c;
  MatrixNorm which;
};
constexpr NormCase kNormCases[] = {{'M', MatrixNorm::max_abs},   {'1', MatrixNorm::one},
                                   {'O', MatrixNorm::one},       {'I', MatrixNorm::inf},
                                   {'F', MatrixNorm::frobenius}, {'E', MatrixNorm::frobenius}};

// Slots of garbage before and after each view: the "leading dimension" padding.
constexpr std::size_t kPad = 3;

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

// Per-type element construction and the same-precision oracle. LAPACKE has no
// ?langt wrapper, so call the Fortran routine through lapack.h's LAPACK_?langt
// (which lapacke.h includes; it appends the hidden string length). R is the
// norm's real type; `exact_max` is whether 'M' compares and copies only (real)
// or takes an inexact |z| hypot (complex).
template<typename T>
struct elem;

template<>
struct elem<float> {
  using R = float;
  static constexpr bool exact_max = true;
  static float make(double re, double) { return static_cast<float>(re); }
  static float ref(char norm, std::size_t n, const float *dl, const float *d, const float *du) {
    const lapack_int ni = static_cast<lapack_int>(n);
    return static_cast<float>(LAPACK_slangt(&norm, &ni, dl, d, du));
  }
};

template<>
struct elem<double> {
  using R = double;
  static constexpr bool exact_max = true;
  static double make(double re, double) { return re; }
  static double ref(char norm, std::size_t n, const double *dl, const double *d,
                    const double *du) {
    const lapack_int ni = static_cast<lapack_int>(n);
    return LAPACK_dlangt(&norm, &ni, dl, d, du);
  }
};

template<>
struct elem<wwr::wwrFloatComplex> {
  using R = float;
  using C = wwr::wwrFloatComplex;
  static constexpr bool exact_max = false;
  static C make(double re, double im) {
    return wwr::make_wwrFloatComplex(static_cast<float>(re), static_cast<float>(im));
  }
  static float ref(char norm, std::size_t n, const C *dl, const C *d, const C *du) {
    const lapack_int ni = static_cast<lapack_int>(n);
    return static_cast<float>(LAPACK_clangt(&norm, &ni,
                                            reinterpret_cast<const lapack_complex_float *>(dl),
                                            reinterpret_cast<const lapack_complex_float *>(d),
                                            reinterpret_cast<const lapack_complex_float *>(du)));
  }
};

template<>
struct elem<wwr::wwrDoubleComplex> {
  using R = double;
  using C = wwr::wwrDoubleComplex;
  static constexpr bool exact_max = false;
  static C make(double re, double im) { return wwr::make_wwrDoubleComplex(re, im); }
  static double ref(char norm, std::size_t n, const C *dl, const C *d, const C *du) {
    const lapack_int ni = static_cast<lapack_int>(n);
    return LAPACK_zlangt(&norm, &ni, reinterpret_cast<const lapack_complex_double *>(dl),
                         reinterpret_cast<const lapack_complex_double *>(d),
                         reinterpret_cast<const lapack_complex_double *>(du));
  }
};

// (dl, d, du) as views at offset kPad into buffers of n + 2*kPad slots; the
// views hold mixed-sign values over a few binades (complex: both parts), every
// other slot alternates NaN and the largest finite component.
template<typename T>
struct Tridiagonal {
  std::vector<T> dl;
  std::vector<T> d;
  std::vector<T> du;

  Tridiagonal(std::size_t n, std::uint32_t seed)
      : dl(n + 2 * kPad), d(n + 2 * kPad), du(n + 2 * kPad) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double big = std::numeric_limits<typename elem<T>::R>::max();
    const std::size_t ne = n == 0 ? 0 : n - 1;
    for (std::size_t k = 0; k < d.size(); ++k) {
      const bool in_d = k >= kPad && k < kPad + n;
      const bool in_e = k >= kPad && k < kPad + ne;
      const double junk = k % 2 == 0 ? nan : big;
      const auto fill = [&](bool in) {
        const double re = dist(gen);
        const double im = dist(gen);
        return in ? elem<T>::make(re, im) : elem<T>::make(junk, junk);
      };
      dl[k] = fill(in_e);
      d[k] = fill(in_d);
      du[k] = fill(in_e);
    }
  }
};

// The real max norm compares and copies only, so it must match exactly; the
// sums differ from the reference in order (and, for 'F', by ?lassq's scaling),
// and a complex |z| is inexact, so those get the shared eps-relative tolerance.
// 'F' sums 3n-2 terms, but as chunks of at most 17 indices then a pairwise
// tree, so its error stays a few eps (lanst's bound).
template<typename T>
typename elem<T>::R norm_tol(MatrixNorm which, typename elem<T>::R ref) {
  using R = typename elem<T>::R;
  if (which == MatrixNorm::max_abs && elem<T>::exact_max) {
    return R{0};
  }
  return factorization_tol<R>(ref, 4, 4);
}

template<typename T>
typename elem<T>::R run_device(MatrixNorm which, std::size_t n, const Tridiagonal<T> &t) {
  using R = typename elem<T>::R;
  auto d_dl = to_device(t.dl);
  auto d_d = to_device(t.d);
  auto d_du = to_device(t.du);
  DeviceBuffer<R> d_result(1, shared_device());
  const Status s = langt<T>(shared_device()->stream().get(), which, n, d_dl.data() + kPad,
                            d_d.data() + kPad, d_du.data() + kPad, d_result.data());
  EXPECT_TRUE(s.ok()) << "langt returned " << s.name() << ": " << s.message();
  return scalar_from_device(d_result);
}

template<typename T>
void expect_matches_reference(const NormCase nc, std::size_t n) {
  using R = typename elem<T>::R;
  const Tridiagonal<T> t(n, static_cast<std::uint32_t>(37 * n + 11));
  const R oracle =
      elem<T>::ref(nc.c, n, t.dl.data() + kPad, t.d.data() + kPad, t.du.data() + kPad);
  ASSERT_TRUE(std::isfinite(oracle)) << "the oracle read the garbage";
  EXPECT_NEAR(run_device(nc.which, n, t), oracle, norm_tol<T>(nc.which, oracle))
      << "norm=" << nc.c << " n=" << n;
}

template<typename T>
void run_sizes() {
  // 1..3: the edge rows (n = 1 reads d alone); 255..257: around the block's
  // thread count (chunk 1 vs 2); 1000/4099: multi-index chunks, ragged last one.
  for (const NormCase nc : kNormCases) {
    for (const std::size_t n : {1u, 2u, 3u, 7u, 64u, 255u, 256u, 257u, 1000u, 4099u}) {
      expect_matches_reference<T>(nc, n);
    }
  }
}

// ========================================================================
// Device: numerical agreement with the reference ?langt, per type
// ========================================================================

TEST(LangtOracleTests, Float) {
  run_sizes<float>();
}
TEST(LangtOracleTests, Double) {
  run_sizes<double>();
}
TEST(LangtOracleTests, ComplexFloat) {
  run_sizes<wwr::wwrFloatComplex>();
}
TEST(LangtOracleTests, ComplexDouble) {
  run_sizes<wwr::wwrDoubleComplex>();
}

// The 1-norm sums columns (du above, dl below) and the infinity-norm rows (dl
// left, du right): with dl large and du small they must differ, and each match
// the oracle. Column 0 = |d0| + |dl0| = 1 + 10; row 1 = |dl0| + |d1| + |du1| =
// 10 + 1 + 0.5.
TEST(LangtOracleTests, OneAndInfDiffer) {
  Tridiagonal<double> t(3, 5);
  const double dl[] = {10.0, 0.25};
  const double d[] = {1.0, 1.0, 1.0};
  const double du[] = {0.125, 0.5};
  for (std::size_t i = 0; i < 3; ++i) {
    t.d[kPad + i] = d[i];
  }
  for (std::size_t i = 0; i < 2; ++i) {
    t.dl[kPad + i] = dl[i];
    t.du[kPad + i] = du[i];
  }
  const double *pdl = t.dl.data() + kPad;
  const double *pd = t.d.data() + kPad;
  const double *pdu = t.du.data() + kPad;
  ASSERT_EQ(elem<double>::ref('1', 3, pdl, pd, pdu), 11.0);
  ASSERT_EQ(elem<double>::ref('I', 3, pdl, pd, pdu), 11.5);
  EXPECT_EQ(run_device(MatrixNorm::one, 3, t), 11.0);
  EXPECT_EQ(run_device(MatrixNorm::inf, 3, t), 11.5);
}

// n == 0 writes a real 0 for every norm, over a sentinel (DLANGT returns 0).
template<typename T>
void expect_empty_writes_zero() {
  using R = typename elem<T>::R;
  auto dummy = to_device(std::vector<T>{elem<T>::make(42.0, 1.0)});
  for (const NormCase nc : kNormCases) {
    auto d_result = to_device(std::vector<R>{R(-12345)});
    ASSERT_TRUE(langt<T>(shared_device()->stream().get(), nc.which, 0, dummy.data(), dummy.data(),
                         dummy.data(), d_result.data())
                    .ok());
    EXPECT_EQ(scalar_from_device(d_result), R{0}) << "norm=" << nc.c;
    EXPECT_EQ(elem<T>::ref(nc.c, 0, nullptr, nullptr, nullptr), R{0}) << "norm=" << nc.c;
  }
}

TEST(LangtOracleTests, EmptyWritesZero) {
  expect_empty_writes_zero<float>();
  expect_empty_writes_zero<double>();
  expect_empty_writes_zero<wwr::wwrFloatComplex>();
  expect_empty_writes_zero<wwr::wwrDoubleComplex>();
}

// A NaN in any of the three diagonals -- at a chunk boundary and near the end
// -- propagates through every norm, as DLANGT's DISNAN guard does for the max.
TEST(LangtOracleTests, NaNPropagates) {
  using C = wwr::wwrDoubleComplex;
  const std::size_t n = 600; // chunk 3: index 2 ends thread 0's chunk
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int which_diag = 0; which_diag < 3; ++which_diag) {
    for (const std::size_t pos : {std::size_t{2}, n - 2}) {
      Tridiagonal<C> t(n, 9);
      std::vector<C> &v = which_diag == 0 ? t.dl : which_diag == 1 ? t.d : t.du;
      v[kPad + pos] = elem<C>::make(1.0, nan);
      for (const NormCase nc : kNormCases) {
        EXPECT_TRUE(std::isnan(run_device(nc.which, n, t)))
            << "norm=" << nc.c << " diag=" << which_diag << " pos=" << pos;
      }
    }
  }
}

} // namespace
} // namespace calaman
