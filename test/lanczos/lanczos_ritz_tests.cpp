// Reference suite for calaman.lanczos's :ritz partition. Synthetic projected
// matrices T are loaded straight into the slices, with an orthonormal basis V
// (LAPACKE ?geqrf + ?orgqr) and a chosen beta_m; the oracle is LAPACKE_?syevd:
//
//   * extract: theta to the shared tolerance, |S(ncv-1, i)| to it over the
//     eigengap (the sign of an eigenvector is free), beta_m and ||T||_2;
//   * select: the positions per LanczosWhich, the residual estimates
//     |beta_m s_{m,i}| and the convergence flags against the reference;
//   * compact: the chosen theta / S columns land, bit for bit, in the leading
//     slots -- including runs long enough to need chunked copies -- and
//     lanczos_arrowhead reads them back as the restart layout;
//   * vectors: X = V S_k against V Z_k (sign-aligned) and X^T X = I.
//
// T is a symmetric tridiagonal, an arrowhead + tridiagonal tail (post-restart
// shape), or a decoupled diagonal block + tridiagonal tail (exactly converged
// pairs). REQUIRES_GPU; float and double.

#include "lanczos/lanczos_bridge.h"

#include <gtest/gtest.h>

#include <lapacke.h>

#include <cstddef>
#include <cstdint>

import std;

import wwr.blas;
import wwr.solver;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.lanczos;
import calaman.error_handling;
import calaman.test.shared.abort_policy;
import calaman.test.shared.device_handle;
import calaman.test.shared.tolerance;
import calaman.test.utils.shared_device;

namespace calaman {
namespace {

using wwr::extension::DeviceBufferWrapper;

using test::AbortPolicy;
using test::DeviceHandle;
using test::factorization_tol;
using test::frobenius_norm;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

// ── LAPACKE overloads ────────────────────────────────────────────────────────

lapack_int ref_syevd(lapack_int n, float *a, float *w) {
  return LAPACKE_ssyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}
lapack_int ref_syevd(lapack_int n, double *a, double *w) {
  return LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'L', n, a, n, w);
}
lapack_int ref_geqrf(lapack_int m, lapack_int n, float *a, float *tau) {
  return LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, a, m, tau);
}
lapack_int ref_geqrf(lapack_int m, lapack_int n, double *a, double *tau) {
  return LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a, m, tau);
}
lapack_int ref_orgqr(lapack_int m, lapack_int n, float *a, const float *tau) {
  return LAPACKE_sorgqr(LAPACK_COL_MAJOR, m, n, n, a, m, tau);
}
lapack_int ref_orgqr(lapack_int m, lapack_int n, double *a, const double *tau) {
  return LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, n, n, a, m, tau);
}

// ── host matrices ────────────────────────────────────────────────────────────

template<typename T>
struct Dense {
  int rows;
  int cols;
  std::vector<T> a;
  Dense(int r, int c) : rows(r), cols(c), a(static_cast<std::size_t>(r) * c, T(0)) {}
  T &operator()(int i, int j) {
    return a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * rows];
  }
  T operator()(int i, int j) const {
    return a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * rows];
  }
};

/// Symmetric tridiagonal tail on rows/cols first..ncv-1: diagonal in [lo, hi],
/// off-diagonal in [0.3, 1] (unreduced, so the eigenvalues are distinct).
template<typename T>
void fill_tridiagonal(Dense<T> &t, int first, std::uint32_t seed, T lo, T hi) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> diag(lo, hi);
  std::uniform_real_distribution<T> off(T(0.3), T(1));
  for (int i = first; i < t.rows; ++i) {
    t(i, i) = diag(gen);
    if (i + 1 < t.rows) {
      t(i + 1, i) = t(i, i + 1) = off(gen);
    }
  }
}

template<typename T>
Dense<T> tridiagonal(int ncv, std::uint32_t seed) {
  Dense<T> t(ncv, ncv);
  fill_tridiagonal(t, 0, seed, T(-2), T(2));
  return t;
}

/// The post-restart shape: diag theta_0..theta_{k-1}, couplings in row/col k,
/// tridiagonal from k on.
template<typename T>
Dense<T> arrowhead(int ncv, int k, std::uint32_t seed) {
  Dense<T> t(ncv, ncv);
  fill_tridiagonal(t, k, seed, T(-1), T(1));
  std::mt19937 gen(seed + 1);
  std::uniform_real_distribution<T> coupling(T(0.2), T(0.8));
  for (int i = 0; i < k; ++i) {
    t(i, i) = T(-3) + T(1.5) * static_cast<T>(i);
    t(k, i) = t(i, k) = coupling(gen);
  }
  return t;
}

/// A decoupled diagonal block -10, -9, ... on 0..k-1 over a tridiagonal tail in
/// [0, 2]: the block's eigenvectors have no last-row component.
template<typename T>
Dense<T> decoupled(int ncv, int k, std::uint32_t seed) {
  Dense<T> t(ncv, ncv);
  fill_tridiagonal(t, k, seed, T(0), T(2));
  for (int i = 0; i < k; ++i) {
    t(i, i) = T(-10) + static_cast<T>(i);
  }
  return t;
}

template<typename T>
Dense<T> orthonormal(int n, int m, std::uint32_t seed) {
  Dense<T> q(n, m);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  for (T &x : q.a) {
    x = dist(gen);
  }
  std::vector<T> tau(static_cast<std::size_t>(m));
  EXPECT_EQ(ref_geqrf(n, m, q.a.data(), tau.data()), 0);
  EXPECT_EQ(ref_orgqr(n, m, q.a.data(), tau.data()), 0);
  return q;
}

// ── device rig ───────────────────────────────────────────────────────────────

struct Handles {
  wwr::wwrblasHandle_t blas{};
  wwr::wwrsolverDnHandle_t solver{};
  explicit Handles(wwr::wwrStream_t stream) {
    EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrblasSetStream(blas, stream), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrsolverDnCreate(&solver), wwr::WWRSOLVER_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrsolverDnSetStream(solver, stream), wwr::WWRSOLVER_STATUS_SUCCESS);
  }
  ~Handles() {
    wwr::wwrblasDestroy(blas);
    wwr::wwrsolverDnDestroy(solver);
  }
  Handles(const Handles &) = delete;
  Handles &operator=(const Handles &) = delete;
};

template<typename T>
struct Rig {
  std::shared_ptr<DeviceHandle> handle = shared_device();
  wwr::wwrStream_t stream = handle->stream().get();
  Handles h{stream};
  int n;
  int ncv;
  std::size_t lwork = 0;
  std::unique_ptr<DeviceBuffer<std::byte>> work;
  LanczosSlices<T> s;

  Rig(int n_, int nev, int ncv_) : n(n_), ncv(ncv_) {
    EXPECT_TRUE(lanczos_bufferSize<T>(h.solver, n, nev, ncv, &lwork).ok());
    work = std::make_unique<DeviceBuffer<std::byte>>(lwork, handle);
    EXPECT_TRUE(make_lanczos_slices<T>(h.solver, n, nev, ncv, work->data(), &s, nullptr).ok());
  }

  template<typename U>
  void upload(U *dst, const std::vector<U> &src) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(dst, src.data(), sizeof(U) * src.size(),
                                  wwr::wwrMemcpyHostToDevice, stream),
              wwr::wwrSuccess);
    ASSERT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
  }
  template<typename U>
  std::vector<U> download(const U *src, std::size_t count) {
    std::vector<U> out(count);
    EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), src, sizeof(U) * count, wwr::wwrMemcpyDeviceToHost,
                                  stream),
              wwr::wwrSuccess);
    EXPECT_EQ(wwr::wwrStreamSynchronize(stream), wwr::wwrSuccess);
    return out;
  }

  /// T into s.t, V into the first ncv columns of s.v, beta_m into s.beta[ncv-1].
  void load(const Dense<T> &t, const Dense<T> &v, T beta_m) {
    upload(s.t, t.a);
    upload(s.v, v.a);
    std::vector<T> beta(static_cast<std::size_t>(ncv), T(0.5));
    beta.back() = beta_m;
    upload(s.beta, beta);
    device::lanczos_status_reset(stream, s.status);
  }
};

// ── the oracle ───────────────────────────────────────────────────────────────

template<typename T>
struct Reference {
  std::vector<T> w; ///< eigenvalues, ascending
  Dense<T> z;       ///< eigenvectors
  std::vector<T> gap;
  explicit Reference(const Dense<T> &t) : w(static_cast<std::size_t>(t.rows)), z(t) {
    EXPECT_EQ(ref_syevd(t.rows, z.a.data(), w.data()), 0);
    gap.assign(w.size(), std::numeric_limits<T>::infinity());
    for (std::size_t i = 0; i < w.size(); ++i) {
      for (std::size_t j = 0; j < w.size(); ++j) {
        if (i != j) {
          gap[i] = std::min(gap[i], std::abs(w[i] - w[j]));
        }
      }
    }
  }
};

struct Case {
  LanczosWhich which;
  int count;
};

const char *name(LanczosWhich which) {
  switch (which) {
  case LanczosWhich::smallest:
    return "smallest";
  case LanczosWhich::largest:
    return "largest";
  case LanczosWhich::both_ends:
    return "both_ends";
  }
  return "?";
}

/// The whole :ritz pipeline on one loaded T, checked stage by stage.
template<typename T>
RitzSelection<T> run_case(const Dense<T> &t, const Case c, int n, T beta_m, T tolerance,
                                 std::uint32_t seed) {
  const int ncv = t.rows;
  SCOPED_TRACE(::testing::Message() << "which=" << name(c.which) << " count=" << c.count
                                    << " ncv=" << ncv << " sizeof(T)=" << sizeof(T));
  Rig<T> rig(n, 1, ncv);
  const Dense<T> v = orthonormal<T>(n, ncv, seed);
  rig.load(t, v, beta_m);
  const Reference<T> ref(t);
  const T tol = factorization_tol<T>(frobenius_norm(t.a), ncv, ncv);
  const auto vec_tol = [&](std::size_t i) { return tol / std::min(T(1), ref.gap[i]); };
  const auto m = static_cast<std::size_t>(ncv);

  // extract
  LanczosRitz<T> ritz;
  EXPECT_TRUE(lanczos_ritz_extract<T>(rig.h.solver, rig.stream, ncv, rig.s, &ritz).ok());
  EXPECT_EQ(ritz.theta.size(), m);
  EXPECT_EQ(ritz.beta_m, beta_m);
  EXPECT_FALSE(ritz.breakdown);
  EXPECT_EQ(ritz.breakdown_step, -1);
  EXPECT_NEAR(ritz.t_norm, std::max(std::abs(ref.w.front()), std::abs(ref.w.back())), tol);
  for (std::size_t i = 0; i < m; ++i) {
    EXPECT_NEAR(ritz.theta[i], ref.w[i], tol) << i;
    EXPECT_NEAR(std::abs(ritz.s_last_row[i]), std::abs(ref.z(ncv - 1, static_cast<int>(i))),
                vec_tol(i))
        << i;
  }
  // s.t survives (syevd ran on the copy in s.s).
  EXPECT_EQ(rig.download(rig.s.t, m * m), t.a);

  // select
  const RitzSelection<T> sel = lanczos_ritz_select(ritz, c.which, c.count, tolerance);
  EXPECT_EQ(sel.index, ritz_select(c.which, ncv, c.count));
  EXPECT_EQ(sel.index.size(), static_cast<std::size_t>(c.count));
  int converged = 0;
  for (std::size_t j = 0; j < sel.index.size(); ++j) {
    const auto i = static_cast<std::size_t>(sel.index[j]);
    EXPECT_EQ(sel.values[j], ritz.theta[i]);
    EXPECT_EQ(sel.residuals[j], std::abs(beta_m * ritz.s_last_row[i]));
    const T ref_res = std::abs(beta_m * ref.z(ncv - 1, static_cast<int>(i)));
    EXPECT_NEAR(sel.residuals[j], ref_res, std::abs(beta_m) * vec_tol(i)) << j;
    const T threshold = tolerance * std::max(std::abs(ref.w[i]), ritz.t_norm);
    if (std::abs(ref_res - threshold) > T(1e-2) * threshold) { // skip a borderline call
      EXPECT_EQ(sel.converged[j], ref_res <= threshold) << j;
    }
    converged += sel.converged[j] ? 1 : 0;
  }
  EXPECT_EQ(sel.converged_count, converged);

  // compact: the chosen pairs, copied bit for bit into the leading slots.
  const std::vector<T> theta_before = rig.download(rig.s.theta, m);
  const std::vector<T> s_before = rig.download(rig.s.s, m * m);
  EXPECT_TRUE(lanczos_ritz_compact<T>(rig.stream, ncv, sel.index, rig.s).ok());
  const std::vector<T> theta_after = rig.download(rig.s.theta, m);
  const std::vector<T> s_after = rig.download(rig.s.s, m * m);
  for (std::size_t j = 0; j < sel.index.size(); ++j) {
    const auto i = static_cast<std::size_t>(sel.index[j]);
    EXPECT_EQ(theta_after[j], theta_before[i]) << j;
    for (std::size_t r = 0; r < m; ++r) {
      EXPECT_EQ(s_after[r + j * m], s_before[r + i * m]) << r << "," << j;
    }
  }

  // vectors: X = V S_k, against V Z_k with each column's sign aligned.
  const auto nz = static_cast<std::size_t>(n);
  const auto kz = static_cast<std::size_t>(c.count);
  DeviceBuffer<T> d_x(nz * kz, rig.handle);
  EXPECT_TRUE(lanczos_ritz_vectors<T>(rig.h.blas, n, ncv, c.count, rig.s, d_x.data(), n).ok());
  const std::vector<T> x = rig.download(d_x.data(), nz * kz);
  for (std::size_t j = 0; j < kz; ++j) {
    const int i = sel.index[j];
    std::vector<T> want(nz, T(0));
    for (std::size_t r = 0; r < nz; ++r) {
      for (int q = 0; q < ncv; ++q) {
        want[r] += v(static_cast<int>(r), q) * ref.z(q, i);
      }
    }
    T dot = T(0);
    for (std::size_t r = 0; r < nz; ++r) {
      dot += want[r] * x[r + j * nz];
    }
    const T sign = dot < T(0) ? T(-1) : T(1);
    for (std::size_t r = 0; r < nz; ++r) {
      EXPECT_NEAR(x[r + j * nz], sign * want[r], vec_tol(static_cast<std::size_t>(i)))
          << r << "," << j;
    }
    for (std::size_t l = 0; l <= j; ++l) {
      T g = T(0);
      for (std::size_t r = 0; r < nz; ++r) {
        g += x[r + j * nz] * x[r + l * nz];
      }
      EXPECT_NEAR(g, l == j ? T(1) : T(0), tol) << j << "," << l;
    }
  }

  // The compacted layout is what lanczos_arrowhead reads: theta on the
  // diagonal, beta_m * S(ncv-1, j) in row/column count.
  if (c.count < ncv) {
    device::lanczos_arrowhead<T>(rig.stream, ncv, c.count, rig.s.theta, rig.s.s + (m - 1), ncv,
                                 rig.s.beta + (m - 1), rig.s.t);
    const std::vector<T> arrow = rig.download(rig.s.t, m * m);
    for (std::size_t j = 0; j < kz; ++j) {
      const auto i = static_cast<std::size_t>(sel.index[j]);
      EXPECT_EQ(arrow[j + j * m], ritz.theta[i]) << j;
      const T coupling = beta_m * ritz.s_last_row[i];
      const T eps_tol = T(2) * std::numeric_limits<T>::epsilon() * std::abs(coupling);
      EXPECT_NEAR(arrow[kz + j * m], coupling, eps_tol) << j;
      EXPECT_NEAR(arrow[j + kz * m], coupling, eps_tol) << j;
    }
  }
  return sel;
}

constexpr Case kCases[] = {
    {LanczosWhich::smallest, 4},  {LanczosWhich::largest, 4},  {LanczosWhich::both_ends, 4},
    {LanczosWhich::both_ends, 5}, {LanczosWhich::largest, 9},  {LanczosWhich::both_ends, 9},
    {LanczosWhich::smallest, 12},
};

constexpr int kN = 40;
constexpr int kNcv = 12;

template<typename T>
void check_tridiagonal() {
  const Dense<T> t = tridiagonal<T>(kNcv, 11);
  for (const Case c : kCases) {
    run_case<T>(t, c, kN, T(0.37), T(1e-3), 21);
  }
}

template<typename T>
void check_arrowhead() {
  const Dense<T> t = arrowhead<T>(kNcv, 5, 13);
  for (const Case c : kCases) {
    run_case<T>(t, c, kN, T(-0.61), T(1e-3), 23);
  }
}

/// The decoupled block's pairs (the 4 smallest) have residual estimate ~0 and
/// converge; the tail's, against a tight tolerance, do not.
template<typename T>
void check_decoupled() {
  const Dense<T> t = decoupled<T>(kNcv, 4, 17);
  const T tolerance = T(1e-3);

  const auto low = run_case<T>(t, {LanczosWhich::smallest, 4}, kN, T(0.8), tolerance, 25);
  EXPECT_TRUE(low.all_converged());

  const auto ends = run_case<T>(t, {LanczosWhich::both_ends, 4}, kN, T(0.8), tolerance, 25);
  EXPECT_TRUE(ends.converged[0]);
  EXPECT_TRUE(ends.converged[1]);
  EXPECT_FALSE(ends.all_converged());

  // beta_m = 0: an invariant subspace, every pair converged.
  const auto all = run_case<T>(t, {LanczosWhich::largest, 4}, kN, T(0), tolerance, 25);
  EXPECT_TRUE(all.all_converged());
  for (const T r : all.residuals) {
    EXPECT_EQ(r, T(0));
  }
}

TEST(LanczosRitzReferenceTests, TridiagonalFloat) { check_tridiagonal<float>(); }
TEST(LanczosRitzReferenceTests, TridiagonalDouble) { check_tridiagonal<double>(); }
TEST(LanczosRitzReferenceTests, ArrowheadFloat) { check_arrowhead<float>(); }
TEST(LanczosRitzReferenceTests, ArrowheadDouble) { check_arrowhead<double>(); }
TEST(LanczosRitzReferenceTests, DecoupledConvergenceFloat) { check_decoupled<float>(); }
TEST(LanczosRitzReferenceTests, DecoupledConvergenceDouble) { check_decoupled<double>(); }

} // namespace
} // namespace calaman
