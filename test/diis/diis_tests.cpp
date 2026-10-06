// Oracle suite for calaman.diis -- Pulay DIIS extrapolation over a device ring.
//
// The oracle is a host DIIS in double over the same pushes in AGE order: the
// residual Gram of the last `cap` pairs, the bordered system solved by
// LAPACKE_dgesv, F* = sum_i c_i F_i. The device ring stores columns in rotated
// physical order, so agreement through and past wrap-around pins both the ring
// bookkeeping and the incremental Gram. Further cases cover the < 2-pair no-op,
// the singular fallback (including a newest column that is not the last
// physical one), reset, argument validation, and a behavioural check that DIIS
// accelerates a linear fixed-point iteration.
//
// REQUIRES_GPU (see CMakeLists.txt).

#include <gtest/gtest.h>
#include <lapacke.h>

import std;

import wwr.blas;
import wwr.runtime_api;
import wwr.extension.memory_buffer;
import calaman.diis;
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
using test::eps;
using test::kTolFactor;
using test::shared_device;

using DeviceAbort = AbortPolicy<wwr::wwrError_t>;
using HostAbort = AbortPolicy<wwr::extension::stdHostMemoryError_t>;

template<typename T>
using HostBuffer = HostBufferWrapper<T, HostAbort, HostAbort>;
template<typename T>
using DeviceBuffer = DeviceBufferWrapper<T, DeviceAbort, DeviceAbort, DeviceAbort, DeviceHandle>;

template<typename T>
void upload(const std::shared_ptr<DeviceHandle> &handle, DeviceBuffer<T> &device,
            const std::vector<T> &host) {
  HostBuffer<T> staging(host.size());
  std::copy(host.begin(), host.end(), staging.data());
  wwr::extension::copy(device, staging, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
}

template<typename T>
std::vector<T> download(const std::shared_ptr<DeviceHandle> &handle, const DeviceBuffer<T> &device,
                        std::size_t n) {
  HostBuffer<T> host(n);
  wwr::extension::copy(host, device, handle->stream().get());
  wwr::wwrStreamSynchronize(handle->stream().get());
  return std::vector<T>(host.data(), host.data() + n);
}

// One DIIS subspace on the device: the state, its workspace, staging buffers for
// the pushed pair, the output and the singular flag.
template<typename T>
struct DeviceDiis {
  std::shared_ptr<DeviceHandle> handle = shared_device();
  wwr::wwrblasHandle_t blas{};
  DiisState state;
  std::size_t work_bytes;
  DeviceBuffer<std::byte> work;
  DeviceBuffer<T> fock, residual, out;
  DeviceBuffer<int> singular;

  DeviceDiis(int len, int cap)
      : state(make_diis_state(len, cap)), work_bytes(diis_bufferSize<T>(state)),
        work(work_bytes, handle), fock(len, handle), residual(len, handle), out(len, handle),
        singular(1, handle) {
    EXPECT_EQ(wwr::wwrblasCreate(&blas), wwr::WWRBLAS_STATUS_SUCCESS);
    EXPECT_EQ(wwr::wwrblasSetStream(blas, handle->stream().get()), wwr::WWRBLAS_STATUS_SUCCESS);
    upload(handle, singular, std::vector<int>{-1});
  }
  ~DeviceDiis() { wwr::wwrblasDestroy(blas); }
  DeviceDiis(const DeviceDiis &) = delete;
  DeviceDiis &operator=(const DeviceDiis &) = delete;

  Status push(const std::vector<T> &f, const std::vector<T> &e) {
    upload(handle, fock, f);
    upload(handle, residual, e);
    const Status status =
        diis_push_and_extrapolate<T>(blas, handle->stream().get(), state, fock.data(),
                                     residual.data(), out.data(), singular.data(), work.data(),
                                     work_bytes);
    wwr::wwrStreamSynchronize(handle->stream().get());
    return status;
  }
  std::vector<T> output() { return download(handle, out, static_cast<std::size_t>(state.len)); }
  int singular_flag() { return download(handle, singular, 1)[0]; }
};

// The host oracle: the last `cap` pairs in age order, solved in double.
struct HostDiis {
  int cap;
  std::deque<std::vector<double>> f_hist, e_hist;

  template<typename T>
  void push(const std::vector<T> &f, const std::vector<T> &e) {
    f_hist.emplace_back(f.begin(), f.end());
    e_hist.emplace_back(e.begin(), e.end());
    if (static_cast<int>(f_hist.size()) > cap) {
      f_hist.pop_front();
      e_hist.pop_front();
    }
  }

  std::vector<double> extrapolate() const {
    const int m = static_cast<int>(e_hist.size());
    const int dim = m + 1;
    const std::size_t len = e_hist.front().size();
    std::vector<double> b(static_cast<std::size_t>(dim) * dim, -1.0); // column-major
    std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
    b.back() = 0.0;
    rhs.back() = -1.0;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        double dot = 0.0;
        for (std::size_t k = 0; k < len; ++k) {
          dot += e_hist[i][k] * e_hist[j][k];
        }
        b[static_cast<std::size_t>(j) * dim + i] = dot;
      }
    }
    std::vector<lapack_int> ipiv(static_cast<std::size_t>(dim));
    const lapack_int info =
        LAPACKE_dgesv(LAPACK_COL_MAJOR, dim, 1, b.data(), dim, ipiv.data(), rhs.data(), dim);
    EXPECT_EQ(info, 0) << "oracle bordered system singular";
    std::vector<double> out(len, 0.0);
    for (int i = 0; i < m; ++i) {
      for (std::size_t k = 0; k < len; ++k) {
        out[k] += rhs[static_cast<std::size_t>(i)] * f_hist[i][k];
      }
    }
    return out;
  }
};

template<typename T>
std::vector<T> random_vector(std::mt19937 &rng, int len, double scale = 1.0) {
  std::uniform_real_distribution<double> dist(-scale, scale);
  std::vector<T> v(static_cast<std::size_t>(len));
  for (auto &x : v) {
    x = static_cast<T>(dist(rng));
  }
  return v;
}

template<typename T>
void expect_near_vector(const std::vector<T> &got, const std::vector<double> &want, double tol,
                        const std::string &what) {
  ASSERT_EQ(got.size(), want.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - want[i]));
  }
  EXPECT_LE(worst, tol) << what;
}

// Push `pushes` random pairs into a cap-`cap` subspace and compare F* against the
// host oracle after every push that extrapolates.
template<typename T>
void expect_matches_oracle(int len, int cap, int pushes, unsigned seed) {
  std::mt19937 rng(seed);
  DeviceDiis<T> dev(len, cap);
  HostDiis host{cap, {}, {}};
  // Residuals are random O(0.1) vectors with len >> cap, so the Gram stays well
  // conditioned and F* is O(|F|); the bound scales with that and the depth.
  const double tol = static_cast<double>(kTolFactor<T> * eps<T>()) * cap * std::sqrt(len);
  for (int p = 0; p < pushes; ++p) {
    const auto f = random_vector<T>(rng, len);
    const auto e = random_vector<T>(rng, len, 0.1);
    ASSERT_TRUE(dev.push(f, e).ok()) << "push " << p;
    host.push(f, e);
    EXPECT_EQ(dev.state.size, std::min(p + 1, cap));
    if (dev.state.size >= 2) {
      expect_near_vector(dev.output(), host.extrapolate(), tol,
                         "push " + std::to_string(p) + " len " + std::to_string(len));
      EXPECT_EQ(dev.singular_flag(), 0) << "push " << p;
    }
  }
}

TEST(DiisOracleTests, BufferSizeCoversHistoriesAndRejectsBadShapes) {
  const auto bytes = diis_bufferSize<double>(make_diis_state(100, 8));
  EXPECT_GE(bytes, sizeof(double) * (2 * 100 * 8 + 8 * 8 + 8));
  EXPECT_EQ(diis_bufferSize<double>(make_diis_state(0, 8)), 0U);
  EXPECT_EQ(diis_bufferSize<double>(make_diis_state(10, kDiisMaxHistory + 1)), 0U);
  EXPECT_EQ(make_diis_state(10, 0).cap, 1);
}

TEST(DiisOracleTests, FirstPushLeavesOutputUntouched) {
  constexpr int kLen = 37;
  std::mt19937 rng(1);
  DeviceDiis<double> dev(kLen, 4);
  const std::vector<double> sentinel(kLen, 12345.0);
  upload(dev.handle, dev.out, sentinel);
  ASSERT_TRUE(dev.push(random_vector<double>(rng, kLen), random_vector<double>(rng, kLen)).ok());
  EXPECT_EQ(dev.state.size, 1);
  EXPECT_EQ(dev.output(), sentinel);
  EXPECT_EQ(dev.singular_flag(), 0);
}

TEST(DiisOracleTests, GrowthPhaseMatchesOracleDouble) { expect_matches_oracle<double>(64, 8, 8, 2); }
TEST(DiisOracleTests, GrowthPhaseMatchesOracleFloat) { expect_matches_oracle<float>(64, 8, 8, 3); }

TEST(DiisOracleTests, RingWrapMatchesOracleDouble) {
  expect_matches_oracle<double>(100, 4, 15, 4);
  expect_matches_oracle<double>(9, 1, 5, 5); // cap 1: never extrapolates past one pair
}
TEST(DiisOracleTests, RingWrapMatchesOracleFloat) { expect_matches_oracle<float>(100, 5, 17, 6); }

TEST(DiisOracleTests, LargeHistoryMatchesOracle) {
  expect_matches_oracle<double>(4096, kDiisMaxHistory, kDiisMaxHistory + 3, 7);
}

TEST(DiisOracleTests, SingularSubspaceReturnsNewestFock) {
  constexpr int kLen = 50;
  std::mt19937 rng(8);
  DeviceDiis<double> dev(kLen, 3);
  const auto e = random_vector<double>(rng, kLen);
  ASSERT_TRUE(dev.push(random_vector<double>(rng, kLen), e).ok());
  const auto f2 = random_vector<double>(rng, kLen);
  ASSERT_TRUE(dev.push(f2, e).ok()); // two identical residuals: rank-1 Gram
  EXPECT_EQ(dev.singular_flag(), 1);
  EXPECT_EQ(dev.output(), f2);
}

TEST(DiisOracleTests, SingularFallbackSelectsNewestAfterWrap) {
  // cap 3: after the fourth push the newest pair sits in physical column 0, not
  // the last one, and duplicates the residual in column 1. Integer residuals keep
  // every Gram entry exact, so the duplicate rows stay identical through the
  // elimination and the pivot is exactly 0 rather than a rounding residue.
  constexpr int kLen = 50;
  std::mt19937 rng(9);
  std::uniform_int_distribution<int> small_int(-3, 3);
  DeviceDiis<double> dev(kLen, 3);
  std::vector<std::vector<double>> es;
  for (int p = 0; p < 3; ++p) {
    std::vector<double> e(kLen);
    for (auto &x : e) {
      x = small_int(rng);
    }
    es.push_back(e);
    ASSERT_TRUE(dev.push(random_vector<double>(rng, kLen), es.back()).ok());
    EXPECT_EQ(dev.singular_flag(), 0);
  }
  const auto f4 = random_vector<double>(rng, kLen);
  ASSERT_TRUE(dev.push(f4, es[1]).ok());
  EXPECT_EQ(dev.state.head, 1);
  EXPECT_EQ(dev.singular_flag(), 1);
  EXPECT_EQ(dev.output(), f4);
}

TEST(DiisOracleTests, ResetRestartsTheSubspace) {
  constexpr int kLen = 80;
  constexpr int kCap = 3;
  std::mt19937 rng(10);
  DeviceDiis<double> dev(kLen, kCap);
  for (int p = 0; p < 5; ++p) {
    ASSERT_TRUE(dev.push(random_vector<double>(rng, kLen), random_vector<double>(rng, kLen)).ok());
  }
  diis_reset(dev.state);
  EXPECT_EQ(dev.state.size, 0);
  EXPECT_EQ(dev.state.head, 0);

  const std::vector<double> sentinel(kLen, -7.0);
  upload(dev.handle, dev.out, sentinel);
  HostDiis host{kCap, {}, {}};
  const auto f1 = random_vector<double>(rng, kLen);
  const auto e1 = random_vector<double>(rng, kLen, 0.1);
  ASSERT_TRUE(dev.push(f1, e1).ok());
  host.push(f1, e1);
  EXPECT_EQ(dev.output(), sentinel);
  // The stale pre-reset Gram entries must not leak into the fresh solve.
  const auto f2 = random_vector<double>(rng, kLen);
  const auto e2 = random_vector<double>(rng, kLen, 0.1);
  ASSERT_TRUE(dev.push(f2, e2).ok());
  host.push(f2, e2);
  expect_near_vector(dev.output(), host.extrapolate(), 1e-12, "after reset");
}

TEST(DiisOracleTests, RejectsBadArguments) {
  constexpr int kLen = 16;
  DeviceDiis<double> dev(kLen, 4);
  const auto stream = dev.handle->stream().get();
  const auto call = [&](DiisState &state, const double *f, std::size_t bytes) {
    return diis_push_and_extrapolate<double>(dev.blas, stream, state, f, dev.residual.data(),
                                             dev.out.data(), nullptr, dev.work.data(), bytes);
  };
  DiisState state = dev.state;
  EXPECT_FALSE(call(state, dev.fock.data(), dev.work_bytes - 1).ok());
  EXPECT_FALSE(call(state, nullptr, dev.work_bytes).ok());
  DiisState too_deep = make_diis_state(kLen, kDiisMaxHistory + 1);
  EXPECT_FALSE(call(too_deep, dev.fock.data(), dev.work_bytes).ok());
  DiisState bad_head = state;
  bad_head.head = 2; // head moves only once the ring is full
  EXPECT_FALSE(call(bad_head, dev.fock.data(), dev.work_bytes).ok());
  EXPECT_EQ(state.size, 0) << "a rejected push must not advance the state";
  EXPECT_TRUE(call(state, dev.fock.data(), dev.work_bytes).ok());
  EXPECT_EQ(state.size, 1);
}

// x <- g(x) = G x + c with a contraction G (spectral radius 0.95): the plain
// iteration needs hundreds of steps, DIIS with a history at least n deep solves
// the linear problem in about n + 2.
TEST(DiisOracleTests, AcceleratesLinearFixedPoint) {
  constexpr int kN = 10;
  std::mt19937 rng(11);
  // G = Q diag(lambda) Q^T with lambda spread over [-0.95, 0.95].
  std::vector<double> q(static_cast<std::size_t>(kN) * kN);
  {
    auto a = random_vector<double>(rng, kN * kN);
    for (int j = 0; j < kN; ++j) { // Gram-Schmidt on the columns of a
      for (int k = 0; k < j; ++k) {
        double dot = 0.0;
        for (int i = 0; i < kN; ++i) {
          dot += a[j * kN + i] * q[k * kN + i];
        }
        for (int i = 0; i < kN; ++i) {
          a[j * kN + i] -= dot * q[k * kN + i];
        }
      }
      double nrm = 0.0;
      for (int i = 0; i < kN; ++i) {
        nrm += a[j * kN + i] * a[j * kN + i];
      }
      for (int i = 0; i < kN; ++i) {
        q[j * kN + i] = a[j * kN + i] / std::sqrt(nrm);
      }
    }
  }
  std::vector<double> g(static_cast<std::size_t>(kN) * kN, 0.0);
  for (int k = 0; k < kN; ++k) {
    const double lambda = -0.95 + 1.9 * k / (kN - 1);
    for (int j = 0; j < kN; ++j) {
      for (int i = 0; i < kN; ++i) {
        g[j * kN + i] += lambda * q[k * kN + i] * q[k * kN + j];
      }
    }
  }
  const auto c = random_vector<double>(rng, kN);
  const auto apply = [&](const std::vector<double> &x) {
    std::vector<double> y = c;
    for (int j = 0; j < kN; ++j) {
      for (int i = 0; i < kN; ++i) {
        y[i] += g[j * kN + i] * x[j];
      }
    }
    return y;
  };
  const auto residual_norm = [](const std::vector<double> &gx, const std::vector<double> &x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      s += (gx[i] - x[i]) * (gx[i] - x[i]);
    }
    return std::sqrt(s);
  };

  constexpr double kTarget = 1e-9;
  constexpr int kMaxIter = 60;

  int plain_iters = kMaxIter;
  {
    std::vector<double> x(kN, 0.0);
    for (int it = 0; it < kMaxIter; ++it) {
      const auto gx = apply(x);
      if (residual_norm(gx, x) < kTarget) {
        plain_iters = it;
        break;
      }
      x = gx;
    }
  }

  int diis_iters = kMaxIter;
  {
    DeviceDiis<double> dev(kN, kN + 2);
    std::vector<double> x(kN, 0.0);
    for (int it = 0; it < kMaxIter; ++it) {
      const auto gx = apply(x);
      std::vector<double> e(kN);
      for (int i = 0; i < kN; ++i) {
        e[i] = gx[i] - x[i];
      }
      if (residual_norm(gx, x) < kTarget) {
        diis_iters = it;
        break;
      }
      ASSERT_TRUE(dev.push(gx, e).ok());
      x = dev.state.size >= 2 ? dev.output() : gx;
    }
  }

  EXPECT_LE(diis_iters, kN + 4) << "plain iteration took " << plain_iters;
  EXPECT_EQ(plain_iters, kMaxIter) << "the plain iteration should not converge this fast";
}

} // namespace
} // namespace calaman
