// hetf2.cu
//
// The device-kernel half of calaman.hetf2: the two stages of the unblocked
// Hermitian Bunch-Kaufman factorization that no BLAS expresses. The pivot search,
// the Bunch-Kaufman decisions and the 1x1 rank-1 update (her/scal) run host-side in
// hetf2(); this file owns
//
//   * hetf2_interchange: the Hermitian swap of a pivot pair -- a conjugating
//     row/column dance over the stored triangle, single-threaded (O(n) per step,
//     dwarfed by the rank update);
//   * hetf2_rank2: the 2x2 pivot block's rank-2 trailing update -- form the two
//     factor columns W, subtract W inv(D) W^H from the trailing triangle, store W,
//     keep the diagonal real; three parallel passes over n-length scratch wa/wb.
//
// Complex only (c/z). Arithmetic goes through calaman::device::elem_ops. Shared
// between both backends, like lacpy.cu.
#include "hetf2_bridge.h"

#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

template<typename T>
__device__ __forceinline__ T &at(T *A, std::size_t lda, int r, int c) {
  return A[static_cast<std::size_t>(c) * lda + static_cast<std::size_t>(r)];
}
template<typename T>
__device__ __forceinline__ T real_only(const T &z) {
  return elem_ops<T>::from_real(elem_ops<T>::real_part(z));
}

/// @brief [kernel] Hermitian interchange of the pivot pair (single thread)
template<typename T>
__global__ void interchange_kernel(const bool upper, const int n, T *__restrict__ A,
                                   const std::size_t lda, const int k, const int kk, const int kp,
                                   const int kstep) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  using ops = elem_ops<T>;
  if (kp != kk) {
    if (!upper) {
      for (int i = kp + 1; i < n; ++i) {
        const T t = at(A, lda, i, kk);
        at(A, lda, i, kk) = at(A, lda, i, kp);
        at(A, lda, i, kp) = t;
      }
      for (int j = kk + 1; j < kp; ++j) {
        const T t = ops::conj(at(A, lda, j, kk));
        at(A, lda, j, kk) = ops::conj(at(A, lda, kp, j));
        at(A, lda, kp, j) = t;
      }
    } else {
      for (int i = 0; i < kp; ++i) {
        const T t = at(A, lda, i, kk);
        at(A, lda, i, kk) = at(A, lda, i, kp);
        at(A, lda, i, kp) = t;
      }
      for (int j = kp + 1; j < kk; ++j) {
        const T t = ops::conj(at(A, lda, j, kk));
        at(A, lda, j, kk) = ops::conj(at(A, lda, kp, j));
        at(A, lda, kp, j) = t;
      }
    }
    at(A, lda, kp, kk) = ops::conj(at(A, lda, kp, kk));
    const T r1 = real_only(at(A, lda, kk, kk));
    at(A, lda, kk, kk) = real_only(at(A, lda, kp, kp));
    at(A, lda, kp, kp) = r1;
    if (kstep == 2) {
      at(A, lda, k, k) = real_only(at(A, lda, k, k));
      const int kadj = upper ? k - 1 : k + 1;
      const T t = at(A, lda, kadj, k);
      at(A, lda, kadj, k) = at(A, lda, kp, k);
      at(A, lda, kp, k) = t;
    }
  } else {
    at(A, lda, k, k) = real_only(at(A, lda, k, k));
    if (kstep == 2) {
      const int d = upper ? k - 1 : k + 1;
      at(A, lda, d, d) = real_only(at(A, lda, d, d));
    }
  }
}

// ── rank-2 trailing update: form W, subtract W inv(D) W^H, store W ────────────
//
// col_a is column k; col_b is k-1 (upper) or k+1 (lower). The j-range is [0, k-1)
// upper / [k+2, n) lower, mapped from the thread index jj. W_a = dfac*(da*A(j,a) -
// doff*A(j,b)); W_b = dfac*(db*A(j,b) - conj(doff)*A(j,a)) -- the ?hetf2 WK/WKM1.

template<typename T>
struct ComputeWFunctor {
  const bool upper_;
  const int n_, k_;
  const T *const A_;
  const std::size_t lda_;
  const typename elem_ops<T>::real_type da_, db_, dfac_;
  // Non-const: a const class-type member (T is a complex struct) would make the
  // functor non-trivially-copyable, which device_functor rejects.
  T doff_;
  T *const wa_;
  T *const wb_;

  __device__ void operator()(const std::size_t jj) const {
    using ops = elem_ops<T>;
    const int col_a = k_;
    const int col_b = upper_ ? k_ - 1 : k_ + 1;
    const int j = upper_ ? static_cast<int>(jj) : (k_ + 2 + static_cast<int>(jj));
    const T aja = A_[static_cast<std::size_t>(col_a) * lda_ + j];
    const T ajb = A_[static_cast<std::size_t>(col_b) * lda_ + j];
    wa_[j] = ops::scale(ops::sub(ops::scale(aja, da_), ops::mul(doff_, ajb)), dfac_);
    wb_[j] = ops::scale(ops::sub(ops::scale(ajb, db_), ops::mul(ops::conj(doff_), aja)), dfac_);
  }
};

template<typename T>
struct UpdateFunctor {
  const bool upper_;
  const int n_, k_, jcount_;
  T *const A_;
  const std::size_t lda_;
  const T *const wa_;
  const T *const wb_;

  __device__ void operator()(const std::size_t idx) const {
    using ops = elem_ops<T>;
    const int i = static_cast<int>(idx % static_cast<std::size_t>(n_));
    const int ljj = static_cast<int>(idx / static_cast<std::size_t>(n_));
    const int col_a = k_;
    const int col_b = upper_ ? k_ - 1 : k_ + 1;
    const int j = upper_ ? ljj : (k_ + 2 + ljj);
    const bool in = upper_ ? (i <= j) : (i >= j);
    if (!in) {
      return;
    }
    const T upd =
        ops::add(ops::mul(A_[static_cast<std::size_t>(col_a) * lda_ + i], ops::conj(wa_[j])),
                 ops::mul(A_[static_cast<std::size_t>(col_b) * lda_ + i], ops::conj(wb_[j])));
    A_[static_cast<std::size_t>(j) * lda_ + i] =
        ops::sub(A_[static_cast<std::size_t>(j) * lda_ + i], upd);
  }
};

template<typename T>
struct StoreWFunctor {
  const bool upper_;
  const int k_;
  T *const A_;
  const std::size_t lda_;
  const T *const wa_;
  const T *const wb_;

  __device__ void operator()(const std::size_t jj) const {
    const int col_a = k_;
    const int col_b = upper_ ? k_ - 1 : k_ + 1;
    const int j = upper_ ? static_cast<int>(jj) : (k_ + 2 + static_cast<int>(jj));
    A_[static_cast<std::size_t>(col_a) * lda_ + j] = wa_[j];
    A_[static_cast<std::size_t>(col_b) * lda_ + j] = wb_[j];
    A_[static_cast<std::size_t>(j) * lda_ + j] =
        real_only(A_[static_cast<std::size_t>(j) * lda_ + j]);
  }
};

} // namespace

template<typename T>
void hetf2_interchange(const wwr::wwrStream_t stream, const bool upper, const int n, T *const A,
                       const std::size_t lda, const int k, const int kk, const int kp,
                       const int kstep) {
  interchange_kernel<T><<<1, 1, 0, stream>>>(upper, n, A, lda, k, kk, kp, kstep);
}

template<typename T>
void hetf2_rank2(const wwr::wwrStream_t stream, const bool upper, const int n, T *const A,
                 const std::size_t lda, const int k, const double d_a, const double d_b,
                 const T doff, const double dfac, T *const wa, T *const wb) {
  using R = typename elem_ops<T>::real_type;
  const int jcount = upper ? (k - 1) : (n - k - 2);
  if (jcount <= 0) {
    return;
  }
  const R da = static_cast<R>(d_a);
  const R db = static_cast<R>(d_b);
  const R df = static_cast<R>(dfac);
  const ComputeWFunctor<T> compute{upper, n, k, A, lda, da, db, df, doff, wa, wb};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(jcount), compute);
  const UpdateFunctor<T> update{upper, n, k, jcount, A, lda, wa, wb};
  wwr::extension::parallel_for<std::size_t>(
      stream, static_cast<std::size_t>(jcount) * static_cast<std::size_t>(n), update);
  const StoreWFunctor<T> store{upper, k, A, lda, wa, wb};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(jcount), store);
}

// One per supported type (complex only), matching hetf2_bridge.h and the module.
template void hetf2_interchange<wwr::wwrFloatComplex>(wwr::wwrStream_t, bool, int,
                                                      wwr::wwrFloatComplex *, std::size_t, int, int,
                                                      int, int);
template void hetf2_interchange<wwr::wwrDoubleComplex>(wwr::wwrStream_t, bool, int,
                                                       wwr::wwrDoubleComplex *, std::size_t, int,
                                                       int, int, int);

template void hetf2_rank2<wwr::wwrFloatComplex>(wwr::wwrStream_t, bool, int, wwr::wwrFloatComplex *,
                                                std::size_t, int, double, double,
                                                wwr::wwrFloatComplex, double,
                                                wwr::wwrFloatComplex *, wwr::wwrFloatComplex *);
template void hetf2_rank2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, bool, int,
                                                 wwr::wwrDoubleComplex *, std::size_t, int, double,
                                                 double, wwr::wwrDoubleComplex, double,
                                                 wwr::wwrDoubleComplex *, wwr::wwrDoubleComplex *);

} // namespace calaman::device
