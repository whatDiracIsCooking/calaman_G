// hetrs2.cu
//
// The device-kernel half of calaman.hetrs2, the Hermitian level-3 solve. Its bulk
// is wrapped trsm (the second one conjugate-transpose) plus host-driven swaps; this
// file owns the pieces no BLAS expresses:
//
//   * hetrs2_syconv_value: the ?syconv VALUE move (unconjugated -- ?hetrs2 reuses
//     the symmetric ?syconv), pulling each 2x2 block's D off-diagonal into E;
//   * the Hermitian D^-1 applies -- the real-reciprocal 1x1 and the conjugated 2x2
//     (structurally calaman.hetrs's, kept separate on the module boundary).
//
// Complex only (c/z). Arithmetic goes through calaman::device::elem_ops. Shared
// between both backends, like lacpy.cu.
#include "hetrs2_bridge.h"

#include "common/constants.h"
#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief [kernel] ?syconv VALUE move (convert or revert), one thread (no conj)
template<typename T>
__global__ void syconv_value_kernel(const bool upper, const bool convert, const int n,
                                    T *__restrict__ A, const std::size_t lda,
                                    const int *__restrict__ ipiv, T *__restrict__ E) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  if (upper) {
    if (convert) {
      E[0] = kZero<T>;
      int i = n - 1;
      while (i > 0) {
        if (ipiv[i] < 0) {
          E[i] = A[static_cast<std::size_t>(i) * lda + (i - 1)];
          E[i - 1] = kZero<T>;
          A[static_cast<std::size_t>(i) * lda + (i - 1)] = kZero<T>;
          --i;
        } else {
          E[i] = kZero<T>;
        }
        --i;
      }
    } else {
      int i = n - 1;
      while (i > 0) {
        if (ipiv[i] < 0) {
          A[static_cast<std::size_t>(i) * lda + (i - 1)] = E[i];
          --i;
        }
        --i;
      }
    }
  } else {
    if (convert) {
      E[n - 1] = kZero<T>;
      int i = 0;
      while (i < n) {
        if (i < n - 1 && ipiv[i] < 0) {
          E[i] = A[static_cast<std::size_t>(i) * lda + (i + 1)];
          E[i + 1] = kZero<T>;
          A[static_cast<std::size_t>(i) * lda + (i + 1)] = kZero<T>;
          ++i;
        } else {
          E[i] = kZero<T>;
        }
        ++i;
      }
    } else {
      int i = 0;
      while (i <= n - 2) {
        if (ipiv[i] < 0) {
          A[static_cast<std::size_t>(i) * lda + (i + 1)] = E[i];
          ++i;
        }
        ++i;
      }
    }
  }
}

/// @brief Scale one B row by 1/Re(D(k,k)): b[j*ldb] *= 1/real(*akk), per column j
template<typename T>
struct ScaleRowRealFunctor {
  T *const b_;
  const int ldb_;
  const T *const akk_;

  __device__ void operator()(const std::size_t j) const {
    using ops = elem_ops<T>;
    using R = typename ops::real_type;
    const R s = R(1) / ops::real_part(*akk_);
    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    b_[idx] = ops::scale(b_[idx], s);
  }
};

/// @brief Apply inv of the Hermitian 2x2 D block to two B rows, per column j
template<typename T>
struct Solve2x2Functor {
  T *const b_top_;
  T *const b_bot_;
  const int ldb_;
  const T *const top_diag_;
  const T *const off_;
  const T *const bot_diag_;
  const bool top_conj_;

  __device__ void operator()(const std::size_t j) const {
    using ops = elem_ops<T>;
    using R = typename ops::real_type;
    const T off = *off_;
    const T off_c = ops::conj(off);
    const T top_div = top_conj_ ? off_c : off;
    const T bot_div = top_conj_ ? off : off_c;

    const T akm1 = ops::div(*top_diag_, top_div);
    const T ak = ops::div(*bot_diag_, bot_div);
    const T denom = ops::sub(ops::mul(akm1, ak), ops::from_real(R(1)));

    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    const T b_top = ops::div(b_top_[idx], top_div);
    const T b_bot = ops::div(b_bot_[idx], bot_div);
    b_top_[idx] = ops::div(ops::sub(ops::mul(ak, b_top), b_bot), denom);
    b_bot_[idx] = ops::div(ops::sub(ops::mul(akm1, b_bot), b_top), denom);
  }
};

} // namespace

template<typename T>
void hetrs2_syconv_value(const wwr::wwrStream_t stream, const bool upper, const bool convert,
                         const int n, T *const A, const std::size_t lda, const int *const d_ipiv,
                         T *const E) {
  if (n <= 0) {
    return;
  }
  syconv_value_kernel<T><<<1, 1, 0, stream>>>(upper, convert, n, A, lda, d_ipiv, E);
}

template<typename T>
void hetrs2_scale_row_real(const wwr::wwrStream_t stream, T *const b_row, const int ldb,
                           const int nrhs, const T *const akk) {
  if (nrhs <= 0) {
    return;
  }
  const ScaleRowRealFunctor<T> functor{b_row, ldb, akk};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

template<typename T>
void hetrs2_solve_2x2(const wwr::wwrStream_t stream, T *const b_top, T *const b_bot, const int ldb,
                      const int nrhs, const T *const top_diag, const T *const off,
                      const T *const bot_diag, const bool top_conj) {
  if (nrhs <= 0) {
    return;
  }
  const Solve2x2Functor<T> functor{b_top, b_bot, ldb, top_diag, off, bot_diag, top_conj};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

// One per supported type (complex only), matching hetrs2_bridge.h and the module.
template void hetrs2_syconv_value<wwr::wwrFloatComplex>(wwr::wwrStream_t, bool, bool, int,
                                                        wwr::wwrFloatComplex *, std::size_t,
                                                        const int *, wwr::wwrFloatComplex *);
template void hetrs2_syconv_value<wwr::wwrDoubleComplex>(wwr::wwrStream_t, bool, bool, int,
                                                         wwr::wwrDoubleComplex *, std::size_t,
                                                         const int *, wwr::wwrDoubleComplex *);

template void hetrs2_scale_row_real<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                          int, int, const wwr::wwrFloatComplex *);
template void hetrs2_scale_row_real<wwr::wwrDoubleComplex>(wwr::wwrStream_t,
                                                           wwr::wwrDoubleComplex *, int, int,
                                                           const wwr::wwrDoubleComplex *);

template void hetrs2_solve_2x2<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                     wwr::wwrFloatComplex *, int, int,
                                                     const wwr::wwrFloatComplex *,
                                                     const wwr::wwrFloatComplex *,
                                                     const wwr::wwrFloatComplex *, bool);
template void hetrs2_solve_2x2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                      wwr::wwrDoubleComplex *, int, int,
                                                      const wwr::wwrDoubleComplex *,
                                                      const wwr::wwrDoubleComplex *,
                                                      const wwr::wwrDoubleComplex *, bool);

} // namespace calaman::device
