// sytrs2.cu
//
// The device-kernel half of calaman.sytrs2: the pieces no BLAS expresses. The
// level-3 solve's bulk is wrapped trsm plus host-driven row swaps; this file owns
//
//   * syconv_value: the ?syconv VALUE move -- pulling each 2x2 block's D
//     off-diagonal out of the stored factor into a separate E vector (and
//     restoring it), so the triangle trsm sees is pure unit-triangular. One
//     thread walks the pivot sequence; O(n) trivial work beside the trsm.
//   * the 1x1 / 2x2 D^-1 applies -- structurally calaman.sytrs's kernels, kept
//     separate on the module boundary (the pstrf/pstf2 convention). The 2x2 now
//     reads its off-diagonal from E, not the factor.
//
// Complex arithmetic goes through calaman::device::elem_ops, so the symmetric
// (unconjugated) forms cover all four element types. Shared between both backends,
// like lacpy.cu; the .cu extension is all CMake needs (HIP forces -x hip).
#include "sytrs2_bridge.h"

#include "common/constants.h"
#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief [kernel] ?syconv VALUE move (convert or revert), one thread
///
/// Mirrors the reference ?syconv VALUE loops: on convert, each 2x2 block's
/// off-diagonal moves from the factor into E (zeroed in the factor) and 1x1
/// positions set E to zero; on revert, the off-diagonals move back. The pivot
/// signs come from ipiv; kZero<T> is the portable zero.
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

/// @brief Scale one B row by 1/D(k,k): b[j*ldb] <- b[j*ldb] / *akk, per column j
template<typename T>
struct ScaleRowFunctor {
  T *const b_;
  const int ldb_;
  const T *const akk_;

  __device__ void operator()(const std::size_t j) const {
    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    b_[idx] = elem_ops<T>::div(b_[idx], *akk_);
  }
};

/// @brief Apply inv of the symmetric 2x2 D block to two B rows, per column j
///
/// Same arithmetic as calaman.sytrs, but the off-diagonal is the E entry ?syconv
/// pulled out. The scaled inputs are read into locals before either row is
/// written, so the overwrite order does not matter.
template<typename T>
struct Solve2x2Functor {
  T *const b_top_;
  T *const b_bot_;
  const int ldb_;
  const T *const top_diag_;
  const T *const off_;
  const T *const bot_diag_;

  __device__ void operator()(const std::size_t j) const {
    using ops = elem_ops<T>;
    using R = typename ops::real_type;
    const T off = *off_;
    const T akm1 = ops::div(*top_diag_, off);
    const T ak = ops::div(*bot_diag_, off);
    const T denom = ops::sub(ops::mul(akm1, ak), ops::from_real(R(1)));

    const std::size_t idx = j * static_cast<std::size_t>(ldb_);
    const T b_top = ops::div(b_top_[idx], off);
    const T b_bot = ops::div(b_bot_[idx], off);
    b_top_[idx] = ops::div(ops::sub(ops::mul(ak, b_top), b_bot), denom);
    b_bot_[idx] = ops::div(ops::sub(ops::mul(akm1, b_bot), b_top), denom);
  }
};

} // namespace

template<typename T>
void syconv_value(const wwr::wwrStream_t stream, const bool upper, const bool convert, const int n,
                  T *const A, const std::size_t lda, const int *const d_ipiv, T *const E) {
  if (n <= 0) {
    return;
  }
  syconv_value_kernel<T><<<1, 1, 0, stream>>>(upper, convert, n, A, lda, d_ipiv, E);
}

template<typename T>
void sytrs2_scale_row(const wwr::wwrStream_t stream, T *const b_row, const int ldb, const int nrhs,
                      const T *const akk) {
  if (nrhs <= 0) {
    return;
  }
  const ScaleRowFunctor<T> functor{b_row, ldb, akk};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

template<typename T>
void sytrs2_solve_2x2(const wwr::wwrStream_t stream, T *const b_top, T *const b_bot, const int ldb,
                      const int nrhs, const T *const top_diag, const T *const off,
                      const T *const bot_diag) {
  if (nrhs <= 0) {
    return;
  }
  const Solve2x2Functor<T> functor{b_top, b_bot, ldb, top_diag, off, bot_diag};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(nrhs), functor);
}

// One per supported type, matching sytrs2_bridge.h and the module's use sites.
template void syconv_value<float>(wwr::wwrStream_t, bool, bool, int, float *, std::size_t,
                                  const int *, float *);
template void syconv_value<double>(wwr::wwrStream_t, bool, bool, int, double *, std::size_t,
                                   const int *, double *);
template void syconv_value<wwr::wwrFloatComplex>(wwr::wwrStream_t, bool, bool, int,
                                                 wwr::wwrFloatComplex *, std::size_t, const int *,
                                                 wwr::wwrFloatComplex *);
template void syconv_value<wwr::wwrDoubleComplex>(wwr::wwrStream_t, bool, bool, int,
                                                  wwr::wwrDoubleComplex *, std::size_t, const int *,
                                                  wwr::wwrDoubleComplex *);

template void sytrs2_scale_row<float>(wwr::wwrStream_t, float *, int, int, const float *);
template void sytrs2_scale_row<double>(wwr::wwrStream_t, double *, int, int, const double *);
template void sytrs2_scale_row<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *, int,
                                                     int, const wwr::wwrFloatComplex *);
template void sytrs2_scale_row<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                      int, int, const wwr::wwrDoubleComplex *);

template void sytrs2_solve_2x2<float>(wwr::wwrStream_t, float *, float *, int, int, const float *,
                                      const float *, const float *);
template void sytrs2_solve_2x2<double>(wwr::wwrStream_t, double *, double *, int, int,
                                       const double *, const double *, const double *);
template void sytrs2_solve_2x2<wwr::wwrFloatComplex>(wwr::wwrStream_t, wwr::wwrFloatComplex *,
                                                     wwr::wwrFloatComplex *, int, int,
                                                     const wwr::wwrFloatComplex *,
                                                     const wwr::wwrFloatComplex *,
                                                     const wwr::wwrFloatComplex *);
template void sytrs2_solve_2x2<wwr::wwrDoubleComplex>(wwr::wwrStream_t, wwr::wwrDoubleComplex *,
                                                      wwr::wwrDoubleComplex *, int, int,
                                                      const wwr::wwrDoubleComplex *,
                                                      const wwr::wwrDoubleComplex *,
                                                      const wwr::wwrDoubleComplex *);

} // namespace calaman::device
