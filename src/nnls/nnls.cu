// nnls.cu
//
// The device-kernel half of calaman.nnls: the Lawson-Hanson active-set
// bookkeeping over the P (passive) / Z (zero) partition that no BLAS call
// expresses. Everything numeric in nnls() is a host composition of wrapped BLAS
// (gemv/trsv/nrm2), calaman::geqp3 (the passive-set pivoted QR) and wwr::ormqr
// (applying Q); this file owns only the scattered-index reductions and gathers
// around them:
//
//   * iota / gather / scatter / line-step: elementwise over an index set, so
//     launched through wwr.extension.parallel_for, like laqp2.cu / laqps.cu;
//   * masked argmax (entering column) and the min-ratio step test: single-block
//     reductions over a SCATTERED subset of order, hand-launched;
//   * compact-zeros / swap / rank: tiny single-thread launches.
//
// Shared unchanged between both backends, like lacpy.cu; the .cu extension is
// all CMake needs (under HIP the CMakeLists forces -x hip), so no __CUDACC__.
#include "nnls_bridge.h"

#include <extension/parallel_for/parallel_for.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

__device__ __forceinline__ float nnls_abs(const float x) { return x < 0.0f ? -x : x; }
__device__ __forceinline__ double nnls_abs(const double x) { return x < 0.0 ? -x : x; }

constexpr int kNnlsBlock = 256;

// ── parallel_for functors: trivially copyable, const members, __device__ const
// operator(), as device_functor requires (a mutable member is UB in the
// grid-constant copy). Each writes only its output slot, never its own members.

struct IotaFunctor {
  int *const order_;
  __device__ void operator()(const std::size_t i) const { order_[i] = static_cast<int>(i); }
};

template<typename T>
struct GatherFunctor {
  T *const AP_;          ///< packed m-by-p output, leading dimension m_
  const T *const A_;     ///< source matrix, leading dimension lda_
  const int *const order_;
  const std::size_t m_;
  const std::size_t lda_;
  __device__ void operator()(const std::size_t idx) const {
    const std::size_t c = idx / m_;
    const std::size_t r = idx % m_;
    AP_[c * m_ + r] = A_[static_cast<std::size_t>(order_[c]) * lda_ + r];
  }
};

template<typename T>
struct ScatterFunctor {
  T *const z_;
  const T *const qb_;
  const int *const order_;
  const int *const jpvt0_;  ///< geqp3's pivot, 0-based
  __device__ void operator()(const std::size_t j) const {
    const int local = jpvt0_[j];
    z_[order_[local]] = qb_[j];
  }
};

template<typename T>
struct LineStepFunctor {
  T *const x_;
  const T *const z_;
  const T alpha_;
  __device__ void operator()(const std::size_t i) const { x_[i] += alpha_ * (z_[i] - x_[i]); }
};

/// Single block, grid-stride: the position in [p, n) maximizing w[order[k]].
template<typename T, int BLOCK>
__global__ void masked_argmax_kernel(const T *__restrict__ w, const int *__restrict__ order,
                                     const int p, const int n, T *max_val, int *arg_pos) {
  __shared__ T sval[BLOCK];
  __shared__ int spos[BLOCK];
  const int tid = static_cast<int>(threadIdx.x);

  T best = T(0);
  int best_pos = -1;
  for (int k = p + tid; k < n; k += BLOCK) {
    const T v = w[order[k]];
    if (best_pos < 0 || v > best) {
      best = v;
      best_pos = k;
    }
  }
  sval[tid] = best;
  spos[tid] = best_pos;
  __syncthreads();

  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const int other_pos = spos[tid + s];
      if (other_pos >= 0 && (spos[tid] < 0 || sval[tid + s] > sval[tid])) {
        sval[tid] = sval[tid + s];
        spos[tid] = other_pos;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    *max_val = sval[0];
    *arg_pos = spos[0];
  }
}

/// Single block, grid-stride: the Lawson-Hanson min-ratio test over [0, p).
template<typename T, int BLOCK>
__global__ void masked_min_ratio_kernel(const T *__restrict__ x, const T *__restrict__ z,
                                        const int *__restrict__ order, const int p, T *alpha,
                                        int *infeasible) {
  __shared__ T sval[BLOCK];
  __shared__ int sflag[BLOCK];
  const int tid = static_cast<int>(threadIdx.x);

  // Strictly negative only: z_j == 0 exactly is already feasible (x >= 0 allows
  // equality), and treating it as a violation would divide by zero whenever
  // x_j == 0 too -- which a freshly-entered column (x_j is 0 until its first
  // solve) lands on exactly whenever it turns out to sit in a rank-deficient
  // subproblem's truncated tail (z_j = 0 there by construction).
  T best = T(0);
  int have = 0;
  for (int k = tid; k < p; k += BLOCK) {
    const int idx = order[k];
    const T zj = z[idx];
    if (zj < T(0)) {
      const T xj = x[idx];
      const T ratio = xj / (xj - zj);
      if (!have || ratio < best) {
        best = ratio;
        have = 1;
      }
    }
  }
  sval[tid] = best;
  sflag[tid] = have;
  __syncthreads();

  for (int s = BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (sflag[tid + s] && (!sflag[tid] || sval[tid + s] < sval[tid])) {
        sval[tid] = sval[tid + s];
        sflag[tid] = 1;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    *alpha = sval[0];
    *infeasible = sflag[0];
  }
}

/// Single thread: stable-partition order[0, p) by |x[order[k]]| > eps.
template<typename T>
__global__ void compact_zeros_kernel(int *order, int *scratch, const T *__restrict__ x, const int p,
                                     const T eps, int *new_p) {
  if (threadIdx.x != 0) {
    return;
  }
  int kept = 0;
  int removed = 0;
  for (int k = 0; k < p; ++k) {
    const int idx = order[k];
    if (nnls_abs(x[idx]) > eps) {
      scratch[kept] = idx;
      ++kept;
    } else {
      scratch[p - 1 - removed] = idx;
      ++removed;
    }
  }
  for (int k = 0; k < p; ++k) {
    order[k] = scratch[k];
  }
  *new_p = kept;
}

__global__ void swap_order_kernel(int *order, const int a, const int b) {
  if (threadIdx.x != 0) {
    return;
  }
  const int t = order[a];
  order[a] = order[b];
  order[b] = t;
}

/// Single thread: count the pivoted-R diagonal entries above the relative tol.
template<typename T>
__global__ void rank_kernel(const T *__restrict__ R, const int ldr, const int k, const T rel_tol,
                            int *rank) {
  if (threadIdx.x != 0) {
    return;
  }
  if (k <= 0) {
    *rank = 0;
    return;
  }
  const T thresh = rel_tol * nnls_abs(R[0]);
  int rk = 0;
  for (int i = 0; i < k; ++i) {
    if (nnls_abs(R[static_cast<std::size_t>(i) * ldr + i]) > thresh) {
      ++rk;
    } else {
      break;  // pivoting orders |R[i,i]| non-increasing: the pass is a prefix
    }
  }
  *rank = rk;
}

} // namespace

template<typename T>
void nnls_init(const wwr::wwrStream_t stream, int *const order, T *const x, const int n) {
  if (n < 1) {
    return;
  }
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(n),
                                            IotaFunctor{order});
  wwr::wwrMemsetAsync(x, 0, static_cast<std::size_t>(n) * sizeof(T), stream);
}

void nnls_swap_order(const wwr::wwrStream_t stream, int *const order, const int a, const int b) {
  swap_order_kernel<<<1, 1, 0, stream>>>(order, a, b);
}

template<typename T>
void nnls_masked_argmax(const wwr::wwrStream_t stream, const T *const w, const int *const order,
                        const int p, const int n, T *const max_val, int *const arg_pos) {
  masked_argmax_kernel<T, kNnlsBlock><<<1, kNnlsBlock, 0, stream>>>(w, order, p, n, max_val,
                                                                    arg_pos);
}

template<typename T>
void nnls_gather_columns(const wwr::wwrStream_t stream, T *const AP, const int m, const T *const A,
                         const int lda, const int *const order, const int p) {
  if (m < 1 || p < 1) {
    return;
  }
  const GatherFunctor<T> functor{AP, A, order, static_cast<std::size_t>(m),
                                 static_cast<std::size_t>(lda)};
  wwr::extension::parallel_for<std::size_t>(
      stream, static_cast<std::size_t>(m) * static_cast<std::size_t>(p), functor);
}

template<typename T>
void nnls_rank(const wwr::wwrStream_t stream, const T *const R, const int ldr, const int k,
               const T rel_tol, int *const rank) {
  rank_kernel<T><<<1, 1, 0, stream>>>(R, ldr, k, rel_tol, rank);
}

template<typename T>
void nnls_scatter_z(const wwr::wwrStream_t stream, T *const z, const T *const qb,
                    const int *const order, const int *const jpvt0, const int p) {
  if (p < 1) {
    return;
  }
  const ScatterFunctor<T> functor{z, qb, order, jpvt0};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(p), functor);
}

template<typename T>
void nnls_masked_min_ratio(const wwr::wwrStream_t stream, const T *const x, const T *const z,
                           const int *const order, const int p, T *const alpha,
                           int *const infeasible) {
  masked_min_ratio_kernel<T, kNnlsBlock><<<1, kNnlsBlock, 0, stream>>>(x, z, order, p, alpha,
                                                                       infeasible);
}

template<typename T>
void nnls_line_step(const wwr::wwrStream_t stream, T *const x, const T *const z, const T alpha,
                    const int n) {
  if (n < 1) {
    return;
  }
  const LineStepFunctor<T> functor{x, z, alpha};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(n), functor);
}

template<typename T>
void nnls_compact_zeros(const wwr::wwrStream_t stream, int *const order, int *const scratch,
                        const T *const x, const int p, const T eps, int *const new_p) {
  if (p < 1) {
    wwr::wwrMemsetAsync(new_p, 0, sizeof(int), stream);
    return;
  }
  compact_zeros_kernel<T><<<1, 1, 0, stream>>>(order, scratch, x, p, eps, new_p);
}

// One per supported type (float, double), matching nnls_bridge.h and the
// module's use sites.
template void nnls_init<float>(wwr::wwrStream_t, int *, float *, int);
template void nnls_init<double>(wwr::wwrStream_t, int *, double *, int);

template void nnls_masked_argmax<float>(wwr::wwrStream_t, const float *, const int *, int, int,
                                        float *, int *);
template void nnls_masked_argmax<double>(wwr::wwrStream_t, const double *, const int *, int, int,
                                         double *, int *);

template void nnls_gather_columns<float>(wwr::wwrStream_t, float *, int, const float *, int,
                                         const int *, int);
template void nnls_gather_columns<double>(wwr::wwrStream_t, double *, int, const double *, int,
                                          const int *, int);

template void nnls_rank<float>(wwr::wwrStream_t, const float *, int, int, float, int *);
template void nnls_rank<double>(wwr::wwrStream_t, const double *, int, int, double, int *);

template void nnls_scatter_z<float>(wwr::wwrStream_t, float *, const float *, const int *,
                                    const int *, int);
template void nnls_scatter_z<double>(wwr::wwrStream_t, double *, const double *, const int *,
                                     const int *, int);

template void nnls_masked_min_ratio<float>(wwr::wwrStream_t, const float *, const float *,
                                           const int *, int, float *, int *);
template void nnls_masked_min_ratio<double>(wwr::wwrStream_t, const double *, const double *,
                                            const int *, int, double *, int *);

template void nnls_line_step<float>(wwr::wwrStream_t, float *, const float *, float, int);
template void nnls_line_step<double>(wwr::wwrStream_t, double *, const double *, double, int);

template void nnls_compact_zeros<float>(wwr::wwrStream_t, int *, int *, const float *, int, float,
                                        int *);
template void nnls_compact_zeros<double>(wwr::wwrStream_t, int *, int *, const double *, int, double,
                                         int *);

} // namespace calaman::device
