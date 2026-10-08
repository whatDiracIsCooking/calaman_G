// shifted_cocg.cu
//
// The device-kernel half of calaman.shifted_cocg: the per-column Lanczos step
// and the per-(shift, column) COCG recurrences. The host driver (the operator
// apply, the loop, the convergence read-back) lives in shifted_cocg.cppm and
// reaches these through the launchers in shifted_cocg_bridge.h.
//
// Shared unchanged between both backends, like lanczos.cu. Every complex value
// is a pair of reals, so no complex type is named. Machine epsilon is
// <cfloat>'s FLT/DBL_EPSILON: nvcc rejects std::numeric_limits in device code.
#include "shifted_cocg_bridge.h"

#include "common/block_reduce.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace calaman::device {

namespace {

// 4 warps per block, matching lanczos/feast: a power of two, as block_reduce's
// halving tree requires.
constexpr unsigned int kBlock = 4 * WWR_WARP_SIZE;
constexpr std::size_t kMaxBlocks = 4096;

inline unsigned int blocks_for(const std::size_t elems) {
  std::size_t blocks = (elems + kBlock - 1) / kBlock;
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<unsigned int>(blocks);
}

__device__ __forceinline__ float abs_(const float x) {
  return fabsf(x);
}
__device__ __forceinline__ double abs_(const double x) {
  return fabs(x);
}
__device__ __forceinline__ float sqrt_(const float x) {
  return sqrtf(x);
}
__device__ __forceinline__ double sqrt_(const double x) {
  return sqrt(x);
}
__device__ __forceinline__ float hypot_(const float x, const float y) {
  return hypotf(x, y);
}
__device__ __forceinline__ double hypot_(const double x, const double y) {
  return hypot(x, y);
}

template<typename T>
struct MachineEps;
template<>
struct MachineEps<float> {
  static constexpr float kValue = FLT_EPSILON;
};
template<>
struct MachineEps<double> {
  static constexpr double kValue = DBL_EPSILON;
};

/// One block per column: bnorm, v_cur = b / bnorm, v_prev = 0, beta_cur = 0.
template<typename T>
__global__ void start_columns_kernel(const int n, const T *B, T *v_prev, T *v_cur, T *beta_cur,
                                     T *bnorm) {
  const std::size_t nz = static_cast<std::size_t>(n);
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * nz;
  T sum = T(0);
  for (std::size_t i = threadIdx.x; i < nz; i += blockDim.x) {
    const T b = B[off + i];
    sum += b * b;
  }
  const T norm = sqrt_(block_reduce<kBlock>(sum, AddOp{}));
  const T scale = norm > T(0) && isfinite(norm) ? T(1) / norm : T(0);
  for (std::size_t i = threadIdx.x; i < nz; i += blockDim.x) {
    v_cur[off + i] = B[off + i] * scale;
    v_prev[off + i] = T(0);
  }
  if (threadIdx.x == 0) {
    bnorm[blockIdx.x] = norm;
    beta_cur[blockIdx.x] = T(0);
  }
}

/// One thread per pair: eta = 1, zeta = bnorm_j (step 1 reads it as zeta_1).
/// A zero column is done at step 0 with residual 0; a non-finite one is done
/// at step 0 with residual NaN, which the host reads as a numerical failure.
template<typename T>
__global__ void start_pairs_kernel(const int k, const int pairs_count, const T *bnorm,
                                   CocgPairs<T> pairs) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= pairs_count) {
    return;
  }
  const T b = bnorm[p % k];
  pairs.eta_r[p] = T(1);
  pairs.eta_i[p] = T(0);
  pairs.ieta_r[p] = T(1);
  pairs.ieta_i[p] = T(0);
  pairs.zeta_r[p] = b;
  pairs.zeta_i[p] = T(0);
  const bool live = b > T(0) && isfinite(b);
  pairs.res[p] = live ? T(1) : (b == T(0) ? T(0) : T(NAN));
  pairs.done_step[p] = live ? -1 : 0;
}

/// One block per column: the three-term Lanczos step on w = A v_cur.
template<typename T>
__global__ void lanczos_kernel(const int n, const T *v_prev, const T *v_cur, T *w,
                               const T *beta_cur, T *alpha, T *beta_next) {
  const std::size_t nz = static_cast<std::size_t>(n);
  const std::size_t off = static_cast<std::size_t>(blockIdx.x) * nz;
  const T b = beta_cur[blockIdx.x];

  T dot = T(0);
  for (std::size_t i = threadIdx.x; i < nz; i += blockDim.x) {
    const T wi = w[off + i] - b * v_prev[off + i];
    w[off + i] = wi;
    dot += v_cur[off + i] * wi;
  }
  const T a = block_reduce<kBlock>(dot, AddOp{});

  T sum = T(0);
  for (std::size_t i = threadIdx.x; i < nz; i += blockDim.x) {
    const T wi = w[off + i] - a * v_cur[off + i];
    w[off + i] = wi;
    sum += wi * wi;
  }
  T bn = sqrt_(block_reduce<kBlock>(sum, AddOp{}));
  // NaN fails the comparison and is kept, so it reaches the residual estimates.
  if (bn <= MachineEps<T>::kValue * (abs_(a) + b)) {
    bn = T(0);
  }
  const T scale = bn > T(0) ? T(1) / bn : T(0);
  for (std::size_t i = threadIdx.x; i < nz; i += blockDim.x) {
    w[off + i] = w[off + i] * scale;
  }
  if (threadIdx.x == 0) {
    alpha[blockIdx.x] = a;
    beta_next[blockIdx.x] = bn;
  }
}

/// One thread per pair. With M = z I - A, whose tridiagonal is z - alpha on the
/// diagonal and -beta off it, D-Lanczos (Saad, Alg. 6.17) gives
///   eta_m  = z - alpha_m - beta_m^2 / eta_{m-1},
///   zeta_m = (beta_m / eta_{m-1}) zeta_{m-1}      (zeta_1 = ||b||),
///   ||r_m|| = beta_{m+1} |zeta_m / eta_m|.
/// Im eta_m >= Im z > 0, so the recurrence cannot break down.
template<typename T>
__global__ void shift_kernel(const int k, const int pairs_count, const int step, const T tol,
                             const T *zr, const T *zi, const T *alpha, const T *beta_cur,
                             const T *beta_next, const T *bnorm, CocgPairs<T> pairs) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= pairs_count || pairs.done_step[p] != -1) {
    return;
  }
  const int e = p / k;
  const int j = p % k;
  const T b = beta_cur[j];
  const T pir = pairs.ieta_r[p]; // 1 / eta_{m-1}
  const T pii = pairs.ieta_i[p];

  T er = zr[e] - alpha[j];
  T ei = zi[e];
  T cr = pairs.zeta_r[p];
  T ci = pairs.zeta_i[p];
  if (step > 1) {
    const T b2 = b * b;
    er -= b2 * pir;
    ei -= b2 * pii;
    // zeta_m = beta_m * ieta_{m-1} * zeta_{m-1}
    const T tr = b * (pir * cr - pii * ci);
    const T ti = b * (pir * ci + pii * cr);
    cr = tr;
    ci = ti;
  }
  const T mag = hypot_(er, ei);
  const T ir = (er / mag) / mag;
  const T ii = (-ei / mag) / mag;
  const T res = beta_next[j] * (hypot_(cr, ci) / mag) / bnorm[j];

  pairs.eta_r[p] = er;
  pairs.eta_i[p] = ei;
  pairs.ieta_r[p] = ir;
  pairs.ieta_i[p] = ii;
  pairs.zeta_r[p] = cr;
  pairs.zeta_i[p] = ci;
  pairs.res[p] = res;
  if (res <= tol) {
    pairs.done_step[p] = step;
  }
}

/// Grid-stride over ne * k * n: the search-direction and solution updates.
template<typename T>
__global__ void update_kernel(const std::size_t n, const int k, const std::size_t total,
                              const int step, const T *v_cur, const T *beta_cur, CocgPairs<T> pairs,
                              T *pr, T *pi, T *xr, T *xi) {
  const std::size_t block = n * static_cast<std::size_t>(k);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t e = idx / block;
    const std::size_t rem = idx - e * block;
    const std::size_t j = rem / n;
    const std::size_t p = e * static_cast<std::size_t>(k) + j;
    const int done = pairs.done_step[p];
    if (done != -1 && done != step) {
      continue;
    }
    const T b = beta_cur[j];
    const T ur = v_cur[rem] + b * pr[idx];
    const T ui = b * pi[idx];
    const T cr = pairs.ieta_r[p];
    const T ci = pairs.ieta_i[p];
    const T qr = cr * ur - ci * ui;
    const T qi = cr * ui + ci * ur;
    pr[idx] = qr;
    pi[idx] = qi;
    const T zr = pairs.zeta_r[p];
    const T zi = pairs.zeta_i[p];
    xr[idx] += zr * qr - zi * qi;
    xi[idx] += zr * qi + zi * qr;
  }
}

/// Grid-stride over k * n, each thread walking the ne shifts: update_kernel's
/// P_e step, with zeta P_e weighted by c_ej and summed into out, not into X_e.
template<typename T>
__global__ void update_accumulate_kernel(const std::size_t n, const int k, const int ne,
                                         const int step, const T *v_cur, const T *beta_cur,
                                         CocgPairs<T> pairs, T *pr, T *pi, const T *cr_all,
                                         const T *ci_all, T *out) {
  const std::size_t block = n * static_cast<std::size_t>(k);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t rem = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       rem < block; rem += stride) {
    const std::size_t j = rem / n;
    const T v = v_cur[rem];
    const T b = beta_cur[j];
    T acc = T(0);
    for (int e = 0; e < ne; ++e) {
      const std::size_t p = static_cast<std::size_t>(e) * static_cast<std::size_t>(k) + j;
      const int done = pairs.done_step[p];
      if (done != -1 && done != step) {
        continue;
      }
      const std::size_t idx = static_cast<std::size_t>(e) * block + rem;
      const T ur = v + b * pr[idx];
      const T ui = b * pi[idx];
      const T er = pairs.ieta_r[p];
      const T ei = pairs.ieta_i[p];
      const T qr = er * ur - ei * ui;
      const T qi = er * ui + ei * ur;
      pr[idx] = qr;
      pi[idx] = qi;
      const T zr = pairs.zeta_r[p];
      const T zi = pairs.zeta_i[p];
      const T tr = zr * qr - zi * qi;
      const T ti = zr * qi + zi * qr;
      acc += cr_all[p] * tr - ci_all[p] * ti; // Re(c zeta q)
    }
    out[rem] += acc;
  }
}

unsigned int pair_blocks(const int pairs_count) {
  return static_cast<unsigned int>((pairs_count + static_cast<int>(kBlock) - 1) /
                                   static_cast<int>(kBlock));
}

} // namespace

template<typename T>
void cocg_start(const wwr::wwrStream_t stream, const int n, const int k, const int ne,
                const T *const d_B, T *const d_v_prev, T *const d_v_cur, T *const d_beta_cur,
                T *const d_bnorm, const CocgPairs<T> pairs) {
  if (n < 1 || k < 1 || ne < 1) {
    return;
  }
  start_columns_kernel<T><<<static_cast<unsigned int>(k), kBlock, 0, stream>>>(
      n, d_B, d_v_prev, d_v_cur, d_beta_cur, d_bnorm);
  const int count = ne * k;
  start_pairs_kernel<T><<<pair_blocks(count), kBlock, 0, stream>>>(k, count, d_bnorm, pairs);
}

template<typename T>
void cocg_lanczos(const wwr::wwrStream_t stream, const int n, const int k, const T *const d_v_prev,
                  const T *const d_v_cur, T *const d_w, const T *const d_beta_cur, T *const d_alpha,
                  T *const d_beta_next) {
  if (n < 1 || k < 1) {
    return;
  }
  lanczos_kernel<T><<<static_cast<unsigned int>(k), kBlock, 0, stream>>>(
      n, d_v_prev, d_v_cur, d_w, d_beta_cur, d_alpha, d_beta_next);
}

template<typename T>
void cocg_shift(const wwr::wwrStream_t stream, const int k, const int ne, const int step,
                const T tol, const T *const d_zr, const T *const d_zi, const T *const d_alpha,
                const T *const d_beta_cur, const T *const d_beta_next, const T *const d_bnorm,
                const CocgPairs<T> pairs) {
  if (k < 1 || ne < 1) {
    return;
  }
  const int count = ne * k;
  shift_kernel<T><<<pair_blocks(count), kBlock, 0, stream>>>(
      k, count, step, tol, d_zr, d_zi, d_alpha, d_beta_cur, d_beta_next, d_bnorm, pairs);
}

template<typename T>
void cocg_update(const wwr::wwrStream_t stream, const int n, const int k, const int ne,
                 const int step, const T *const d_v_cur, const T *const d_beta_cur,
                 const CocgPairs<T> pairs, T *const d_pr, T *const d_pi, T *const d_xr,
                 T *const d_xi) {
  if (n < 1 || k < 1 || ne < 1) {
    return;
  }
  const std::size_t total =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(k) * static_cast<std::size_t>(ne);
  update_kernel<T><<<blocks_for(total), kBlock, 0, stream>>>(static_cast<std::size_t>(n), k, total,
                                                             step, d_v_cur, d_beta_cur, pairs, d_pr,
                                                             d_pi, d_xr, d_xi);
}

template<typename T>
void cocg_update_accumulate(const wwr::wwrStream_t stream, const int n, const int k, const int ne,
                            const int step, const T *const d_v_cur, const T *const d_beta_cur,
                            const CocgPairs<T> pairs, T *const d_pr, T *const d_pi,
                            const T *const d_cr, const T *const d_ci, T *const d_out) {
  if (n < 1 || k < 1 || ne < 1) {
    return;
  }
  const std::size_t block = static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
  update_accumulate_kernel<T><<<blocks_for(block), kBlock, 0, stream>>>(
      static_cast<std::size_t>(n), k, ne, step, d_v_cur, d_beta_cur, pairs, d_pr, d_pi, d_cr, d_ci,
      d_out);
}

// One per supported type (float, double), matching shifted_cocg_bridge.h.
#define CLM_COCG_INSTANTIATE(T)                                                                    \
  template void cocg_start<T>(wwr::wwrStream_t, int, int, int, const T *, T *, T *, T *, T *,      \
                              CocgPairs<T>);                                                       \
  template void cocg_lanczos<T>(wwr::wwrStream_t, int, int, const T *, const T *, T *, const T *,  \
                                T *, T *);                                                         \
  template void cocg_shift<T>(wwr::wwrStream_t, int, int, int, T, const T *, const T *, const T *, \
                              const T *, const T *, const T *, CocgPairs<T>);                      \
  template void cocg_update<T>(wwr::wwrStream_t, int, int, int, int, const T *, const T *,         \
                               CocgPairs<T>, T *, T *, T *, T *);                                  \
  template void cocg_update_accumulate<T>(wwr::wwrStream_t, int, int, int, int, const T *,         \
                                          const T *, CocgPairs<T>, T *, T *, const T *, const T *, \
                                          T *);

CLM_COCG_INSTANTIATE(float)
CLM_COCG_INSTANTIATE(double)

#undef CLM_COCG_INSTANTIATE

} // namespace calaman::device
