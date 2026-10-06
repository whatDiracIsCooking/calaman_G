// lanczos.cu
//
// The device-kernel half of calaman.lanczos: the per-step bookkeeping no BLAS
// call expresses -- writing alpha_j / beta_j into the dense projected matrix T,
// the breakdown guard, normalising v_{j+1} -- and the thick-restart arrowhead.
// The host driver (matvec, CGS2, syevd) lives in the module partitions and
// reaches these through the launchers declared in lanczos_bridge.h.
//
// Shared unchanged between both backends, like feast.cu: the .cu extension is
// all CMake needs (under HIP the CMakeLists forces -x hip). Machine epsilon is
// <cfloat>'s FLT/DBL_EPSILON: nvcc rejects std::numeric_limits in device code.
#include "lanczos_bridge.h"

#include "common/block_reduce.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace calaman::device {

namespace {

// 4 warps per block, matching feast/gebal: a power of two, as block_reduce's
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

__device__ __forceinline__ float abs_(const float x) { return fabsf(x); }
__device__ __forceinline__ double abs_(const double x) { return fabs(x); }
__device__ __forceinline__ float sqrt_(const float x) { return sqrtf(x); }
__device__ __forceinline__ double sqrt_(const double x) { return sqrt(x); }

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

__global__ void status_reset_kernel(LanczosStatus *status) {
  status->breakdown = 0;
  status->breakdown_step = -1;
  status->eig_info = 0;
}

/// One block: the T writes for step j, ||T_j||_F, and the breakdown verdict.
template<typename T>
__global__ void step_matrix_kernel(const int ncv, const int j, const T *alpha, const T *beta,
                                   T *Tm, LanczosStatus *status) {
  const std::size_t ld = static_cast<std::size_t>(ncv);
  const std::size_t jz = static_cast<std::size_t>(j);
  for (std::size_t i = jz + 2 + threadIdx.x; i < ld; i += blockDim.x) {
    Tm[i + jz * ld] = T(0);
    Tm[jz + i * ld] = T(0);
  }
  if (threadIdx.x == 0) {
    Tm[jz + jz * ld] = alpha[j];
  }
  // alpha_j must be visible to every thread before the block is read.
  __syncthreads();

  // ||T_j||_F over the leading (j+1) x (j+1) block, scaled by its max |entry|
  // so the squares cannot overflow. NaN wins the max and so trips the guard.
  const std::size_t m = jz + 1;
  T big = T(0);
  for (std::size_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
    big = max_nan(big, abs_(Tm[(idx % m) + (idx / m) * ld]));
  }
  big = block_reduce<kBlock>(big, MaxNanOp{});
  T sum = T(0);
  if (big > T(0)) {
    for (std::size_t idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
      const T x = Tm[(idx % m) + (idx / m) * ld] / big;
      sum += x * x;
    }
  }
  sum = block_reduce<kBlock>(sum, AddOp{});

  if (threadIdx.x == 0) {
    const T norm = big > T(0) ? big * sqrt_(sum) : big; // big is 0 or NaN otherwise
    const T b = beta[j];
    const bool trip = !(b > MachineEps<T>::kValue * norm);
    if (trip && status->breakdown == 0) {
      status->breakdown = 1;
      status->breakdown_step = j;
    }
    if (jz + 1 < ld) {
      const T coupling = trip ? T(0) : b;
      Tm[(jz + 1) + jz * ld] = coupling;
      Tm[jz + (jz + 1) * ld] = coupling;
    }
  }
}

/// Grid-stride: v /= beta_j, unless any step of this cycle has broken down.
template<typename T>
__global__ void step_normalise_kernel(const std::size_t n, const int j, const T *beta, T *v,
                                      const LanczosStatus *status) {
  if (status->breakdown != 0) {
    return;
  }
  const T b = beta[j];
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    v[i] = v[i] / b;
  }
}

/// Grid-stride over the ncv x k entries of columns 0..k-1, each mirrored into
/// its row: theta on the diagonal, beta_m * s_{m,c} in row k, zero elsewhere.
template<typename T>
__global__ void arrowhead_kernel(const int ncv, const int k, const T *theta, const T *s_last_row,
                                 const int inc, const T *beta_m, T *Tm) {
  const std::size_t ld = static_cast<std::size_t>(ncv);
  const std::size_t kz = static_cast<std::size_t>(k);
  const std::size_t total = ld * kz;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t r = idx % ld;
    const std::size_t c = idx / ld;
    T value = T(0);
    if (r == c) {
      value = theta[c];
    } else if (r == kz) {
      value = *beta_m * s_last_row[c * static_cast<std::size_t>(inc)];
    }
    Tm[r + c * ld] = value;
    Tm[c + r * ld] = value;
  }
}

} // namespace

void lanczos_status_reset(const wwr::wwrStream_t stream, LanczosStatus *const d_status) {
  status_reset_kernel<<<1, 1, 0, stream>>>(d_status);
}

template<typename T>
void lanczos_step(const wwr::wwrStream_t stream, const int n, const int ncv, const int j,
                  const T *const d_alpha, const T *const d_beta, T *const d_v_next, T *const d_T,
                  LanczosStatus *const d_status) {
  if (n < 1 || ncv < 1 || j < 0 || j >= ncv) {
    return;
  }
  step_matrix_kernel<T><<<1, kBlock, 0, stream>>>(ncv, j, d_alpha, d_beta, d_T, d_status);
  const std::size_t nz = static_cast<std::size_t>(n);
  step_normalise_kernel<T><<<blocks_for(nz), kBlock, 0, stream>>>(nz, j, d_beta, d_v_next,
                                                                  d_status);
}

template<typename T>
void lanczos_arrowhead(const wwr::wwrStream_t stream, const int ncv, const int k,
                       const T *const d_theta, const T *const d_s_last_row, const int inc,
                       const T *const d_beta_m, T *const d_T) {
  if (k < 1 || k >= ncv) {
    return;
  }
  const std::size_t total = static_cast<std::size_t>(ncv) * static_cast<std::size_t>(k);
  arrowhead_kernel<T><<<blocks_for(total), kBlock, 0, stream>>>(ncv, k, d_theta, d_s_last_row,
                                                                inc, d_beta_m, d_T);
}

// One per supported type (float, double), matching lanczos_bridge.h.
template void lanczos_step<float>(wwr::wwrStream_t, int, int, int, const float *, const float *,
                                  float *, float *, LanczosStatus *);
template void lanczos_step<double>(wwr::wwrStream_t, int, int, int, const double *,
                                   const double *, double *, double *, LanczosStatus *);

template void lanczos_arrowhead<float>(wwr::wwrStream_t, int, int, const float *, const float *,
                                       int, const float *, float *);
template void lanczos_arrowhead<double>(wwr::wwrStream_t, int, int, const double *,
                                        const double *, int, const double *, double *);

} // namespace calaman::device
