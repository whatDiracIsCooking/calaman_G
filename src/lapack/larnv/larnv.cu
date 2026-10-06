// larnv.cu
//
// The device-kernel half of calaman.larnv. The reference walks x in chunks of
// 64 outputs, each one ?laruv call of il2 draws (2*il for Box-Muller and every
// complex case) threading ISEED through. A call with no retry multiplies the
// seed by multiplier row il2-1 mod 2^48, so chunk c starts from
// seed * J^c (J the full-chunk multiplier): larnv_chunks_kernel runs every chunk
// in parallel from that jump-ahead seed.
//
// Only a single-precision draw can round to 1 and take ?laruv's retry, which
// shifts every later chunk's seed. larnv_finish_kernel (one block) scans for the
// first chunk where that happens, re-runs from it serially with the threaded
// seed -- the reference's own walk -- and writes the final ISEED. In double the
// largest draw is 1 - 2^-48, exact, so the scan is compiled out.
#include "larnv_bridge.h"

#include "common/elem_ops.cuh"
#include "lapack/laruv/laruv.cuh"
#include <complex.h>
#include <runtime.h>
#include <wrappers/math/math.cuh>

#include <cstddef>
#include <type_traits>

namespace calaman::device {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

constexpr unsigned int kFinishThreads = 256;
static_assert(kFinishThreads >= kLaruvMaxN, "laruv_block needs blockDim.x >= 128");

template<typename T>
using RealOf = typename elem_ops<T>::real_type;

template<typename T>
constexpr bool kIsComplex = !std::is_same_v<T, RealOf<T>>;

/// @brief ?laruv draws a chunk of @p il outputs consumes
template<typename T>
__device__ __forceinline__ int draws_for(const int idist, const int il) {
  return (kIsComplex<T> || idist == 3) ? 2 * il : il;
}

/// @brief Output @p k of a chunk from its draws @p u, as ?larnv computes it
template<typename T>
__device__ __forceinline__ T larnv_value(const int idist, const RealOf<T> *const u, const int k) {
  using R = RealOf<T>;
  constexpr R one = R(1);
  constexpr R two = R(2);
  constexpr R twopi = R(6.28318530717958647692528676655900576839);
  if constexpr (kIsComplex<T>) {
    const R a = u[2 * k];
    const R b = u[2 * k + 1];
    switch (idist) {
    case 1:
      return make_complex(a, b);
    case 2:
      return make_complex(two * a - one, two * b - one);
    default: {
      const R r = idist == 3 ? wwr::sqrt(-two * wwr::log(a)) : (idist == 4 ? wwr::sqrt(a) : one);
      const R t = twopi * b;
      return make_complex(r * wwr::cos(t), r * wwr::sin(t));
    }
    }
  } else {
    switch (idist) {
    case 1:
      return u[k];
    case 2:
      return two * u[k] - one;
    default:
      return wwr::sqrt(-two * wwr::log(u[2 * k])) * wwr::cos(twopi * u[2 * k + 1]);
    }
  }
}

/// @brief Chunk @p c's output count, min(64, n - 64c)
__device__ __forceinline__ int chunk_len(const std::size_t n, const unsigned int c) {
  const std::size_t rest = n - static_cast<std::size_t>(c) * kLarnvChunk;
  return rest < kLarnvChunk ? static_cast<int>(rest) : kLarnvChunk;
}

/// @brief [kernel] One block of kLaruvMaxN per chunk, from the jump-ahead seed
template<typename T>
__global__ void larnv_chunks_kernel(const int idist, const int *const iseed, const std::size_t n,
                                    T *const x) {
  __shared__ RealOf<T> u[kLaruvMaxN];
  __shared__ int seed_next[4]; // laruv_block's output; the finish kernel owns ISEED
  const unsigned int c = blockIdx.x;
  const int il = chunk_len(n, c);
  const int il2 = draws_for<T>(idist, il);
  const int seed0[4] = {iseed[0], iseed[1], iseed[2], iseed[3]};
  const unsigned long long jump = laruv_multiplier(draws_for<T>(idist, kLarnvChunk) - 1);
  int s[4];
  laruv_split(laruv_mulmod(laruv_join(seed0), laruv_powmod(jump, c)), s);
  laruv_block(s, il2, u, seed_next);
  __syncthreads();
  const int k = static_cast<int>(threadIdx.x);
  if (k < il) {
    x[static_cast<std::size_t>(c) * kLarnvChunk + k] = larnv_value<T>(idist, u, k);
  }
}

/// @brief [kernel] One block: redo the tail after a retry; write the final ISEED
template<typename T>
__global__ void larnv_finish_kernel(const int idist, int *const iseed, const std::size_t n,
                                    const unsigned int nchunks, T *const x) {
  using R = RealOf<T>;
  __shared__ R u[kLaruvMaxN];
  __shared__ int s_seed[4];
  __shared__ unsigned int s_first; // first chunk whose jump-ahead draws hit 1
  const unsigned int tid = threadIdx.x;
  const int seed0_limbs[4] = {iseed[0], iseed[1], iseed[2], iseed[3]};
  const unsigned long long seed0 = laruv_join(seed0_limbs);
  const unsigned long long jump = laruv_multiplier(draws_for<T>(idist, kLarnvChunk) - 1);
  if (tid == 0) {
    s_first = nchunks;
  }
  __syncthreads();
  if constexpr (std::is_same_v<R, float>) {
    const unsigned long long stride_jump = laruv_powmod(jump, blockDim.x);
    unsigned long long seed = laruv_mulmod(seed0, laruv_powmod(jump, tid));
    for (unsigned int c = tid; c < nchunks; c += blockDim.x) {
      const int il2 = draws_for<T>(idist, chunk_len(n, c));
      int s[4];
      laruv_split(seed, s);
      for (int i = 0; i < il2; ++i) {
        int it[4];
        if (laruv_draw<R>(i, s, it) == R(1)) {
          atomicMin(&s_first, c);
          break;
        }
      }
      seed = laruv_mulmod(seed, stride_jump);
    }
  }
  __syncthreads();
  const unsigned int first = s_first;
  unsigned long long final_seed = 0;
  if (first == nchunks) {
    const int il2_last = draws_for<T>(idist, chunk_len(n, nchunks - 1));
    final_seed = laruv_mulmod(laruv_mulmod(seed0, laruv_powmod(jump, nchunks - 1)),
                              laruv_multiplier(il2_last - 1));
  } else {
    if (tid == 0) {
      int s[4];
      laruv_split(laruv_mulmod(seed0, laruv_powmod(jump, first)), s);
      for (int k = 0; k < 4; ++k) {
        s_seed[k] = s[k];
      }
    }
    for (unsigned int c = first; c < nchunks; ++c) {
      __syncthreads();
      const int s[4] = {s_seed[0], s_seed[1], s_seed[2], s_seed[3]};
      const int il = chunk_len(n, c);
      // laruv_block synchronizes before thread il2-1 overwrites s_seed, so every
      // thread has read it above.
      laruv_block(s, draws_for<T>(idist, il), u, s_seed);
      __syncthreads();
      if (static_cast<int>(tid) < il) {
        x[static_cast<std::size_t>(c) * kLarnvChunk + tid] =
            larnv_value<T>(idist, u, static_cast<int>(tid));
      }
    }
    __syncthreads();
    final_seed = laruv_join(s_seed);
  }
  if (tid == 0) {
    int s[4];
    laruv_split(final_seed, s);
    for (int k = 0; k < 4; ++k) {
      iseed[k] = s[k];
    }
  }
}

} // namespace

template<typename T>
void larnv(const wwr::wwrStream_t stream, const int idist, int *const iseed, const std::size_t n,
           T *const x) {
  if (n < 1) {
    return;
  }
  const auto nchunks = static_cast<unsigned int>((n + kLarnvChunk - 1) / kLarnvChunk);
  larnv_chunks_kernel<T><<<nchunks, kLaruvMaxN, 0, stream>>>(idist, iseed, n, x);
  larnv_finish_kernel<T><<<1, kFinishThreads, 0, stream>>>(idist, iseed, n, nchunks, x);
}

// Matching interface.cppm's extern template list and instantiations.cpp's.
template void larnv<float>(wwr::wwrStream_t, int, int *, std::size_t, float *);
template void larnv<double>(wwr::wwrStream_t, int, int *, std::size_t, double *);
template void larnv<wwrFloatComplex>(wwr::wwrStream_t, int, int *, std::size_t,
                                     wwrFloatComplex *);
template void larnv<wwrDoubleComplex>(wwr::wwrStream_t, int, int *, std::size_t,
                                      wwrDoubleComplex *);

} // namespace calaman::device
