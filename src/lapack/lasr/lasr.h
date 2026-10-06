/**
 * @file lasr.h
 * @brief lasr_block: ?lasr's sequence of plane rotations, applied by one block
 *
 * The block-cooperative device form of ?lasr, for a kernel that applies the
 * rotations from inside its own launch (?steqr updating Z). The chain runs
 * sequentially along the rotated dimension; the other dimension is split across
 * the block's threads (any blockDim shape), and each thread carries its lines
 * through the whole chain, so no barrier is needed between rotations.
 *
 * Every thread of the block must call lasr_block: it opens and closes with
 * __syncthreads(). Device code only; the stream-level entry point is the module
 * calaman.lasr. Downstream .cu files reach it as "lapack/lasr/lasr.h" by
 * linking the INTERFACE target calaman::lasr::header.
 *
 * Usage (DSTEQR's DLASR('R', 'V', 'B', N, MM, WORK(L), WORK(N-1+L), Z(1,L),
 * LDZ), 0-based):
 *   #include "lapack/lasr/lasr.h"
 *   calaman::lasr_block(Side::R, Pivot::V, Direct::B, n, mm, work + l,
 *                       work + n - 1 + l, Z + l * ldz, ldz);
 */

#pragma once

#include "common/enums.h"

#include <cstddef>

namespace calaman {

/// @brief Apply the k-1 rotations (c[j], s[j]) to one line x of length @p k
///
/// x(i) is x[i * inc]. Rotation j acts on the pair (p, q) @p pivot selects as
/// x(p) <- c x(p) + s x(q), x(q) <- c x(q) - s x(p); an identity rotation
/// (c == 1, s == 0) is skipped, as DLASR does. One thread, no barrier.
template<typename T>
__device__ void lasr_line(const Pivot pivot, const Direct direct, const std::size_t k,
                          const T *const c, const T *const s, T *const x, const std::size_t inc) {
  for (std::size_t r = 0; r + 1 < k; ++r) {
    const std::size_t j = direct == Direct::F ? r : k - 2 - r;
    const T ct = c[j];
    const T st = s[j];
    if (ct == T{1} && st == T{0}) {
      continue;
    }
    const std::size_t p = pivot == Pivot::T ? 0 : j;
    const std::size_t q = pivot == Pivot::B ? k - 1 : j + 1;
    const T xp = x[p * inc];
    const T xq = x[q * inc];
    x[q * inc] = ct * xq - st * xp;
    x[p * inc] = st * xq + ct * xp;
  }
}

/// @brief ?lasr on the m-by-n column-major A, by every thread of the block
///
/// Side::L applies A <- P A with P of order m (rotations c, s of length m-1);
/// Side::R applies A <- A P^T with P of order n (length n-1). Opens with a
/// barrier, so inputs written by other threads beforehand are seen, and closes
/// with one, so the update is visible block-wide on return.
///
/// @tparam T Real element type (float, double)
/// @param c, s Cosines and sines of the rotations, in plane order
/// @param A    Column-major matrix, leading dimension @p lda, updated in place
template<typename T>
__device__ void lasr_block(const Side side, const Pivot pivot, const Direct direct,
                           const std::size_t m, const std::size_t n, const T *const c,
                           const T *const s, T *const A, const std::size_t lda) {
  const std::size_t bx = blockDim.x;
  const std::size_t by = blockDim.y;
  const std::size_t bz = blockDim.z;
  const std::size_t tx = threadIdx.x;
  const std::size_t ty = threadIdx.y;
  const std::size_t tz = threadIdx.z;
  const std::size_t tid = tx + bx * (ty + by * tz);
  const std::size_t nthreads = bx * by * bz;
  __syncthreads();
  if (side == Side::L) {
    // Column i is untouched by every other column's chain: one thread per column.
    for (std::size_t i = tid; i < n; i += nthreads) {
      lasr_line(pivot, direct, m, c, s, A + i * lda, std::size_t{1});
    }
  } else {
    for (std::size_t i = tid; i < m; i += nthreads) {
      lasr_line(pivot, direct, n, c, s, A + i, lda);
    }
  }
  __syncthreads();
}

} // namespace calaman
