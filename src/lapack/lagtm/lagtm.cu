// lagtm.cu
//
// The device half of calaman.lagtm: B := alpha * op(A) * X + beta * B for the
// general tridiagonal A(i+1,i) = dl(i), A(i,i) = d(i), A(i,i+1) = du(i), one
// thread per element of B. Row i of op(A) holds three entries:
//
//   N: dl(i-1), d(i), du(i)      -- A's own row i
//   T: du(i-1), d(i), dl(i)      -- A's column i
//   C: conj of T's three
//
// The thread accumulates in DLAGTM's order -- beta * b first, then the three
// alpha-scaled terms left to right -- so alpha, beta in {-1, 0, 1} round the
// way the reference does. beta == 0 never reads B and alpha == 0 never reads
// A or X, so a NaN there does not leak in, as in DLAGTM.
//
// Grid: rows along x in blocks of 4 warps (coalesced, column-major), columns
// along y with a stride loop so nrhs is not bounded by gridDim.y's 65535.
// Shared unchanged between both backends (compiled -x hip under HIP).
#include "lagtm_bridge.h"

#include "common/align_up.h"
#include "common/elem_ops.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

constexpr unsigned int kMaxGridY = 65535;

/// @brief [kernel] One thread per B(i,j): the update of that one element
template<typename T, typename R>
__global__ void lagtm_kernel(const Trans trans, const std::size_t n, const std::size_t nrhs,
                             const R alpha, const T *const dl, const T *const d,
                             const T *const du, const T *const x, const std::size_t ldx,
                             const R beta, T *const b, const std::size_t ldb) {
  using ops = elem_ops<T>;
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  // N reads row i as (dl, du); T and C read column i, i.e. (du, dl).
  const T *const before = trans == Trans::N ? dl : du;
  const T *const after = trans == Trans::N ? du : dl;
  const bool conj = trans == Trans::C;
  const auto coef = [conj](const T a) { return conj ? ops::conj(a) : a; };

  for (std::size_t j = blockIdx.y; j < nrhs; j += gridDim.y) {
    const T *const xj = x + j * ldx;
    T *const bij = b + i + j * ldb;
    T acc = beta == R{0} ? ops::from_real(R{0}) : ops::scale(*bij, beta);
    if (alpha != R{0}) {
      if (i > 0) {
        acc = ops::add(acc, ops::scale(ops::mul(coef(before[i - 1]), xj[i - 1]), alpha));
      }
      acc = ops::add(acc, ops::scale(ops::mul(coef(d[i]), xj[i]), alpha));
      if (i + 1 < n) {
        acc = ops::add(acc, ops::scale(ops::mul(coef(after[i]), xj[i + 1]), alpha));
      }
    }
    *bij = acc;
  }
}

} // namespace

template<typename T, typename R>
void lagtm(const wwr::wwrStream_t stream, const Trans trans, const std::size_t n,
           const std::size_t nrhs, const R alpha, const T *const d_dl, const T *const d_d,
           const T *const d_du, const T *const d_x, const std::size_t ldx, const R beta,
           T *const d_b, const std::size_t ldb) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const dim3 grid(idivup<std::size_t>(n, kBlockSize), nrhs < kMaxGridY ? nrhs : kMaxGridY);
  lagtm_kernel<T, R><<<grid, kBlockSize, 0, stream>>>(trans, n, nrhs, alpha, d_dl, d_d, d_du,
                                                       d_x, ldx, beta, d_b, ldb);
}

// One per supported type, matching lagtm_bridge.h and interface.cppm's `extern
// template` list.
#define CLM_LAGTM_INSTANTIATE(T, R)                                                                \
  template void lagtm<T, R>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, R, const T *,       \
                            const T *, const T *, const T *, std::size_t, R, T *, std::size_t);
CLM_LAGTM_INSTANTIATE(float, float)
CLM_LAGTM_INSTANTIATE(double, double)
CLM_LAGTM_INSTANTIATE(wwr::wwrFloatComplex, float)
CLM_LAGTM_INSTANTIATE(wwr::wwrDoubleComplex, double)
#undef CLM_LAGTM_INSTANTIATE

} // namespace calaman::device
