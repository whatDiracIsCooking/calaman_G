// la_gbamv.cu
//
// The device half of calaman.la_gbamv: y(i) := alpha * sum_j |op(A)(i,j)| *
// |x(j)| + beta * |y(i)|, then the symbolic-zero nudge, one thread per y(i),
// for A in LAPACK band storage, A(r, c) = AB(ku + r - c, c) (0-based). The
// per-row body and its no-FMA rounding are la_amv_row (la_amv.cuh); this file
// supplies only the band: j over the band's columns (N) or rows (T/C) at row
// (column) i, ascending, so AB's off-band corners and padding are never read.
//
// Both walks step ldab - 1 (N) or 1 (T/C) through AB within a thread, and
// ldab across threads; neither is coalesced. Fine until a caller needs speed.
#include "la_gbamv_bridge.h"

#include "common/align_up.h"
#include "lapack/la_geamv/la_amv.cuh"

#include <cstddef>

#if defined(__HIP__)
#pragma clang fp contract(off)
#endif

namespace calaman::device {

namespace {

/// @brief [kernel] One thread per y(i): the band slice of ?LA_GBAMV's loop body
template<typename T, typename R>
__global__ void la_gbamv_kernel(const Trans trans, const std::size_t m, const std::size_t n,
                                const std::size_t kl, const std::size_t ku, const R alpha,
                                const T *const ab, const std::size_t ldab, const T *const x,
                                const std::ptrdiff_t incx, const R beta, R *const y,
                                const std::ptrdiff_t incy, const R safe1) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  R *const yi = y + static_cast<std::ptrdiff_t>(i) * incy;
  if (trans == Trans::N) {
    if (i >= m) {
      return;
    }
    // Row i of A: columns i-kl .. i+ku, clipped; A(i, j) = AB(ku + i - j, j).
    const std::size_t j0 = i > kl ? i - kl : 0;
    const std::size_t j1 = i + ku + 1 < n ? i + ku + 1 : n;
    la_amv_row<T, R>(yi, alpha, beta, safe1, x, incx, j0, j1,
                     [=](const std::size_t j) { return ab[(ku + i - j) + j * ldab]; });
  } else {
    if (i >= n) {
      return;
    }
    // Column i of A: rows i-ku .. i+kl, clipped; A(j, i) = AB(ku + j - i, i).
    const T *const abi = ab + i * ldab + ku - i;
    const std::size_t j0 = i > ku ? i - ku : 0;
    const std::size_t j1 = i + kl + 1 < m ? i + kl + 1 : m;
    la_amv_row<T, R>(yi, alpha, beta, safe1, x, incx, j0, j1,
                     [=](const std::size_t j) { return abi[j]; });
  }
}

} // namespace

template<typename T, typename R>
void la_gbamv(const wwr::wwrStream_t stream, const Trans trans, const std::size_t m,
              const std::size_t n, const std::size_t kl, const std::size_t ku, const R alpha,
              const T *const ab, const std::size_t ldab, const T *const x,
              const std::ptrdiff_t incx, const R beta, R *const y, const std::ptrdiff_t incy,
              const R safe1) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const std::size_t leny = trans == Trans::N ? m : n;
  const dim3 grid(idivup<std::size_t>(leny, kBlockSize));
  la_gbamv_kernel<T, R><<<grid, kBlockSize, 0, stream>>>(trans, m, n, kl, ku, alpha, ab, ldab, x,
                                                         incx, beta, y, incy, safe1);
}

// One per supported type, matching la_gbamv_bridge.h and interface.cppm's
// `extern template` list.
#define CLM_LA_GBAMV_INSTANTIATE(T, R)                                                             \
  template void la_gbamv<T, R>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, std::size_t,     \
                               std::size_t, R, const T *, std::size_t, const T *, std::ptrdiff_t,  \
                               R, R *, std::ptrdiff_t, R);
CLM_LA_GBAMV_INSTANTIATE(float, float)
CLM_LA_GBAMV_INSTANTIATE(double, double)
CLM_LA_GBAMV_INSTANTIATE(wwr::wwrFloatComplex, float)
CLM_LA_GBAMV_INSTANTIATE(wwr::wwrDoubleComplex, double)
#undef CLM_LA_GBAMV_INSTANTIATE

} // namespace calaman::device
