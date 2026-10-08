// la_geamv.cu
//
// The device half of calaman.la_geamv: y(i) := alpha * sum_j |op(A)(i,j)| *
// |x(j)| + beta * |y(i)|, then the symbolic-zero nudge, one thread per y(i).
// The per-row body, and the no-FMA rounding that makes it bitwise the
// reference's, is la_amv_row in la_amv.cuh; this file supplies the dense
// storage: j over all of op(A)'s columns, op(A)(i, j) = A(i, j) or A(j, i).
//
// Trans::N walks row i of A (coalesced across the block's rows); T and C walk
// column i -- identical, as |conj(a)| == |a|. That walk is strided across
// threads; a block-cooperative transpose would coalesce it, but would also
// have to keep the per-thread sequential sum order the bitwise oracle needs.
#include "la_geamv_bridge.h"

#include "common/align_up.h"
#include "lapack/la_geamv/la_amv.cuh"

#include <cstddef>

#if defined(__HIP__)
#pragma clang fp contract(off)
#endif

namespace calaman::device {

namespace {

/// @brief [kernel] One thread per y(i): ?LA_GEAMV's loop body for that i
template<typename T, typename R>
__global__ void la_geamv_kernel(const Trans trans, const std::size_t lenx, const std::size_t leny,
                                const R alpha, const T *const a, const std::size_t lda,
                                const T *const x, const std::ptrdiff_t incx, const R beta,
                                R *const y, const std::ptrdiff_t incy, const R safe1) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= leny) {
    return;
  }
  // op(A)(i, j): A(i, j) for N, A(j, i) for T and C.
  const bool by_row = trans == Trans::N;
  const T *const ai = by_row ? a + i : a + i * lda;
  const std::size_t step = by_row ? lda : 1;
  la_amv_row<T, R>(y + static_cast<std::ptrdiff_t>(i) * incy, alpha, beta, safe1, x, incx, 0, lenx,
                   [=](const std::size_t j) { return ai[j * step]; });
}

} // namespace

template<typename T, typename R>
void la_geamv(const wwr::wwrStream_t stream, const Trans trans, const std::size_t lenx,
              const std::size_t leny, const R alpha, const T *const a, const std::size_t lda,
              const T *const x, const std::ptrdiff_t incx, const R beta, R *const y,
              const std::ptrdiff_t incy, const R safe1) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const dim3 grid(idivup<std::size_t>(leny, kBlockSize));
  la_geamv_kernel<T, R><<<grid, kBlockSize, 0, stream>>>(trans, lenx, leny, alpha, a, lda, x, incx,
                                                         beta, y, incy, safe1);
}

// One per supported type, matching la_geamv_bridge.h and interface.cppm's
// `extern template` list.
#define CLM_LA_GEAMV_INSTANTIATE(T, R)                                                             \
  template void la_geamv<T, R>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, R, const T *,    \
                               std::size_t, const T *, std::ptrdiff_t, R, R *, std::ptrdiff_t, R);
CLM_LA_GEAMV_INSTANTIATE(float, float)
CLM_LA_GEAMV_INSTANTIATE(double, double)
CLM_LA_GEAMV_INSTANTIATE(wwr::wwrFloatComplex, float)
CLM_LA_GEAMV_INSTANTIATE(wwr::wwrDoubleComplex, double)
#undef CLM_LA_GEAMV_INSTANTIATE

} // namespace calaman::device
