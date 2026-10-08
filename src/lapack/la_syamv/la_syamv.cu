// la_syamv.cu
//
// The device half of calaman.la_syamv and calaman.la_heamv: y(i) := alpha *
// sum_j |A(i,j)| * |x(j)| + beta * |y(i)|, then the symbolic-zero nudge, one
// thread per y(i), for an n-by-n A held in the triangle uplo names. The
// per-row body and its no-FMA rounding are la_amv_row (la_amv.cuh); this file
// supplies only the triangle: A(i, j) is read from the stored entry at
// (min(i,j), max(i,j)) for Uplo::U and (max, min) for Uplo::L, so the other
// triangle is never touched. No Hermitian flag: CABS1(conj(a)) == CABS1(a).
//
// For j > i (U) or j <= i (L) a thread reads row i of the stored triangle,
// coalesced across the block; the other half walks column i, strided.
#include "la_syamv_bridge.h"

#include "common/align_up.h"
#include "lapack/la_geamv/la_amv.cuh"

#include <cstddef>

#if defined(__HIP__)
#pragma clang fp contract(off)
#endif

namespace calaman::device {

namespace {

/// @brief [kernel] One thread per y(i): ?LA_SYAMV's loop body for that i
template<typename T, typename R>
__global__ void la_syamv_kernel(const Uplo uplo, const std::size_t n, const R alpha,
                                const T *const a, const std::size_t lda, const T *const x,
                                const std::ptrdiff_t incx, const R beta, R *const y,
                                const std::ptrdiff_t incy, const R safe1) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  // Column i of the stored triangle, and row i of it (stepping lda).
  const T *const col_i = a + i * lda;
  const T *const row_i = a + i;
  const bool upper = uplo == Uplo::U;
  la_amv_row<T, R>(y + static_cast<std::ptrdiff_t>(i) * incy, alpha, beta, safe1, x, incx, 0, n,
                   [=](const std::size_t j) {
                     // U stores (min, max): A(j, i) for j <= i, A(i, j) above.
                     // L stores (max, min): A(i, j) for j <= i, A(j, i) below.
                     return (j <= i) == upper ? col_i[j] : row_i[j * lda];
                   });
}

} // namespace

template<typename T, typename R>
void la_syamv(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n, const R alpha,
              const T *const a, const std::size_t lda, const T *const x, const std::ptrdiff_t incx,
              const R beta, R *const y, const std::ptrdiff_t incy, const R safe1) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const dim3 grid(idivup<std::size_t>(n, kBlockSize));
  la_syamv_kernel<T, R>
      <<<grid, kBlockSize, 0, stream>>>(uplo, n, alpha, a, lda, x, incx, beta, y, incy, safe1);
}

// One per supported type, matching la_syamv_bridge.h and the `extern template`
// lists of la_syamv's and la_heamv's interface.cppm.
#define CLM_LA_SYAMV_INSTANTIATE(T, R)                                                             \
  template void la_syamv<T, R>(wwr::wwrStream_t, Uplo, std::size_t, R, const T *, std::size_t,     \
                               const T *, std::ptrdiff_t, R, R *, std::ptrdiff_t, R);
CLM_LA_SYAMV_INSTANTIATE(float, float)
CLM_LA_SYAMV_INSTANTIATE(double, double)
CLM_LA_SYAMV_INSTANTIATE(wwr::wwrFloatComplex, float)
CLM_LA_SYAMV_INSTANTIATE(wwr::wwrDoubleComplex, double)
#undef CLM_LA_SYAMV_INSTANTIATE

} // namespace calaman::device
