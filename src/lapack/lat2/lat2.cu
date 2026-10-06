// lat2.cu
//
// The device-kernel half of calaman.lat2: calaman.lag2's kernel restricted to
// one triangle. One thread per element over lacpy.cu's 2-D grid
// (4*WWR_WARP_SIZE threads along the rows, blockIdx.y names the column); the
// in-triangle threads convert through lag2_convert (lapack/lag2/lag2.cuh), the
// conversion and overflow check calaman.lag2 uses. Every overflowing thread
// stores the same 1 to *info, so the race is benign.
#include "lat2_bridge.h"

#include "common/align_up.h"
#include "lapack/lag2/lag2.cuh"
#include <complex.h>
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

/// @brief [kernel] SA(i,j) <- A(i,j) narrowed, over triangle U; *info <- 1 on overflow
template<typename From, typename To, Uplo U>
__global__ void lat2_kernel(const From *const a, To *const sa, const std::size_t n,
                            const std::size_t lda, const std::size_t ldsa, int *const info) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;
  if (i >= n) {
    return;
  }
  const bool in_triangle = U == Uplo::U ? i <= j : i >= j;
  if (in_triangle && !lag2_convert(a[i + j * lda], sa[i + j * ldsa])) {
    *info = 1;
  }
}

} // namespace

template<typename From, typename To>
void lat2(const wwr::wwrStream_t stream, const Uplo uplo, const std::size_t n, const From *a,
          const std::size_t lda, To *sa, const std::size_t ldsa, int *info) {
  if (n < 1) {
    return;
  }
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  // One block per column in y: gridDim.y bounds n at 65535, as lacpy's does.
  const dim3 grid(idivup<std::size_t>(n, kBlockSize), n);
  const dim3 block(kBlockSize);
  if (uplo == Uplo::U) {
    lat2_kernel<From, To, Uplo::U><<<grid, block, 0, stream>>>(a, sa, n, lda, ldsa, info);
  } else {
    lat2_kernel<From, To, Uplo::L><<<grid, block, 0, stream>>>(a, sa, n, lda, ldsa, info);
  }
}

// Matching interface.cppm's extern template list and instantiations.cpp's.
template void lat2<double, float>(wwr::wwrStream_t, Uplo, std::size_t, const double *,
                                  std::size_t, float *, std::size_t, int *);
template void lat2<wwrDoubleComplex, wwrFloatComplex>(wwr::wwrStream_t, Uplo, std::size_t,
                                                      const wwrDoubleComplex *, std::size_t,
                                                      wwrFloatComplex *, std::size_t, int *);

} // namespace calaman::device
