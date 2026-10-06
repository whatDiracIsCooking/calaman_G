// lag2.cu
//
// The device-kernel half of calaman.lag2: one thread per element over a 2-D
// grid of 1-D blocks (lacpy.cu's layout -- 4*WWR_WARP_SIZE threads along the
// rows, blockIdx.y names the column), each converting through lag2_convert
// (lag2.cuh). Every overflowing thread stores the same 1 to *info, so the race
// is benign.
#include "lag2_bridge.h"

#include "common/align_up.h"
#include "lapack/lag2/lag2.cuh"
#include <complex.h>
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

/// @brief [kernel] SA(i,j) <- A(i,j) converted; *info <- 1 on any overflow
template<typename From, typename To>
__global__ void lag2_kernel(const From *const a, To *const sa, const std::size_t m,
                            const std::size_t lda, const std::size_t ldsa, int *const info) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;
  if (i >= m) {
    return;
  }
  if (!lag2_convert(a[i + j * lda], sa[i + j * ldsa])) {
    *info = 1;
  }
}

} // namespace

template<typename From, typename To>
void lag2(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n,
          const From *a, const std::size_t lda, To *sa, const std::size_t ldsa, int *info) {
  if (m < 1 || n < 1) {
    return;
  }
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  // One block per column in y: gridDim.y bounds n at 65535, as lacpy's does.
  const dim3 grid(idivup<std::size_t>(m, kBlockSize), n);
  const dim3 block(kBlockSize);
  lag2_kernel<From, To><<<grid, block, 0, stream>>>(a, sa, m, lda, ldsa, info);
}

// Matching interface.cppm's extern template list and instantiations.cpp's.
template void lag2<double, float>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                  std::size_t, float *, std::size_t, int *);
template void lag2<float, double>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                  std::size_t, double *, std::size_t, int *);
template void lag2<wwrDoubleComplex, wwrFloatComplex>(wwr::wwrStream_t, std::size_t, std::size_t,
                                                      const wwrDoubleComplex *, std::size_t,
                                                      wwrFloatComplex *, std::size_t, int *);
template void lag2<wwrFloatComplex, wwrDoubleComplex>(wwr::wwrStream_t, std::size_t, std::size_t,
                                                      const wwrFloatComplex *, std::size_t,
                                                      wwrDoubleComplex *, std::size_t, int *);

} // namespace calaman::device
