// lacp2.cu
//
// The device-kernel half of calaman.lacp2: lacpy.cu's per-element copy with a
// widening store, B(i,j) <- (A(i,j), 0). One thread per element over a 2-D grid
// of 1-D blocks: 4*WWR_WARP_SIZE threads along the rows (x), blockIdx.y names
// the column, so a warp's accesses are coalesced. The complex element is built
// with make_complex (common/elem_ops.cuh), never by .x/.y, which is not portable.
#include "lacp2_bridge.h"

#include "common/align_up.h"
#include "common/elem_ops.cuh"
#include <complex.h>
#include <runtime.h>

#include <cstddef>

namespace calaman::device {

using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

/// @brief [kernel] B(i,j) <- (A(i,j), 0) for the in-region elements
///
/// @tparam R the copied region (Region::U/L/A), resolved at compile time.
template<typename ComplexT, typename RealT, Region R>
__global__ void lacp2_kernel(const RealT *const a, ComplexT *const b, const std::size_t m,
                             const std::size_t lda, const std::size_t ldb) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;
  if (i >= m) {
    return;
  }
  bool in_region = true;
  if constexpr (R == Region::U) {
    in_region = i <= j;
  } else if constexpr (R == Region::L) {
    in_region = i >= j;
  }
  if (in_region) {
    b[i + j * ldb] = make_complex(a[i + j * lda], RealT(0));
  }
}

} // namespace

template<typename ComplexT, typename RealT>
void lacp2(const wwr::wwrStream_t stream, const Region region, const std::size_t m,
           const std::size_t n, const RealT *a, const std::size_t lda, ComplexT *b,
           const std::size_t ldb) {
  if (m < 1 || n < 1) {
    return;
  }
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  // One block per column in y: gridDim.y bounds n at 65535, as lacpy's does.
  const dim3 grid(idivup<std::size_t>(m, kBlockSize), n);
  const dim3 block(kBlockSize);
  switch (region) {
  case Region::U:
    lacp2_kernel<ComplexT, RealT, Region::U><<<grid, block, 0, stream>>>(a, b, m, lda, ldb);
    break;
  case Region::L:
    lacp2_kernel<ComplexT, RealT, Region::L><<<grid, block, 0, stream>>>(a, b, m, lda, ldb);
    break;
  default:
    lacp2_kernel<ComplexT, RealT, Region::A><<<grid, block, 0, stream>>>(a, b, m, lda, ldb);
    break;
  }
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's.
template void lacp2<wwrFloatComplex, float>(wwr::wwrStream_t, Region, std::size_t, std::size_t,
                                            const float *, std::size_t, wwrFloatComplex *,
                                            std::size_t);
template void lacp2<wwrDoubleComplex, double>(wwr::wwrStream_t, Region, std::size_t,
                                              std::size_t, const double *, std::size_t,
                                              wwrDoubleComplex *, std::size_t);

} // namespace calaman::device
