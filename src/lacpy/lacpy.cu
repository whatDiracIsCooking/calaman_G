// lacpy.cu
//
// The device-kernel half of calaman.lacpy: the single per-element copy stage.
// Shared unchanged between both backends -- under CUDA the .cu extension is all
// CMake needs, under HIP this directory's CMakeLists.txt forces LANGUAGE CXX
// back on so clang compiles it with -x hip.
//
// The copy is one thread per element over a 2-D grid of 1-D blocks: the block's
// 4*WWR_WARP_SIZE threads run along x (matrix rows), and blockIdx.y names the
// column. Column-major storage makes a column contiguous, so a warp's
// consecutive rows are consecutive addresses -- a coalesced access. Rows need
// idivup(m, block) blocks in x (common/align_up.h); columns map one block each
// in y, so gridDim.y is n and no bound on the column index is needed.
//
// A triangular region still launches the full grid and skips the elements
// outside it: correct, and simplest. Mapping a packed triangular index range to
// (i, j) without the skipped threads is a possible later optimisation (README).
#include "lacpy_bridge.h"

#include "common/align_up.h"
#include "runtime.cuh"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief [kernel] Copy column-major A(i,j) -> B(i,j) for the in-region elements
///
/// The row index i comes from the 1-D block laid along x; the column j is
/// blockIdx.y. The grid is sized so every j is in range, so only i needs a bound.
///
/// @tparam R the copied region (Region::U/L/A), a non-type template argument so
///         the region test below resolves at compile time.
template<typename T, Region R>
__global__ void lacpy_kernel(const T *const a, T *const b, const std::size_t m,
                             const std::size_t lda, const std::size_t ldb) {
  // The usual flattened thread index. i and j keep the builtins' unsigned int
  // and widen to size_t in the address below, where j * ldb -- the offset that
  // can exceed 32 bits -- is formed in 64-bit because ldb is size_t.
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;

  if (i >= m) {
    return;
  }

  // Upper is the diagonal and above (row <= col); lower is the diagonal and
  // below (row >= col). R is a template argument, so this is an if constexpr --
  // each specialization is branchless but for the diagonal, where neighbouring
  // threads disagree on in_region.
  bool in_region = true;
  if constexpr (R == Region::U) {
    in_region = i <= j;
  } else if constexpr (R == Region::L) {
    in_region = i >= j;
  }

  if (in_region) {
    b[i + j * ldb] = a[i + j * lda];
  }
}

} // namespace

template<typename T>
void lacpy(const wwr::wwrStream_t stream, const Region region, const std::size_t m,
           const std::size_t n, const T *a, const std::size_t lda, T *b, const std::size_t ldb) {
  if (m < 1 || n < 1) {
    return;
  }

  // 4 warps per block, laid along the rows. WWR_WARP_SIZE (runtime.cuh, carried
  // as a define by wwr.device) is a configure-time value -- 32 by default, so
  // 128 unless a CDNA build sets 64 and makes it 256. unsigned int is dim3's
  // own field type.
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;

  // The row-block count is computed in size_t (idivup<std::size_t>) so a tall
  // matrix cannot overflow a 32-bit intermediate; it and n then narrow to dim3's
  // unsigned int fields. One block per column in y, whose 65535 bound maps
  // matrices up to that many columns -- wider than any first-cut caller (the
  // triangular-packing optimisation in the README revisits it).
  const dim3 grid(idivup<std::size_t>(m, kBlockSize), n);
  const dim3 block(kBlockSize);

  // Dispatch the runtime region to the matching compile-time specialization.
  switch (region) {
  case Region::U:
    lacpy_kernel<T, Region::U><<<grid, block, 0, stream>>>(a, b, m, lda, ldb);
    break;
  case Region::L:
    lacpy_kernel<T, Region::L><<<grid, block, 0, stream>>>(a, b, m, lda, ldb);
    break;
  default:
    lacpy_kernel<T, Region::A><<<grid, block, 0, stream>>>(a, b, m, lda, ldb);
    break;
  }
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's -- all three lists cover the same types.
template void lacpy<float>(wwr::wwrStream_t, Region, std::size_t, std::size_t, const float *,
                           std::size_t, float *, std::size_t);
template void lacpy<double>(wwr::wwrStream_t, Region, std::size_t, std::size_t, const double *,
                            std::size_t, double *, std::size_t);

} // namespace calaman::device
