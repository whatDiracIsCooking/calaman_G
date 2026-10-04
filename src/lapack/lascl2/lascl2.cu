// lascl2.cu
//
// The device-kernel half of calaman.lascl2: the single per-element scaling
// stage x(i,j) <-- d(i) * x(i,j). Shared unchanged between both backends --
// under CUDA the .cu extension is all CMake needs, under HIP this directory's
// CMakeLists.txt forces LANGUAGE CXX back on so clang compiles it with -x hip.
//
// The scaling is one thread per element over a 2-D grid of 1-D blocks: the
// block's 4*WWR_WARP_SIZE threads run along x (matrix rows), and blockIdx.y
// names the column. Column-major storage makes a column contiguous, so a warp's
// consecutive rows are consecutive addresses -- a coalesced access, and the d(i)
// read is the same consecutive-row pattern. Rows need idivup(m, block) blocks in
// x (common/align_up.h); columns map one block each in y, so gridDim.y is n and
// no bound on the column index is needed.
#include "lascl2_bridge.h"

#include "common/align_up.h"
#include "runtime.h"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief [kernel] Scale column-major x(i,j) by the diagonal entry d(i)
///
/// The row index i comes from the 1-D block laid along x; the column j is
/// blockIdx.y. The grid is sized so every j is in range, so only i needs a
/// bound. d is the length-m diagonal vector, indexed by row.
template<typename T>
__global__ void lascl2_kernel(const T *const d, T *const x, const std::size_t m,
                              const std::size_t ldx) {
  // The usual flattened thread index. i and j keep the builtins' unsigned int
  // and widen to size_t in the address below, where j * ldx -- the offset that
  // can exceed 32 bits -- is formed in 64-bit because ldx is size_t.
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;

  if (i >= m) {
    return;
  }

  x[i + j * ldx] *= d[i];
}

} // namespace

template<typename T>
void lascl2(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n, const T *d,
            T *x, const std::size_t ldx) {
  if (m < 1 || n < 1) {
    return;
  }

  // 4 warps per block, laid along the rows. WWR_WARP_SIZE (runtime.h, carried
  // as a define by wwr.device) is a configure-time value -- 32 by default, so
  // 128 unless a CDNA build sets 64 and makes it 256. unsigned int is dim3's
  // own field type.
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;

  // The row-block count is computed in size_t (idivup<std::size_t>) so a tall
  // matrix cannot overflow a 32-bit intermediate; it and n then narrow to dim3's
  // unsigned int fields. One block per column in y, whose 65535 bound maps
  // matrices up to that many columns -- wider than any first-cut caller.
  const dim3 grid(idivup<std::size_t>(m, kBlockSize), n);
  const dim3 block(kBlockSize);

  lascl2_kernel<T><<<grid, block, 0, stream>>>(d, x, m, ldx);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's -- all three lists cover the same types.
template void lascl2<float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *, float *,
                            std::size_t);
template void lascl2<double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *, double *,
                             std::size_t);

} // namespace calaman::device
