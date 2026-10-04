// lascl.cu
//
// The device-kernel half of calaman.lascl: multiply the leading m-by-n block of
// a column-major matrix in place by one scalar factor, A(i,j) <-- mul * A(i,j).
// Shared unchanged between both backends -- under CUDA the .cu extension is all
// CMake needs, under HIP this directory's CMakeLists.txt forces LANGUAGE CXX
// back on so clang compiles it with -x hip.
//
// The overflow-safe multiplier sequence lives entirely in the host wrapper
// (interface.cppm): it reduces cto/cfrom to a chain of scalar factors, each in
// range, and calls this launcher once per factor. The kernel itself knows
// nothing of that -- it is a plain per-element multiply, so it needs no machine
// constants (and so none of the nvcc-rejects-std::numeric_limits device trap).
//
// The scaling is one thread per element over a 2-D grid of 1-D blocks: the
// block's 4*WWR_WARP_SIZE threads run along x (matrix rows), and blockIdx.y
// names the column. Column-major storage makes a column contiguous, so a warp's
// consecutive rows are consecutive addresses -- a coalesced access. Rows need
// idivup(m, block) blocks in x (common/align_up.h); columns map one block each.
#include "lascl_bridge.h"

#include "common/align_up.h"
#include "runtime.h"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief [kernel] Multiply column-major A(i,j) by the scalar mul
///
/// The row index i comes from the 1-D block laid along x; the column j is
/// blockIdx.y. The grid is sized so every j is in range, so only i needs a
/// bound.
template<typename T>
__global__ void lascl_kernel(const T mul, T *const x, const std::size_t m,
                             const std::size_t ldx) {
  // The usual flattened thread index. i and j keep the builtins' unsigned int
  // and widen to size_t in the address below, where j * ldx -- the offset that
  // can exceed 32 bits -- is formed in 64-bit because ldx is size_t.
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;

  if (i >= m) {
    return;
  }

  x[i + j * ldx] *= mul;
}

} // namespace

template<typename T>
void lascl(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n, const T mul,
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

  lascl_kernel<T><<<grid, block, 0, stream>>>(mul, x, m, ldx);
}

// One per supported type, matching interface.cppm's extern template list and
// instantiations.cpp's -- all three lists cover the same types.
template void lascl<float>(wwr::wwrStream_t, std::size_t, std::size_t, float, float *, std::size_t);
template void lascl<double>(wwr::wwrStream_t, std::size_t, std::size_t, double, double *,
                            std::size_t);

} // namespace calaman::device
