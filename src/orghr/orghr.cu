// orghr.cu
//
// The device-kernel half of calaman.orghr: the ?orghr-specific reflector
// shuffle that precedes the ?orgqr call. Shared unchanged between both backends
// -- under CUDA the .cu extension is all CMake needs, under HIP this directory's
// CMakeLists.txt forces LANGUAGE CXX back on so clang compiles it with -x hip.
//
// DORGHR (before its DORGQR call) shifts the gehrd reflector columns one place
// to the right over the window [ilo, ihi], zeros the parts above each shifted
// subdiagonal, and turns the leading ilo and trailing n-ihi rows/columns into
// the unit matrix. Done in place that right-to-left column move aliases (column
// j-1 is both a source and a destination), so instead this reads an untouched
// snapshot a_in and writes a_out: every output element is a pure gather, so one
// thread per element over the whole n-by-n grid is race-free in a single launch.
//
// The grid mirrors laset's: a 2-D grid of 1-D blocks, 4*WWR_WARP_SIZE threads
// along x (matrix rows, contiguous in column-major storage -- coalesced) and
// blockIdx.y naming the column, so only the row index needs a bound.
#include "orghr_bridge.h"

#include "common/align_up.h"
#include "runtime.h"

#include <cstddef>

namespace calaman::device {

namespace {

/// @brief [kernel] Write a_out(i,j) by gathering from the a_in snapshot
///
/// Reproduces DORGHR's pre-orgqr loops (1-based @p ilo / @p ihi) element by
/// element. In 0-based terms: the window is columns [ilo, ihi-1] and rows
/// [ilo, ihi-1]. A window column j takes a_in(i, j-1) for i in (j, ihi-1], is 1
/// on the diagonal and 0 elsewhere; every column outside the window is an
/// identity column. The row i comes from the 1-D block along x, the column j is
/// blockIdx.y; the grid covers every j, so only i needs the bound.
template<typename T>
__global__ void orghr_prep_kernel(const T *const a_in, T *const a_out, const std::size_t n,
                                   const int ilo0, const int ihi0, const std::size_t lda) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;

  if (i >= n) {
    return;
  }

  const int ii = static_cast<int>(i);
  const int jj = static_cast<int>(j);

  T value = T{0};
  if (jj > ilo0 && jj <= ihi0) {
    // A shifted window column: the subdiagonal tail moves over from j-1, the
    // diagonal becomes 1, everything above it 0.
    if (ii == jj) {
      value = T{1};
    } else if (ii > jj && ii <= ihi0) {
      value = a_in[static_cast<std::size_t>(ii) + static_cast<std::size_t>(jj - 1) * lda];
    }
  } else {
    // Outside the window (the leading ilo and trailing n-ihi columns): an
    // identity column.
    value = (ii == jj) ? T{1} : T{0};
  }

  a_out[static_cast<std::size_t>(ii) + static_cast<std::size_t>(jj) * lda] = value;
}

} // namespace

template<typename T>
void orghr_prep(const wwr::wwrStream_t stream, const std::size_t n, const int ilo, const int ihi,
                const T *const a_in, T *const a_out, const std::size_t lda) {
  if (n == 0) {
    return;
  }

  // 4 warps per block along the rows, one block per column in y -- laset's
  // layout; see laset.cu for the coalescing and the 65535 y-bound.
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const dim3 grid(idivup<std::size_t>(n, kBlockSize), static_cast<unsigned int>(n));
  const dim3 block(kBlockSize);

  // 1-based ilo/ihi to 0-based for the kernel's index tests.
  orghr_prep_kernel<T><<<grid, block, 0, stream>>>(a_in, a_out, n, ilo - 1, ihi - 1, lda);
}

// One per supported type, matching orghr_bridge.h and the host-side list.
template void orghr_prep<float>(wwr::wwrStream_t, std::size_t, int, int, const float *, float *,
                                std::size_t);
template void orghr_prep<double>(wwr::wwrStream_t, std::size_t, int, int, const double *, double *,
                                 std::size_t);

} // namespace calaman::device
