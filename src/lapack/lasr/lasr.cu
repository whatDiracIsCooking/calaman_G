// lasr.cu
//
// The device half of calaman.lasr: ?lasr in ONE launch. The lines of the
// unrotated dimension (columns for Side::L, rows for Side::R) are independent,
// so the grid splits them into slabs of kSlab, and each block runs the shared
// block-cooperative lasr_block (lasr.h) on its slab -- the same helper ?steqr
// calls from inside its own kernel, so the oracle suite exercises it.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lasr_bridge.h"

#include "lasr.h"

#include <cstddef>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;
// Lines per block: several per thread, so lasr_block's strided loop is covered.
constexpr std::size_t kSlab = 4 * kBlock;

/// @brief [kernel] Block b applies ?lasr to lines [b*kSlab, (b+1)*kSlab)
template<typename T, typename R>
__global__ void lasr_kernel(const Side side, const Pivot pivot, const Direct direct,
                            const std::size_t m, const std::size_t n, const R *const c,
                            const R *const s, T *const A, const std::size_t lda) {
  const std::size_t first = static_cast<std::size_t>(blockIdx.x) * kSlab;
  const std::size_t lines = side == Side::L ? n : m;
  const std::size_t count = lines - first < kSlab ? lines - first : kSlab;
  if (side == Side::L) {
    lasr_block(side, pivot, direct, m, count, c, s, A + first * lda, lda);
  } else {
    lasr_block(side, pivot, direct, count, n, c, s, A + first, lda);
  }
}

} // namespace

template<typename T, typename R>
void lasr(const wwr::wwrStream_t stream, const Side side, const Pivot pivot, const Direct direct,
          const std::size_t m, const std::size_t n, const R *const c, const R *const s, T *const A,
          const std::size_t lda) {
  const std::size_t lines = side == Side::L ? n : m;
  const auto blocks = static_cast<unsigned int>((lines + kSlab - 1) / kSlab);
  lasr_kernel<T, R><<<blocks, kBlock, 0, stream>>>(side, pivot, direct, m, n, c, s, A, lda);
}

// complex.h (via lasr.h's elem_ops.cuh) puts the neutral complex types in
// namespace wwr; pull them in so the instantiations below can spell them bare.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

// One per supported type, matching lasr_bridge.h and interface.cppm's
// `extern template` list.
template void lasr<float, float>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t, std::size_t,
                                 const float *, const float *, float *, std::size_t);
template void lasr<double, double>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                   std::size_t, const double *, const double *, double *,
                                   std::size_t);
template void lasr<wwrFloatComplex, float>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                           std::size_t, const float *, const float *,
                                           wwrFloatComplex *, std::size_t);
template void lasr<wwrDoubleComplex, double>(wwr::wwrStream_t, Side, Pivot, Direct, std::size_t,
                                             std::size_t, const double *, const double *,
                                             wwrDoubleComplex *, std::size_t);

} // namespace calaman::device
