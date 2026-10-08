// larscl2.cu
//
// The device half of calaman.larscl2: the per-element reciprocal scaling
// x(i,j) <-- x(i,j) / d(i), lascl2.cu's inverse twin and the same launch shape
// -- a 2-D grid of 1-D blocks, 4*WWR_WARP_SIZE threads along the rows (x) and
// one block per column (y), so both the x access and the d(i) read coalesce.
//
// The reference is a literal `X(I,J) / D(I)`, and the comparison in
// test/larscl2/ is bitwise, so this stays an IEEE divide: no reciprocal-then-
// multiply, no fast-math. A complex x is divided per component by the real
// d(i) -- what a Fortran COMPLEX / REAL reduces to, and never a full complex
// divide whose Smith scaling could perturb the last bit.
#include "larscl2_bridge.h"

#include "common/align_up.h"
#include "common/elem_ops.cuh"
#include <runtime.h>

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

/// @brief a / d for a real element, or (Re a / d, Im a / d) for a complex one
template<typename T, typename RealT>
__device__ __forceinline__ T div_by_real(const T a, const RealT d) {
  if constexpr (std::is_same_v<T, RealT>) {
    return a / d;
  } else {
    return make_complex(elem_ops<T>::real_part(a) / d, elem_ops<T>::imag_part(a) / d);
  }
}

/// @brief [kernel] Divide column-major x(i,j) by the diagonal entry d(i)
///
/// Row i from the 1-D block along x, column j = blockIdx.y; the grid covers
/// every j, so only i needs a bound.
template<typename T, typename RealT>
__global__ void larscl2_kernel(const RealT *const d, T *const x, const std::size_t m,
                               const std::size_t ldx) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y;
  if (i >= m) {
    return;
  }
  // j * ldx is formed in 64-bit because ldx is size_t.
  T *const xij = x + i + j * ldx;
  *xij = div_by_real(*xij, d[i]);
}

} // namespace

template<typename T, typename RealT>
void larscl2(const wwr::wwrStream_t stream, const std::size_t m, const std::size_t n,
             const RealT *const d, T *const x, const std::size_t ldx) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  // Row-block count in size_t so a tall matrix cannot overflow a 32-bit
  // intermediate; one block per column in y (gridDim.y <= 65535, as lascl2).
  const dim3 grid(idivup<std::size_t>(m, kBlockSize), n);
  const dim3 block(kBlockSize);
  larscl2_kernel<T, RealT><<<grid, block, 0, stream>>>(d, x, m, ldx);
}

// One per supported type, matching interface.cppm's `extern template` list and
// instantiations.cpp's.
template void larscl2<float, float>(wwr::wwrStream_t, std::size_t, std::size_t, const float *,
                                    float *, std::size_t);
template void larscl2<double, double>(wwr::wwrStream_t, std::size_t, std::size_t, const double *,
                                      double *, std::size_t);
template void larscl2<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, std::size_t, std::size_t,
                                                   const float *, wwr::wwrFloatComplex *,
                                                   std::size_t);
template void larscl2<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, std::size_t, std::size_t,
                                                     const double *, wwr::wwrDoubleComplex *,
                                                     std::size_t);

} // namespace calaman::device
