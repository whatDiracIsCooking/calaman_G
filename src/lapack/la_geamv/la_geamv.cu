// la_geamv.cu
//
// The device half of calaman.la_geamv: y(i) := alpha * sum_j |op(A)(i,j)| *
// |x(j)| + beta * |y(i)|, then the symbolic-zero nudge, one thread per y(i).
// |.| is CABS1, |re| + |im|, for a complex element. The thread replays
// ?LA_GEAMV's loop body exactly -- beta * |y| first, then (alpha * |x|) * |a|
// added left to right in j -- carrying the reference's SYMB_ZERO flag, so the
// result is bitwise the reference's. That needs every multiply and add rounded
// on its own, never contracted into an FMA: CUDA's _rn intrinsics guarantee
// it; HIP's lower to a plain operator in a header that HIP's clang contracts
// (its default is fp-contract=fast), so under HIP the operators are spelled
// here, inside this file's `fp contract(off)`.
//
// Trans::N walks row i of A (coalesced across the block's rows); T and C walk
// column i -- identical, as |conj(a)| == |a|. That walk is strided across
// threads; a block-cooperative transpose would coalesce it, but would also
// have to keep the per-thread sequential sum order the bitwise oracle needs.
#include "la_geamv_bridge.h"

#include "common/align_up.h"
#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

#if defined(__HIP__)
#pragma clang fp contract(off)
#endif

namespace calaman::device {

namespace {

/// @brief a * b, rounded once and never fused with a neighbouring add
template<typename R>
__device__ __forceinline__ R mul_rn(const R a, const R b) {
#if defined(__HIP__)
  return a * b;
#else
  if constexpr (std::is_same_v<R, float>) {
    return __fmul_rn(a, b);
  } else {
    return __dmul_rn(a, b);
  }
#endif
}

/// @brief a + b, rounded once and never fused with a neighbouring multiply
template<typename R>
__device__ __forceinline__ R add_rn(const R a, const R b) {
#if defined(__HIP__)
  return a + b;
#else
  if constexpr (std::is_same_v<R, float>) {
    return __fadd_rn(a, b);
  } else {
    return __dadd_rn(a, b);
  }
#endif
}

/// @brief BLAS CABS1: |a| for a real a, |Re a| + |Im a| for a complex one
template<typename T, typename R>
__device__ __forceinline__ R abs1(const T a) {
  using ops = elem_ops<T>;
  if constexpr (std::is_same_v<T, R>) {
    return wwr::fabs(a);
  } else {
    return add_rn(wwr::fabs(ops::real_part(a)), wwr::fabs(ops::imag_part(a)));
  }
}

/// @brief [kernel] One thread per y(i): ?LA_GEAMV's loop body for that i
template<typename T, typename R>
__global__ void la_geamv_kernel(const Trans trans, const std::size_t lenx, const std::size_t leny,
                                const R alpha, const T *const a, const std::size_t lda,
                                const T *const x, const std::ptrdiff_t incx, const R beta,
                                R *const y, const std::ptrdiff_t incy, const R safe1) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= leny) {
    return;
  }
  R *const yi = y + static_cast<std::ptrdiff_t>(i) * incy;
  // SYMB_ZERO starts true when beta * |y(i)| is exactly zero by construction.
  bool symb_zero = true;
  R acc = R{0};
  if (beta != R{0}) {
    acc = *yi;
    if (acc != R{0}) {
      symb_zero = false;
      acc = mul_rn(beta, wwr::fabs(acc));
    }
  }
  if (alpha != R{0}) {
    // op(A)(i, j): A(i, j) for N, A(j, i) for T and C.
    const bool by_row = trans == Trans::N;
    const T *const ai = by_row ? a + i : a + i * lda;
    const std::size_t step = by_row ? lda : 1;
    for (std::size_t j = 0; j < lenx; ++j) {
      const R temp = abs1<T, R>(ai[j * step]);
      const R xa = abs1<T, R>(x[static_cast<std::ptrdiff_t>(j) * incx]);
      symb_zero = symb_zero && (xa == R{0} || temp == R{0});
      acc = add_rn(acc, mul_rn(mul_rn(alpha, xa), temp));
    }
  }
  if (!symb_zero) {
    acc = add_rn(acc, wwr::copysign(safe1, acc));
  }
  *yi = acc;
}

} // namespace

template<typename T, typename R>
void la_geamv(const wwr::wwrStream_t stream, const Trans trans, const std::size_t lenx,
              const std::size_t leny, const R alpha, const T *const a, const std::size_t lda,
              const T *const x, const std::ptrdiff_t incx, const R beta, R *const y,
              const std::ptrdiff_t incy, const R safe1) {
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  const dim3 grid(idivup<std::size_t>(leny, kBlockSize));
  la_geamv_kernel<T, R><<<grid, kBlockSize, 0, stream>>>(trans, lenx, leny, alpha, a, lda, x, incx,
                                                         beta, y, incy, safe1);
}

// One per supported type, matching la_geamv_bridge.h and interface.cppm's
// `extern template` list.
#define CLM_LA_GEAMV_INSTANTIATE(T, R)                                                             \
  template void la_geamv<T, R>(wwr::wwrStream_t, Trans, std::size_t, std::size_t, R, const T *,    \
                               std::size_t, const T *, std::ptrdiff_t, R, R *, std::ptrdiff_t, R);
CLM_LA_GEAMV_INSTANTIATE(float, float)
CLM_LA_GEAMV_INSTANTIATE(double, double)
CLM_LA_GEAMV_INSTANTIATE(wwr::wwrFloatComplex, float)
CLM_LA_GEAMV_INSTANTIATE(wwr::wwrDoubleComplex, double)
#undef CLM_LA_GEAMV_INSTANTIATE

} // namespace calaman::device
