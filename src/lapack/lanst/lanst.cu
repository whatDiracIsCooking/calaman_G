// lanst.cu
//
// The device half of calaman.lanst AND calaman.lanht: the ?lanst/?lanht norm of
// a symmetric (or Hermitian) tridiagonal (d, e) in ONE single-block launch. Each
// thread takes a contiguous chunk of rows and folds its share of the norm;
// block_reduce (common/block_reduce.cuh) then folds the per-thread partials, and
// thread 0 writes the result.
//
//   max_abs  ('M'): max(|d(i)|, |e(i)|) over the chunk, then max
//   one/inf  ('1'/'I', equal by symmetry): per-row |e(i-1)|+|d(i)|+|e(i)|, max
//   frobenius('F'): sum d(i)^2 + 2 |e(i)|^2, then sqrt -- a plain sum, not ?lassq
//
// d is always real (R); e is T, real for ?lanst and complex for ?lanht, whose
// |e| is the modulus and |e|^2 is re^2 + im^2. A real T's max norm runs through
// lanst_max_abs (lanst.h), the per-thread helper ?sterf/?steqr share.
//
// One block is deliberate: a tridiagonal of order n holds 2n-1 numbers, so a
// block of threads covers any n this library's O(n^2)-and-up callers reach. The
// max folds propagate a NaN from either operand, matching DLANST's DISNAN guard.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "lanst_bridge.h"

#include "lanst.h"

#include "common/block_reduce.cuh"
#include "common/elem_ops.cuh"

#include <cstddef>
#include <type_traits>

namespace calaman::device {

namespace {

constexpr unsigned int kBlock = 256;

/// @brief |x|^2 as re^2 + im^2 (im is 0 for a real T)
template<typename T, typename R>
__device__ R entry_sq(const T x) {
  const R re = elem_ops<T>::real_part(x);
  const R im = elem_ops<T>::imag_part(x);
  return re * re + im * im;
}

/// @brief [kernel] One block folds the @p which norm of (d, e) into *d_result
template<typename T, typename R>
__global__ void lanst_kernel(const MatrixNorm which, const std::size_t n, const R *const d,
                             const T *const e, R *const d_result) {
  const std::size_t chunk = (n + kBlock - 1) / kBlock;
  const unsigned int nactive = static_cast<unsigned int>((n + chunk - 1) / chunk);
  const std::size_t lo = threadIdx.x * chunk;
  const std::size_t hi = lo + chunk < n ? lo + chunk : n;

  R acc = R{0};
  if (lo < n) {
    switch (which) {
    case MatrixNorm::max_abs:
      // This chunk's d[lo, hi) and e[lo, hi-1); the e(hi-1) linking it to the
      // next chunk is folded here, by its left owner.
      if constexpr (std::is_same_v<T, R>) {
        acc = lanst_max_abs(hi - lo, d + lo, e + lo);
      } else {
        for (std::size_t i = lo; i < hi; ++i) {
          acc = max_nan(acc, elem_ops<R>::modulus(d[i]));
          if (i + 1 < hi) {
            acc = max_nan(acc, elem_ops<T>::modulus(e[i]));
          }
        }
      }
      if (hi < n) {
        acc = max_nan(acc, elem_ops<T>::modulus(e[hi - 1]));
      }
      break;
    case MatrixNorm::one:
    case MatrixNorm::inf:
      for (std::size_t i = lo; i < hi; ++i) {
        R row = elem_ops<R>::modulus(d[i]);
        if (i > 0) {
          row += elem_ops<T>::modulus(e[i - 1]);
        }
        if (i + 1 < n) {
          row += elem_ops<T>::modulus(e[i]);
        }
        acc = (i == lo) ? row : max_nan(acc, row);
      }
      break;
    case MatrixNorm::frobenius:
      for (std::size_t i = lo; i < hi; ++i) {
        acc += d[i] * d[i];
        if (i + 1 < n) {
          acc += R{2} * entry_sq<T, R>(e[i]);
        }
      }
      break;
    }
  }

  const R total = (which == MatrixNorm::frobenius) ? block_reduce<kBlock>(acc, AddOp{}, nactive)
                                                   : block_reduce<kBlock>(acc, MaxNanOp{}, nactive);
  if (threadIdx.x == 0) {
    if (which != MatrixNorm::frobenius) {
      *d_result = total;
    } else if constexpr (std::is_same_v<R, float>) {
      *d_result = sqrtf(total);
    } else {
      *d_result = sqrt(total);
    }
  }
}

} // namespace

template<typename T, typename R>
void lanst(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
           const R *const d_d, const T *const d_e, R *const d_result) {
  lanst_kernel<T, R><<<1, kBlock, 0, stream>>>(which, n, d_d, d_e, d_result);
}

// One per supported type, matching lanst_bridge.h and the `extern template`
// lists of calaman.lanst (the real two) and calaman.lanht (the complex two).
template void lanst<float, float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                                  const float *, float *);
template void lanst<double, double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                                    const double *, double *);
template void lanst<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                 const float *, const wwr::wwrFloatComplex *,
                                                 float *);
template void lanst<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   const double *, const wwr::wwrDoubleComplex *,
                                                   double *);

} // namespace calaman::device
