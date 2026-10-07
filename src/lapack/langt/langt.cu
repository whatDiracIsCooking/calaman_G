// langt.cu
//
// The device half of calaman.langt: the ?langt norm of a general tridiagonal
// (dl, d, du) -- A(i+1,i) = dl(i), A(i,i) = d(i), A(i,i+1) = du(i) -- in ONE
// single-block launch, calaman.lanst's shape. Each thread takes a contiguous
// chunk of indices i and folds its share; block_reduce
// (common/block_reduce.cuh) folds the per-thread partials, and thread 0 writes
// the result.
//
//   max_abs  ('M'): max |d(i)|, |dl(i)|, |du(i)|, then max
//   one      ('1'): column i's |du(i-1)| + |d(i)| + |dl(i)|, then max
//   inf      ('I'): row i's    |dl(i-1)| + |d(i)| + |du(i)|, then max
//   frobenius('F'): sum |d(i)|^2 + |dl(i)|^2 + |du(i)|^2, then sqrt -- a
//                   plain sum, not ?lassq (the calaman.lange precedent)
//
// Index i owns dl(i) and du(i), so no off-diagonal straddles two chunks. A
// complex T has a real norm R; |z| is the modulus, |z|^2 is re^2 + im^2. The
// max folds propagate a NaN, matching DLANGT's DISNAN guard. lanst's kernel is
// not reused: its diagonal is real and its row and column sums coincide.
//
// Shared unchanged between both backends -- under HIP this directory's
// CMakeLists.txt (via calaman_add_gpu_device_library) compiles it as -x hip.
#include "langt_bridge.h"

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

/// @brief [kernel] One block folds the @p which norm of (dl, d, du) into *d_result
template<typename T, typename R>
__global__ void langt_kernel(const MatrixNorm which, const std::size_t n, const T *const dl,
                             const T *const d, const T *const du, R *const d_result) {
  const std::size_t chunk = (n + kBlock - 1) / kBlock;
  const unsigned int nactive = static_cast<unsigned int>((n + chunk - 1) / chunk);
  const std::size_t lo = threadIdx.x * chunk;
  const std::size_t hi = lo + chunk < n ? lo + chunk : n;

  R acc = R{0};
  if (lo < n) {
    switch (which) {
    case MatrixNorm::max_abs:
      for (std::size_t i = lo; i < hi; ++i) {
        acc = max_nan(acc, elem_ops<T>::modulus(d[i]));
        if (i + 1 < n) {
          acc = max_nan(acc, elem_ops<T>::modulus(dl[i]));
          acc = max_nan(acc, elem_ops<T>::modulus(du[i]));
        }
      }
      break;
    case MatrixNorm::one:
    case MatrixNorm::inf: {
      // Column i reads du(i-1) above and dl(i) below; row i reads dl(i-1) to
      // the left and du(i) to the right.
      const T *const before = which == MatrixNorm::one ? du : dl;
      const T *const after = which == MatrixNorm::one ? dl : du;
      for (std::size_t i = lo; i < hi; ++i) {
        R sum = elem_ops<T>::modulus(d[i]);
        if (i > 0) {
          sum += elem_ops<T>::modulus(before[i - 1]);
        }
        if (i + 1 < n) {
          sum += elem_ops<T>::modulus(after[i]);
        }
        acc = (i == lo) ? sum : max_nan(acc, sum);
      }
      break;
    }
    case MatrixNorm::frobenius:
      for (std::size_t i = lo; i < hi; ++i) {
        acc += entry_sq<T, R>(d[i]);
        if (i + 1 < n) {
          acc += entry_sq<T, R>(dl[i]) + entry_sq<T, R>(du[i]);
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
void langt(const wwr::wwrStream_t stream, const MatrixNorm which, const std::size_t n,
           const T *const d_dl, const T *const d_d, const T *const d_du, R *const d_result) {
  langt_kernel<T, R><<<1, kBlock, 0, stream>>>(which, n, d_dl, d_d, d_du, d_result);
}

// One per supported type, matching langt_bridge.h and interface.cppm's `extern
// template` list.
template void langt<float, float>(wwr::wwrStream_t, MatrixNorm, std::size_t, const float *,
                                  const float *, const float *, float *);
template void langt<double, double>(wwr::wwrStream_t, MatrixNorm, std::size_t, const double *,
                                    const double *, const double *, double *);
template void langt<wwr::wwrFloatComplex, float>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                 const wwr::wwrFloatComplex *,
                                                 const wwr::wwrFloatComplex *,
                                                 const wwr::wwrFloatComplex *, float *);
template void langt<wwr::wwrDoubleComplex, double>(wwr::wwrStream_t, MatrixNorm, std::size_t,
                                                   const wwr::wwrDoubleComplex *,
                                                   const wwr::wwrDoubleComplex *,
                                                   const wwr::wwrDoubleComplex *, double *);

} // namespace calaman::device
