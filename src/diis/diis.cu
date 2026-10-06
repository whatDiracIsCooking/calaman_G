/**
 * @file diis.cu
 * @brief The bordered DIIS coefficient solve -- calaman.diis's device kernel
 *
 * One block of WWR_WARP_SIZE threads builds the (m+1)-square bordered system in
 * dynamic shared memory; thread 0 then runs partial-pivot Gaussian elimination
 * on it serially (m is the DIIS history depth, a few dozen at most). Shared
 * unchanged between both backends.
 */

#include "diis_bridge.h"

#include <wrappers/math/math.cuh>

#include <cstddef>

namespace calaman::device {

namespace {

template<typename T>
__global__ void diis_solve_kernel(const int m, const int ld, const int newest,
                                  const T *__restrict__ gram, T *__restrict__ coeff,
                                  int *__restrict__ singular_out, const T pivot_floor) {
  // One untyped declaration for every T: a templated `extern __shared__ T[]`
  // redeclares one symbol with conflicting types. double alignment covers float.
  extern __shared__ double smem_storage[];
  T *const b = reinterpret_cast<T *>(smem_storage); // dim x dim, row-major
  const int dim = m + 1;
  T *const rhs = b + (dim * dim); // dim
  const int tid = static_cast<int>(threadIdx.x);

  // b[i][j] = <e_i, e_j> (i, j < m; only the upper triangle of gram is valid),
  // the Lagrange border -1, b[m][m] = 0; rhs = (0, ..., 0, -1).
  for (int idx = tid; idx < dim * dim; idx += static_cast<int>(blockDim.x)) {
    const int i = idx / dim;
    const int j = idx % dim;
    T v;
    if (i < m && j < m) {
      const int a = i <= j ? i : j;
      const int c = i <= j ? j : i;
      v = gram[(static_cast<std::size_t>(a) * ld) + c];
    } else if (i == m && j == m) {
      v = T{0};
    } else {
      v = T{-1};
    }
    b[idx] = v;
  }
  for (int i = tid; i < dim; i += static_cast<int>(blockDim.x)) {
    rhs[i] = (i == m) ? T{-1} : T{0};
  }
  __syncthreads();

  if (tid != 0) {
    return;
  }

  bool singular = false;
  for (int col = 0; col < dim; ++col) {
    int pivot = col;
    T best = wwr::fabs(b[(col * dim) + col]);
    for (int r = col + 1; r < dim; ++r) {
      const T vv = wwr::fabs(b[(r * dim) + col]);
      if (vv > best) {
        best = vv;
        pivot = r;
      }
    }
    if (best < pivot_floor) {
      singular = true;
      break;
    }
    if (pivot != col) {
      for (int j = 0; j < dim; ++j) {
        const T t = b[(col * dim) + j];
        b[(col * dim) + j] = b[(pivot * dim) + j];
        b[(pivot * dim) + j] = t;
      }
      const T t = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = t;
    }
    const T diag = b[(col * dim) + col];
    for (int r = 0; r < dim; ++r) {
      if (r == col) {
        continue;
      }
      const T factor = b[(r * dim) + col] / diag;
      if (factor == T{0}) {
        continue;
      }
      for (int j = col; j < dim; ++j) {
        b[(r * dim) + j] -= factor * b[(col * dim) + j];
      }
      rhs[r] -= factor * rhs[col];
    }
  }

  if (singular_out != nullptr) {
    *singular_out = singular ? 1 : 0;
  }
  // Singular: c = e_newest, so the caller's extrapolation reproduces the newest
  // Fock -- no extrapolation, without a host round-trip to branch on.
  for (int i = 0; i < m; ++i) {
    coeff[i] = singular ? (i == newest ? T{1} : T{0}) : rhs[i] / b[(i * dim) + i];
  }
}

} // namespace

template<typename T>
void diis_solve(const wwr::wwrStream_t stream, const int m, const int ld, const int newest,
                const T *const gram, T *const coeff, int *const singular_out, const T pivot_floor) {
  const auto dim = static_cast<std::size_t>(m) + 1;
  const std::size_t smem_bytes = (dim * dim + dim) * sizeof(T);
  diis_solve_kernel<T><<<1, WWR_WARP_SIZE, smem_bytes, stream>>>(m, ld, newest, gram, coeff,
                                                                  singular_out, pivot_floor);
}

template void diis_solve<float>(wwr::wwrStream_t, int, int, int, const float *, float *, int *,
                                float);
template void diis_solve<double>(wwr::wwrStream_t, int, int, int, const double *, double *, int *,
                                 double);

} // namespace calaman::device
