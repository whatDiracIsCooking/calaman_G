// gebal.cu
//
// The device-kernel half of calaman.gebal: every kernel of both balancing
// stages, plus one launcher per kernel declared in gebal_bridge.h. The host
// DRIVER is not here -- it branches on device data each step and so lives in
// interface.cppm, where it can import wwr.runtime_api for the memcpy/memset/sync
// between these launches. Shared unchanged between both backends, like lacpy.cu:
// a .cu is compiled by the backend compiler, so the raw <<<>>> launch syntax and
// the device intrinsics (__syncthreads, atomicMax/Min, isfinite) are available
// directly, and the complex types/accessors arrive through complex.h.
//
// Permutation stage (full-grid): mark_nonzero flags the window's off-diagonal
// non-zeros, a pick reduction chooses the row/column that isolates an eigenvalue,
// and the two swap kernels apply the symmetric transposition over full
// rows/columns (the out-of-range entries are provably zero, so the full-length
// swap matches LAPACK's sub-range one).
//
// Scaling stage (one cooperating block): the Parlett-Reinsch sweep. It is
// Gauss-Seidel in the index, so it cannot parallelise across the grid -- a
// simultaneous (Jacobi) update does not converge. The block cooperates on the
// O(n) work per index, so a sweep is O(n^2) work on one SM, fine at the moderate
// n where balancing is worth doing.
#include "gebal_bridge.h"

#include <complex.h>
#include "common/elem_ops.cuh"
#include <extension/parallel_for/parallel_for.cuh>

#include <cmath>
#include <cstddef>

namespace calaman::device {

// complex.h puts the neutral complex types in namespace wwr; pull the two type
// names in so the explicit instantiations below can spell them bare. The
// accessor / constructor calls stay wwr::-qualified.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// ── element access ─────────────────────────────────────────────────────────
//
// Scaling is calaman::device::elem_ops<T>::scale (common/elem_ops.cuh). The
// two measurements balancing needs are expm-free helpers over that core:
//
// LAPACK's complex balancing (CGEBAL/ZGEBAL) measures magnitude with CABS1,
// |Re| + |Im|, not the true modulus (elem_ops::modulus): balancing only cares
// about orders of magnitude, and CABS1 keeps a sqrt out of the innermost loop.
// imag_part is 0 for a real T, so both helpers are one type-generic line --
// no per-precision wwrC* branch, which the core already absorbed.

/// @brief CABS1 magnitude |Re| + |Im| (just |x| for a real element).
template<typename T>
__device__ __forceinline__ typename elem_ops<T>::real_type abs1(const T x) {
  return wwr::fabs(elem_ops<T>::real_part(x)) + wwr::fabs(elem_ops<T>::imag_part(x));
}

/// @brief Whether an element is exactly zero (both components, for complex).
template<typename T>
__device__ __forceinline__ bool is_zero(const T x) {
  using R = typename elem_ops<T>::real_type;
  return elem_ops<T>::real_part(x) == R(0) && elem_ops<T>::imag_part(x) == R(0);
}

/// Column-major offset of A(i, j).
__device__ __forceinline__ std::size_t idx(const int i, const int j, const int lda) {
  return static_cast<std::size_t>(j) * static_cast<std::size_t>(lda) + static_cast<std::size_t>(i);
}

template<typename R>
__device__ __forceinline__ R rmax(const R a, const R b) {
  return a > b ? a : b;
}
template<typename R>
__device__ __forceinline__ R rmin(const R a, const R b) {
  return a < b ? a : b;
}

inline int grid_for(const int count, const int block) { return (count + block - 1) / block; }

constexpr int kLinearBlock = 256;
constexpr int kSweepBlock = 256;

// ── permutation stage kernels ──────────────────────────────────────────────

/// @brief [kernel] Flag every row and column of the active window that owns an
///        off-diagonal non-zero.
template<typename T>
__global__ void mark_nonzero_kernel(const T *__restrict__ A, const int lda, const int r0,
                                    const int r1, const int c0, const int c1,
                                    int *__restrict__ row_flag, int *__restrict__ col_flag) {
  const int i = r0 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int j = c0 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i > r1 || j > c1 || i == j) {
    return;
  }
  if (!is_zero(A[idx(i, j, lda)])) {
    // Every writer stores the same value, so the concurrent stores agree.
    row_flag[i] = 1;
    col_flag[j] = 1;
  }
}

/// @brief [kernel] Largest unflagged index in [lo, hi]; leaves *out alone when none.
__global__ void pick_max_unflagged_kernel(const int *__restrict__ flag, const int lo, const int hi,
                                          int *out) {
  const int j = lo + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j > hi) {
    return;
  }
  if (flag[j] == 0) {
    atomicMax(out, j);
  }
}

/// @brief [kernel] Smallest unflagged index in [lo, hi]; leaves *out alone when none.
__global__ void pick_min_unflagged_kernel(const int *__restrict__ flag, const int lo, const int hi,
                                          int *out) {
  const int j = lo + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (j > hi) {
    return;
  }
  if (flag[j] == 0) {
    atomicMin(out, j);
  }
}

/// @brief [kernel] Write a single int (one thread).
__global__ void set_int_kernel(int *p, const int v) { *p = v; }

/// @brief [kernel] Swap columns j and m over all n rows.
template<typename T>
__global__ void swap_cols_kernel(T *A, const int lda, const int n, const int j, const int m) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) {
    return;
  }
  const std::size_t oj = idx(i, j, lda);
  const std::size_t om = idx(i, m, lda);
  const T t = A[oj];
  A[oj] = A[om];
  A[om] = t;
}

/// @brief [kernel] Swap rows j and m over all n columns.
template<typename T>
__global__ void swap_rows_kernel(T *A, const int lda, const int n, const int j, const int m) {
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (c >= n) {
    return;
  }
  const std::size_t oj = idx(j, c, lda);
  const std::size_t om = idx(m, c, lda);
  const T t = A[oj];
  A[oj] = A[om];
  A[om] = t;
}

/// @brief [kernel] scale[m] = j_one_based, the transposition record (1-based).
template<typename R>
__global__ void record_perm_kernel(R *scale, const int m, const int j_one_based) {
  scale[m] = static_cast<R>(j_one_based);
}

/// @brief Per-index fill functor for parallel_for -- seeds scale[i] = 1.
template<typename R>
struct FillOnesFunctor {
  R *const scale_;
  __device__ void operator()(const std::size_t i) const { scale_[i] = R(1); }
};

// ── scaling stage kernel ───────────────────────────────────────────────────

/// @brief [kernel] One full Parlett-Reinsch sweep over k0..l0, in a single block.
///
/// Each index i: measure the off-diagonal 1-norms c (column) and r (row) over the
/// window and the overflow sentinels ca/ra (largest magnitude in column/row),
/// pick a power-of-two factor f that balances c against r while every
/// intermediate stays in [sfmin2, sfmax2], demand a 5% improvement, and -- if
/// accepted -- scale row i by 1/f and column i by f. The 5% margin is what makes
/// the sweep terminate instead of chattering by one radix step forever.
template<typename T, typename R, int BLOCK>
__global__ void gebal_sweep_kernel(const int n, T *A, const int lda, const int k0, const int l0,
                                   R *scale, int *noconv, const scale_limits<R> lim) {
  using ops = elem_ops<T>;
  static_assert(sizeof(typename ops::real_type) == sizeof(R),
                "sweep real type R must match the element's real type");

  constexpr R kSclfac = R(2);
  constexpr R kFactor = R(0.95);

  __shared__ R s_c[BLOCK];
  __shared__ R s_r[BLOCK];
  __shared__ R s_ca[BLOCK];
  __shared__ R s_ra[BLOCK];
  __shared__ R s_bcast[4];

  const int t = static_cast<int>(threadIdx.x);

  for (int i = k0; i <= l0; ++i) {
    // c, r: off-diagonal 1-norms of column i and row i over the window.
    // ca, ra: largest magnitude in column i (rows 0..l0) and in row i
    // (columns k0..n-1), diagonal included -- LAPACK's overflow sentinels.
    R c = R(0);
    R r = R(0);
    R ca = R(0);
    R ra = R(0);
    for (int j = t; j < n; j += BLOCK) {
      const R a_col = abs1(A[idx(j, i, lda)]);
      const R a_row = abs1(A[idx(i, j, lda)]);
      if (j >= k0 && j <= l0 && j != i) {
        c += a_col;
        r += a_row;
      }
      if (j <= l0) {
        ca = rmax(ca, a_col);
      }
      if (j >= k0) {
        ra = rmax(ra, a_row);
      }
    }
    s_c[t] = c;
    s_r[t] = r;
    s_ca[t] = ca;
    s_ra[t] = ra;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
      if (t < s) {
        s_c[t] += s_c[t + s];
        s_r[t] += s_r[t + s];
        s_ca[t] = rmax(s_ca[t], s_ca[t + s]);
        s_ra[t] = rmax(s_ra[t], s_ra[t + s]);
      }
      __syncthreads();
    }
    if (t == 0) {
      s_bcast[0] = s_c[0];
      s_bcast[1] = s_r[0];
      s_bcast[2] = s_ca[0];
      s_bcast[3] = s_ra[0];
    }
    __syncthreads();
    c = s_bcast[0];
    r = s_bcast[1];
    ca = s_bcast[2];
    ra = s_bcast[3];

    // Every thread runs the decision on identical inputs, so `accept` and `f`
    // are block-uniform and the barriers below stay uniform too.
    R f = R(1);
    bool accept = false;

    // A NaN makes every comparison below false, which would spin the search
    // loops forever; newer LAPACK revisions bail out with INFO = -3 instead.
    const bool usable = (c != R(0)) && (r != R(0)) && isfinite(c + r + ca + ra);
    if (usable) {
      const R s0 = c + r;
      R g = r / kSclfac;
      while (!(c >= g || rmax(rmax(f, c), ca) >= lim.sfmax2 ||
               rmin(rmin(r, g), ra) <= lim.sfmin2)) {
        f *= kSclfac;
        c *= kSclfac;
        ca *= kSclfac;
        r /= kSclfac;
        g /= kSclfac;
        ra /= kSclfac;
      }
      g = c / kSclfac;
      while (!(g < r || rmax(r, ra) >= lim.sfmax2 ||
               rmin(rmin(rmin(f, c), g), ca) <= lim.sfmin2)) {
        f /= kSclfac;
        c /= kSclfac;
        g /= kSclfac;
        ca /= kSclfac;
        r *= kSclfac;
        ra *= kSclfac;
      }
      // Demand a 5% improvement -- that margin is what makes the sweep terminate
      // instead of chattering by one radix step forever.
      if (c + r < kFactor * s0) {
        accept = true;
        const R sc = scale[i];
        if (f < R(1) && sc < R(1) && f * sc <= lim.sfmin1) {
          accept = false;
        }
        if (f > R(1) && sc > R(1) && sc >= lim.sfmax1 / f) {
          accept = false;
        }
      }
    }
    __syncthreads(); // every read of A for this index is done

    // Row i and column i share A(i,i); scaling both at once would race on that
    // entry, so a barrier separates the halves. f * (1/f) is exact for a power
    // of two, so the diagonal comes back unchanged.
    if (accept) {
      const R inv_f = R(1) / f;
      for (int j = k0 + t; j < n; j += BLOCK) {
        A[idx(i, j, lda)] = ops::scale(A[idx(i, j, lda)], inv_f);
      }
    }
    __syncthreads();
    if (accept) {
      for (int j = t; j <= l0; j += BLOCK) {
        A[idx(j, i, lda)] = ops::scale(A[idx(j, i, lda)], f);
      }
      if (t == 0) {
        scale[i] *= f;
        *noconv = 1;
      }
    }
    __syncthreads();
  }
}

} // namespace

// ── launchers ────────────────────────────────────────────────────────────────

template<typename R>
void gebal_fill_ones(const wwr::wwrStream_t stream, const int n, R *const scale) {
  if (n < 1) {
    return;
  }
  const FillOnesFunctor<R> functor{scale};
  wwr::extension::parallel_for<std::size_t>(stream, static_cast<std::size_t>(n), functor);
}

template<typename T>
void gebal_mark_nonzero(const wwr::wwrStream_t stream, const T *const A, const int lda, const int r0,
                        const int r1, const int c0, const int c1, int *const row_flag,
                        int *const col_flag) {
  const int rows = r1 - r0 + 1;
  const int cols = c1 - c0 + 1;
  if (rows < 1 || cols < 1) {
    return;
  }
  const dim3 block(16, 16);
  const dim3 grid(grid_for(cols, static_cast<int>(block.x)), grid_for(rows, static_cast<int>(block.y)));
  mark_nonzero_kernel<T><<<grid, block, 0, stream>>>(A, lda, r0, r1, c0, c1, row_flag, col_flag);
}

void gebal_set_int(const wwr::wwrStream_t stream, int *const p, const int v) {
  set_int_kernel<<<1, 1, 0, stream>>>(p, v);
}

void gebal_pick_max_unflagged(const wwr::wwrStream_t stream, const int *const flag, const int lo,
                              const int hi, int *const out) {
  const int span = hi - lo + 1;
  if (span < 1) {
    return;
  }
  pick_max_unflagged_kernel<<<grid_for(span, kLinearBlock), kLinearBlock, 0, stream>>>(flag, lo, hi,
                                                                                       out);
}

void gebal_pick_min_unflagged(const wwr::wwrStream_t stream, const int *const flag, const int lo,
                              const int hi, int *const out) {
  const int span = hi - lo + 1;
  if (span < 1) {
    return;
  }
  pick_min_unflagged_kernel<<<grid_for(span, kLinearBlock), kLinearBlock, 0, stream>>>(flag, lo, hi,
                                                                                       out);
}

template<typename R>
void gebal_record_perm(const wwr::wwrStream_t stream, R *const scale, const int m,
                       const int j_one_based) {
  record_perm_kernel<R><<<1, 1, 0, stream>>>(scale, m, j_one_based);
}

template<typename T>
void gebal_swap_cols(const wwr::wwrStream_t stream, T *const A, const int lda, const int n,
                     const int j, const int m) {
  if (n < 1) {
    return;
  }
  swap_cols_kernel<T><<<grid_for(n, kLinearBlock), kLinearBlock, 0, stream>>>(A, lda, n, j, m);
}

template<typename T>
void gebal_swap_rows(const wwr::wwrStream_t stream, T *const A, const int lda, const int n,
                     const int j, const int m) {
  if (n < 1) {
    return;
  }
  swap_rows_kernel<T><<<grid_for(n, kLinearBlock), kLinearBlock, 0, stream>>>(A, lda, n, j, m);
}

template<typename T, typename R>
void gebal_sweep(const wwr::wwrStream_t stream, const int n, T *const A, const int lda, const int k0,
                 const int l0, R *const scale, int *const noconv, const scale_limits<R> lim) {
  gebal_sweep_kernel<T, R, kSweepBlock><<<1, kSweepBlock, 0, stream>>>(n, A, lda, k0, l0, scale,
                                                                       noconv, lim);
}

// One instantiation per supported type, matching gebal_bridge.h's declarations
// and interface.cppm's extern template list -- float, double, and the two
// complex types. The real scale type R is ComplexToRealType<T>: float for
// float/wwrFloatComplex, double for double/wwrDoubleComplex.
template void gebal_fill_ones<float>(wwr::wwrStream_t, int, float *);
template void gebal_fill_ones<double>(wwr::wwrStream_t, int, double *);

template void gebal_mark_nonzero<float>(wwr::wwrStream_t, const float *, int, int, int, int, int,
                                        int *, int *);
template void gebal_mark_nonzero<double>(wwr::wwrStream_t, const double *, int, int, int, int, int,
                                         int *, int *);
template void gebal_mark_nonzero<wwrFloatComplex>(wwr::wwrStream_t, const wwrFloatComplex *, int,
                                                  int, int, int, int, int *, int *);
template void gebal_mark_nonzero<wwrDoubleComplex>(wwr::wwrStream_t, const wwrDoubleComplex *, int,
                                                   int, int, int, int, int *, int *);

template void gebal_record_perm<float>(wwr::wwrStream_t, float *, int, int);
template void gebal_record_perm<double>(wwr::wwrStream_t, double *, int, int);

template void gebal_swap_cols<float>(wwr::wwrStream_t, float *, int, int, int, int);
template void gebal_swap_cols<double>(wwr::wwrStream_t, double *, int, int, int, int);
template void gebal_swap_cols<wwrFloatComplex>(wwr::wwrStream_t, wwrFloatComplex *, int, int, int,
                                               int);
template void gebal_swap_cols<wwrDoubleComplex>(wwr::wwrStream_t, wwrDoubleComplex *, int, int, int,
                                                int);

template void gebal_swap_rows<float>(wwr::wwrStream_t, float *, int, int, int, int);
template void gebal_swap_rows<double>(wwr::wwrStream_t, double *, int, int, int, int);
template void gebal_swap_rows<wwrFloatComplex>(wwr::wwrStream_t, wwrFloatComplex *, int, int, int,
                                               int);
template void gebal_swap_rows<wwrDoubleComplex>(wwr::wwrStream_t, wwrDoubleComplex *, int, int, int,
                                                int);

template void gebal_sweep<float, float>(wwr::wwrStream_t, int, float *, int, int, int, float *,
                                        int *, scale_limits<float>);
template void gebal_sweep<double, double>(wwr::wwrStream_t, int, double *, int, int, int, double *,
                                          int *, scale_limits<double>);
template void gebal_sweep<wwrFloatComplex, float>(wwr::wwrStream_t, int, wwrFloatComplex *, int, int,
                                                  int, float *, int *, scale_limits<float>);
template void gebal_sweep<wwrDoubleComplex, double>(wwr::wwrStream_t, int, wwrDoubleComplex *, int,
                                                    int, int, double *, int *, scale_limits<double>);

} // namespace calaman::device
