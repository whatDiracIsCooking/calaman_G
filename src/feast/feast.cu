// feast.cu
//
// The device-kernel half of calaman.feast: everything the solver does that is
// not a wrapped BLAS or eigensolver call -- building the shifted resolvents,
// moving between the real subspace and the Ne complex right-hand sides, the
// 1-norm, and the per-iteration bookkeeping (pair selection, residuals) that
// decides an iteration's outcome. The host DRIVER is not here: it branches on
// device data each iteration and so lives in the module partitions, reaching
// these kernels through the launchers declared in feast_bridge.h.
//
// Shared unchanged between both backends, like lacpy.cu / gebal.cu: a .cu is
// compiled by the backend compiler, so the raw <<<>>> launch syntax and the
// complex types/accessors (through complex.h) are available directly.
//
// NEUTRAL COMPLEX, NEVER .x/.y. cuFloatComplex is an operator-less float2 whose
// components are reached as .x/.y, but hipFloatComplex is a class -- so raw
// field access is not portable. Components are read with complex.h's
// wwrCreal*/wwrCimag* accessors and values built with make_wwr*Complex, matching
// gebal.cu / complex_cast.cu. The block reductions and their AddOp/MaxNanOp
// folds come from common/block_reduce.cuh, whose header carries the
// shared-tree-not-shuffle rationale.
#include "feast_bridge.h"

#include "common/block_reduce.cuh"
#include <complex.h>

#include <cstddef>

namespace calaman::device {

// complex.h puts the neutral complex types in namespace wwr; pull the two type
// names in so the explicit instantiations below can spell them bare. The
// accessor / constructor calls stay wwr::-qualified.
using wwr::wwrDoubleComplex;
using wwr::wwrFloatComplex;

namespace {

// 4 warps per block, matching lacpy/gebal/reduce_columns: a configure-time value
// (WWR_WARP_SIZE is 32 default, 64 on CDNA), so 128 or 256, and a power of two --
// block_reduce<kBlock>'s halving tree requires that.
constexpr unsigned int kBlock = 4 * WWR_WARP_SIZE;
constexpr std::size_t kMaxBlocks = 4096;

/// Blocks for a grid-stride loop over @p elems; the loop covers any excess.
inline unsigned int blocks_for(const std::size_t elems) {
  std::size_t blocks = (elems + kBlock - 1) / kBlock;
  if (blocks < 1) {
    blocks = 1;
  }
  if (blocks > kMaxBlocks) {
    blocks = kMaxBlocks;
  }
  return static_cast<unsigned int>(blocks);
}

// ── neutral complex helpers ──────────────────────────────────────────────────
//
// Overloaded on the real type so a kernel templated on <ComplexT, R> can build
// and read a complex value without naming the vendor type; the paired
// instantiations (wwrFloatComplex, float) / (wwrDoubleComplex, double) keep the
// return type and ComplexT in step.

__device__ __forceinline__ wwrFloatComplex mk(const float re, const float im) {
  return wwr::make_wwrFloatComplex(re, im);
}
__device__ __forceinline__ wwrDoubleComplex mk(const double re, const double im) {
  return wwr::make_wwrDoubleComplex(re, im);
}
__device__ __forceinline__ float re_of(const wwrFloatComplex z) { return wwr::wwrCrealf(z); }
__device__ __forceinline__ float im_of(const wwrFloatComplex z) { return wwr::wwrCimagf(z); }
__device__ __forceinline__ double re_of(const wwrDoubleComplex z) { return wwr::wwrCreal(z); }
__device__ __forceinline__ double im_of(const wwrDoubleComplex z) { return wwr::wwrCimag(z); }

template<typename R>
__device__ __forceinline__ R abs_(const R x) {
  return x < R(0) ? -x : x;
}

/// Element (i, j) of the symmetric matrix whose lower (or upper) triangle is
/// stored. The other triangle is never read.
template<typename R>
__device__ __forceinline__ R sym_at(const R *A, const std::size_t lda, const bool lower,
                                     const std::size_t i, const std::size_t j) {
  const bool stored = lower ? (i >= j) : (i <= j);
  return stored ? A[i + j * lda] : A[j + i * lda];
}

/// How many of @p count ascending values are below @p key (or at most @p key,
/// when @p include_equal).
template<typename R>
__device__ int count_below(const R *v, const int count, const R key, const bool include_equal) {
  int lo = 0;
  int hi = count;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const bool below = include_equal ? (v[mid] <= key) : (v[mid] < key);
    if (below) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ── kernels ──────────────────────────────────────────────────────────────────

template<typename ComplexT, typename R>
__global__ void resolvents_kernel(const int lower, const int n, const R *A, const std::size_t lda,
                                  const FeastContour<R> contour, ComplexT *out,
                                  const std::size_t stride) {
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < nn;
       k += step) {
    const std::size_t i = k % static_cast<std::size_t>(n);
    const std::size_t j = k / static_cast<std::size_t>(n);
    const R a = sym_at(A, lda, lower != 0, i, j);
    const bool diag = (i == j);
    for (int e = 0; e < contour.count; ++e) {
      const R vr = (diag ? contour.zr[e] : R(0)) - a;
      const R vi = diag ? contour.zi[e] : R(0);
      out[static_cast<std::size_t>(e) * stride + k] = mk(vr, vi);
    }
  }
}

template<typename ComplexT>
__global__ void pointer_array_kernel(ComplexT *base, const std::size_t stride, const int count,
                                     ComplexT **ptrs) {
  const int e = static_cast<int>(threadIdx.x);
  if (e < count) {
    ptrs[e] = base + static_cast<std::size_t>(e) * stride;
  }
}

template<typename ComplexT, typename R>
__global__ void broadcast_kernel(const std::size_t elems, const R *Y, const int count,
                                 ComplexT *out, const std::size_t stride) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < elems;
       k += step) {
    const ComplexT v = mk(Y[k], R(0));
    for (int e = 0; e < count; ++e) {
      out[static_cast<std::size_t>(e) * stride + k] = v;
    }
  }
}

template<typename ComplexT, typename R>
__global__ void accumulate_kernel(const std::size_t elems, const FeastContour<R> contour,
                                  const ComplexT *X, const std::size_t stride, R *out) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < elems;
       k += step) {
    R acc = R(0);
    for (int e = 0; e < contour.count; ++e) {
      const ComplexT x = X[static_cast<std::size_t>(e) * stride + k];
      acc += contour.wr[e] * re_of(x) - contour.wi[e] * im_of(x); // Re(w x)
    }
    out[k] = acc;
  }
}

/// accumulate_kernel over split storage: X_e = Xr_e + i Xi_e, two real arrays.
template<typename R>
__global__ void accumulate_split_kernel(const std::size_t elems, const FeastContour<R> contour,
                                        const R *Xr, const R *Xi, const std::size_t stride,
                                        R *out) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < elems;
       k += step) {
    R acc = R(0);
    for (int e = 0; e < contour.count; ++e) {
      const std::size_t at = static_cast<std::size_t>(e) * stride + k;
      acc += contour.wr[e] * Xr[at] - contour.wi[e] * Xi[at]; // Re(w x)
    }
    out[k] = acc;
  }
}

/// One block per column j: compare the two forms' error bounds (feast_bridge.h),
/// copy r_j or x_j into B(:, j), and record the choice in form[j].
template<typename R>
__global__ void residual_select_kernel(const int n, const FeastContour<R> contour, const R *X,
                                       const R *lambda, const R *Rres, R *B, R *form) {
  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(n);
  R rr = R(0);
  R xx = R(0);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    rr += Rres[col + i] * Rres[col + i];
    xx += X[col + i] * X[col + i];
  }
  rr = block_reduce<kBlock>(rr, AddOp{});
  xx = block_reduce<kBlock>(xx, AddOp{});
  const R lam = lambda[blockIdx.x];
  R plain = R(0);
  R residual = R(0);
  for (int e = 0; e < contour.count; ++e) {
    const R w = sqrt(contour.wr[e] * contour.wr[e] + contour.wi[e] * contour.wi[e]);
    const R dr = contour.zr[e] - lam;
    const R d = sqrt(dr * dr + contour.zi[e] * contour.zi[e]);
    plain += w / contour.zi[e];
    residual += w / (contour.zi[e] * d);
  }
  // NaN anywhere compares false: the plain form, which surfaces it downstream.
  const bool use_residual = sqrt(rr) * residual < sqrt(xx) * plain;
  const R *src = use_residual ? Rres : X;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    B[col + i] = src[col + i];
  }
  if (threadIdx.x == 0) {
    form[blockIdx.x] = use_residual ? R(1) : R(0);
  }
}

/// The filter sum over B's solutions: form[j] = 1 gives the residual form,
/// c_e = w_e / (Z_e - lambda_j) and out = sum_e Re[ c_e (x + Xr_e + i Xi_e) ]
/// (Im Z_e > 0, so Z_e - lambda_j != 0); form[j] = 0 the plain sum_e Re[ w_e X_e ].
template<typename R>
__global__ void accumulate_residual_split_kernel(const std::size_t n, const std::size_t elems,
                                                 const FeastContour<R> contour, const R *X,
                                                 const R *lambda, const R *form, const R *Xr,
                                                 const R *Xi, const std::size_t stride, R *out) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < elems;
       k += step) {
    R acc = R(0);
    if (form[k / n] == R(0)) {
      for (int e = 0; e < contour.count; ++e) {
        const std::size_t at = static_cast<std::size_t>(e) * stride + k;
        acc += contour.wr[e] * Xr[at] - contour.wi[e] * Xi[at]; // Re(w x)
      }
      out[k] = acc;
      continue;
    }
    const R lam = lambda[k / n];
    const R x = X[k];
    for (int e = 0; e < contour.count; ++e) {
      // c = w / d with d = Z - lambda: (w conj(d)) / |d|^2.
      const R dr = contour.zr[e] - lam;
      const R di = contour.zi[e];
      const R inv = R(1) / (dr * dr + di * di);
      const R cr = (contour.wr[e] * dr + contour.wi[e] * di) * inv;
      const R ci = (contour.wi[e] * dr - contour.wr[e] * di) * inv;
      const std::size_t at = static_cast<std::size_t>(e) * stride + k;
      acc += cr * (x + Xr[at]) - ci * Xi[at];
    }
    out[k] = acc;
  }
}

/// AX(:, j) -= lambda_j X(:, j), elementwise.
template<typename R>
__global__ void ritz_residual_kernel(const std::size_t n, const std::size_t elems, const R *X,
                                     const R *lambda, R *AX) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < elems;
       k += step) {
    AX[k] -= lambda[k / n] * X[k];
  }
}

/// One block per column j: colsum[j] = sum_i |A(i, j)|.
template<typename R>
__global__ void sym_colsum_kernel(const int lower, const int n, const R *A, const std::size_t lda,
                                  R *colsum) {
  const std::size_t j = blockIdx.x;
  R acc = R(0);
  for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(n); i += blockDim.x) {
    acc += abs_(sym_at(A, lda, lower != 0, i, j));
  }
  acc = block_reduce<kBlock>(acc, AddOp{});
  if (threadIdx.x == 0) {
    colsum[j] = acc;
  }
}

/// One block: *out = max of count non-negative values (NaN wins).
template<typename R>
__global__ void max_kernel(const int count, const R *v, R *out) {
  R acc = R(0);
  for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(blockDim.x)) {
    acc = max_nan(acc, v[i]);
  }
  acc = block_reduce<kBlock>(acc, MaxNanOp{});
  if (threadIdx.x == 0) {
    *out = acc;
  }
}

/// One block: status->max_residual = max of the m0 residuals (NaN wins).
template<typename R>
__global__ void residual_max_kernel(const int m0, const R *residuals, FeastStatus<R> *status) {
  R acc = R(0);
  for (int i = static_cast<int>(threadIdx.x); i < m0; i += static_cast<int>(blockDim.x)) {
    acc = max_nan(acc, residuals[i]);
  }
  acc = block_reduce<kBlock>(acc, MaxNanOp{});
  if (threadIdx.x == 0) {
    status->max_residual = acc;
  }
}

template<typename R>
__global__ void select_kernel(const int m0, const R emin, const R emax, const R *ritz, const R *vecs,
                              R *lambda, R *rotated, FeastStatus<R> *status) {
  const int lo = count_below(ritz, m0, emin, false);
  const int hi = count_below(ritz, m0, emax, true);

  const std::size_t m0z = static_cast<std::size_t>(m0);
  const std::size_t mm = m0z * m0z;
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < mm + m0z;
       k += step) {
    if (k < mm) {
      const std::size_t i = k % m0z;
      const std::size_t j = k / m0z;
      rotated[k] = vecs[i + ((j + lo) % m0z) * m0z];
    } else {
      const std::size_t j = k - mm;
      lambda[j] = ritz[(j + lo) % m0z];
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->m = hi - lo;
    status->lo = lo;
  }
}

/// One block per Ritz pair.
template<typename R>
__global__ void residuals_kernel(const int n, const R *X, const R *AX, const R *lambda,
                                 const R *norm_a, R *residuals, FeastStatus<R> *status) {
  const int j = static_cast<int>(blockIdx.x);
  const bool inactive = (j >= status->m); // uniform across the block
  if (inactive) {
    if (threadIdx.x == 0) {
      residuals[j] = R(0);
    }
    return;
  }

  const R lam = lambda[j];
  const R *x = X + static_cast<std::size_t>(j) * n;
  const R *ax = AX + static_cast<std::size_t>(j) * n;

  R r = R(0);
  R q = R(0);
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const R xi = x[i];
    r += abs_(ax[i] - lam * xi);
    q += abs_(xi);
  }
  r = block_reduce<kBlock>(r, AddOp{});
  q = block_reduce<kBlock>(q, AddOp{});

  if (threadIdx.x == 0) {
    residuals[j] = r / ((*norm_a + abs_(lam)) * q);
  }
}

} // namespace

// ── launchers ──────────────────────────────────────────────────────────────
//
// Each returns void; a launch-configuration failure surfaces through
// wwr::wwrGetLastError() on the host side, the gebal_bridge.h pattern. The
// module validates every dimension before calling, so the guards here are the
// backstop that keeps a bad argument from launching, not the reporting path.

template<typename ComplexT, typename RealT>
void feast_resolvents(const wwr::wwrStream_t stream, const bool lower, const int n,
                      const RealT *const d_A, const int lda, const FeastContour<RealT> contour,
                      ComplexT *const d_out, const std::size_t stride) {
  if (n < 1 || lda < n || d_A == nullptr || d_out == nullptr) {
    return;
  }
  if (contour.count < 1 || contour.count > kFeastMaxNodes) {
    return;
  }
  const std::size_t nn = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  if (stride < nn) {
    return;
  }
  resolvents_kernel<ComplexT, RealT><<<blocks_for(nn), kBlock, 0, stream>>>(
      lower ? 1 : 0, n, d_A, static_cast<std::size_t>(lda), contour, d_out, stride);
}

template<typename ComplexT>
void feast_pointer_array(const wwr::wwrStream_t stream, ComplexT *const d_base,
                         const std::size_t stride, const int count, ComplexT **const d_ptrs) {
  if (d_base == nullptr || d_ptrs == nullptr) {
    return;
  }
  if (count < 1 || count > kFeastMaxNodes) {
    return;
  }
  pointer_array_kernel<ComplexT>
      <<<1, kFeastMaxNodes, 0, stream>>>(d_base, stride, count, d_ptrs);
}

template<typename ComplexT, typename RealT>
void feast_broadcast(const wwr::wwrStream_t stream, const std::size_t elems, const RealT *const d_Y,
                     const int count, ComplexT *const d_out, const std::size_t stride) {
  if (d_Y == nullptr || d_out == nullptr || stride < elems) {
    return;
  }
  if (count < 1 || count > kFeastMaxNodes) {
    return;
  }
  if (elems == 0) {
    return;
  }
  broadcast_kernel<ComplexT, RealT>
      <<<blocks_for(elems), kBlock, 0, stream>>>(elems, d_Y, count, d_out, stride);
}

template<typename ComplexT, typename RealT>
void feast_accumulate(const wwr::wwrStream_t stream, const std::size_t elems,
                      const FeastContour<RealT> contour, const ComplexT *const d_X,
                      const std::size_t stride, RealT *const d_out) {
  if (d_X == nullptr || d_out == nullptr || stride < elems) {
    return;
  }
  if (contour.count < 1 || contour.count > kFeastMaxNodes) {
    return;
  }
  if (elems == 0) {
    return;
  }
  accumulate_kernel<ComplexT, RealT>
      <<<blocks_for(elems), kBlock, 0, stream>>>(elems, contour, d_X, stride, d_out);
}

template<typename RealT>
void feast_accumulate_split(const wwr::wwrStream_t stream, const std::size_t elems,
                            const FeastContour<RealT> contour, const RealT *const d_Xr,
                            const RealT *const d_Xi, const std::size_t stride,
                            RealT *const d_out) {
  if (d_Xr == nullptr || d_Xi == nullptr || d_out == nullptr || stride < elems) {
    return;
  }
  if (contour.count < 1 || contour.count > kFeastMaxNodes) {
    return;
  }
  if (elems == 0) {
    return;
  }
  accumulate_split_kernel<RealT>
      <<<blocks_for(elems), kBlock, 0, stream>>>(elems, contour, d_Xr, d_Xi, stride, d_out);
}

template<typename RealT>
void feast_residual_select(const wwr::wwrStream_t stream, const int n, const int k,
                           const FeastContour<RealT> contour, const RealT *const d_X,
                           const RealT *const d_lambda, const RealT *const d_R,
                           RealT *const d_B, RealT *const d_form) {
  if (n < 1 || k < 1 || d_X == nullptr || d_lambda == nullptr || d_R == nullptr ||
      d_B == nullptr || d_form == nullptr) {
    return;
  }
  if (contour.count < 1 || contour.count > kFeastMaxNodes) {
    return;
  }
  residual_select_kernel<RealT><<<static_cast<unsigned int>(k), kBlock, 0, stream>>>(
      n, contour, d_X, d_lambda, d_R, d_B, d_form);
}

template<typename RealT>
void feast_accumulate_residual_split(const wwr::wwrStream_t stream, const int n, const int k,
                                     const FeastContour<RealT> contour, const RealT *const d_X,
                                     const RealT *const d_lambda, const RealT *const d_form,
                                     const RealT *const d_Xr, const RealT *const d_Xi,
                                     const std::size_t stride, RealT *const d_out) {
  if (n < 1 || k < 1) {
    return;
  }
  const std::size_t nz = static_cast<std::size_t>(n);
  const std::size_t elems = nz * static_cast<std::size_t>(k);
  if (d_X == nullptr || d_lambda == nullptr || d_form == nullptr || d_Xr == nullptr ||
      d_Xi == nullptr || d_out == nullptr || stride < elems) {
    return;
  }
  if (contour.count < 1 || contour.count > kFeastMaxNodes) {
    return;
  }
  accumulate_residual_split_kernel<RealT><<<blocks_for(elems), kBlock, 0, stream>>>(
      nz, elems, contour, d_X, d_lambda, d_form, d_Xr, d_Xi, stride, d_out);
}

template<typename RealT>
void feast_ritz_residual_block(const wwr::wwrStream_t stream, const int n, const int k,
                               const RealT *const d_X, const RealT *const d_lambda,
                               RealT *const d_AX) {
  if (n < 1 || k < 1 || d_X == nullptr || d_lambda == nullptr || d_AX == nullptr) {
    return;
  }
  const std::size_t nz = static_cast<std::size_t>(n);
  const std::size_t elems = nz * static_cast<std::size_t>(k);
  ritz_residual_kernel<RealT>
      <<<blocks_for(elems), kBlock, 0, stream>>>(nz, elems, d_X, d_lambda, d_AX);
}

template<typename RealT>
void feast_sym_norm1(const wwr::wwrStream_t stream, const bool lower, const int n,
                     const RealT *const d_A, const int lda, RealT *const d_colsum,
                     RealT *const d_norm) {
  if (n < 1 || lda < n || d_A == nullptr || d_colsum == nullptr || d_norm == nullptr) {
    return;
  }
  sym_colsum_kernel<RealT><<<static_cast<unsigned int>(n), kBlock, 0, stream>>>(
      lower ? 1 : 0, n, d_A, static_cast<std::size_t>(lda), d_colsum);
  max_kernel<RealT><<<1, kBlock, 0, stream>>>(n, d_colsum, d_norm);
}

template<typename RealT>
void feast_select(const wwr::wwrStream_t stream, const int m0, const RealT emin, const RealT emax,
                  const RealT *const d_ritz, const RealT *const d_vecs, RealT *const d_lambda,
                  RealT *const d_rotated, FeastStatus<RealT> *const d_status) {
  if (m0 < 1 || !(emin < emax)) {
    return;
  }
  if (d_ritz == nullptr || d_vecs == nullptr || d_lambda == nullptr || d_rotated == nullptr ||
      d_status == nullptr) {
    return;
  }
  const std::size_t total = static_cast<std::size_t>(m0) * static_cast<std::size_t>(m0) +
                            static_cast<std::size_t>(m0);
  select_kernel<RealT><<<blocks_for(total), kBlock, 0, stream>>>(m0, emin, emax, d_ritz, d_vecs,
                                                                 d_lambda, d_rotated, d_status);
}

template<typename RealT>
void feast_residuals(const wwr::wwrStream_t stream, const int n, const int m0, const RealT *const d_X,
                     const RealT *const d_AX, const RealT *const d_lambda, const RealT *const d_norm,
                     RealT *const d_residuals, FeastStatus<RealT> *const d_status) {
  if (n < 1 || m0 < 1) {
    return;
  }
  if (d_X == nullptr || d_AX == nullptr || d_lambda == nullptr || d_norm == nullptr ||
      d_residuals == nullptr || d_status == nullptr) {
    return;
  }
  residuals_kernel<RealT><<<static_cast<unsigned int>(m0), kBlock, 0, stream>>>(
      n, d_X, d_AX, d_lambda, d_norm, d_residuals, d_status);
  residual_max_kernel<RealT><<<1, kBlock, 0, stream>>>(m0, d_residuals, d_status);
}

// One instantiation per supported type, matching feast_bridge.h's declarations:
// the complex launchers for (wwrFloatComplex, float) and (wwrDoubleComplex,
// double), the real-only ones for float and double. FEAST is real-symmetric
// only, so there is no complex-input instantiation.

template void feast_resolvents<wwrFloatComplex, float>(wwr::wwrStream_t, bool, int, const float *,
                                                       int, FeastContour<float>, wwrFloatComplex *,
                                                       std::size_t);
template void feast_resolvents<wwrDoubleComplex, double>(wwr::wwrStream_t, bool, int, const double *,
                                                         int, FeastContour<double>,
                                                         wwrDoubleComplex *, std::size_t);

template void feast_pointer_array<wwrFloatComplex>(wwr::wwrStream_t, wwrFloatComplex *, std::size_t,
                                                   int, wwrFloatComplex **);
template void feast_pointer_array<wwrDoubleComplex>(wwr::wwrStream_t, wwrDoubleComplex *,
                                                    std::size_t, int, wwrDoubleComplex **);

template void feast_broadcast<wwrFloatComplex, float>(wwr::wwrStream_t, std::size_t, const float *,
                                                      int, wwrFloatComplex *, std::size_t);
template void feast_broadcast<wwrDoubleComplex, double>(wwr::wwrStream_t, std::size_t,
                                                        const double *, int, wwrDoubleComplex *,
                                                        std::size_t);

template void feast_accumulate<wwrFloatComplex, float>(wwr::wwrStream_t, std::size_t,
                                                       FeastContour<float>, const wwrFloatComplex *,
                                                       std::size_t, float *);
template void feast_accumulate<wwrDoubleComplex, double>(wwr::wwrStream_t, std::size_t,
                                                         FeastContour<double>,
                                                         const wwrDoubleComplex *, std::size_t,
                                                         double *);

template void feast_accumulate_split<float>(wwr::wwrStream_t, std::size_t, FeastContour<float>,
                                           const float *, const float *, std::size_t, float *);
template void feast_accumulate_split<double>(wwr::wwrStream_t, std::size_t, FeastContour<double>,
                                            const double *, const double *, std::size_t,
                                            double *);

template void feast_residual_select<float>(wwr::wwrStream_t, int, int, FeastContour<float>,
                                           const float *, const float *, const float *, float *,
                                           float *);
template void feast_residual_select<double>(wwr::wwrStream_t, int, int, FeastContour<double>,
                                            const double *, const double *, const double *,
                                            double *, double *);
template void feast_accumulate_residual_split<float>(wwr::wwrStream_t, int, int,
                                                     FeastContour<float>, const float *,
                                                     const float *, const float *, const float *,
                                                     const float *, std::size_t, float *);
template void feast_accumulate_residual_split<double>(wwr::wwrStream_t, int, int,
                                                      FeastContour<double>, const double *,
                                                      const double *, const double *,
                                                      const double *, const double *,
                                                      std::size_t, double *);

template void feast_ritz_residual_block<float>(wwr::wwrStream_t, int, int, const float *,
                                               const float *, float *);
template void feast_ritz_residual_block<double>(wwr::wwrStream_t, int, int, const double *,
                                                const double *, double *);

template void feast_sym_norm1<float>(wwr::wwrStream_t, bool, int, const float *, int, float *,
                                     float *);
template void feast_sym_norm1<double>(wwr::wwrStream_t, bool, int, const double *, int, double *,
                                      double *);

template void feast_select<float>(wwr::wwrStream_t, int, float, float, const float *, const float *,
                                  float *, float *, FeastStatus<float> *);
template void feast_select<double>(wwr::wwrStream_t, int, double, double, const double *,
                                   const double *, double *, double *, FeastStatus<double> *);

template void feast_residuals<float>(wwr::wwrStream_t, int, int, const float *, const float *,
                                     const float *, const float *, float *, FeastStatus<float> *);
template void feast_residuals<double>(wwr::wwrStream_t, int, int, const double *, const double *,
                                      const double *, const double *, double *,
                                      FeastStatus<double> *);

} // namespace calaman::device
