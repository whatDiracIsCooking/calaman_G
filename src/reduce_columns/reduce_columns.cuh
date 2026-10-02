/// @file
/// @brief Backend-neutral, header-only per-column reduction of a column-major
///        matrix, with a per-element pre-transform and a binary fold
///
/// For each column j, computes `reduce(pre(a_0j), pre(a_1j), ...)` under an
/// associative binary @p op and reports it in `d_result[j]`. A per-element
/// unary @p pre runs before the fold, so the input type T and the reduction
/// type ValT may differ (e.g. complex -> real for a norm). A thin
/// reduce_columns() wraps the common ValT == T, identity-pre case.
///
/// `#include`d directly into a .cu (CUDA) or `-x hip` device-compiled (HIP)
/// translation unit, exactly like wwr.extension.parallel_for's parallel_for.cuh
/// -- there is no module at the point of use, so it reaches the backend through
/// the gpu* layer's runtime.cuh rather than gpu_backend.h. Link
/// calaman.reduce_columns (which carries wwr.device + wwr.extension.bridge).
///
/// Thrust is deliberately NOT used. WarpWraps' parallel_for.cuh documents why a
/// hand-written kernel replaced the Thrust path (its algorithms break under
/// relocatable device code); the segmented-reduce here follows that precedent
/// rather than reaching for thrust::reduce_by_key. The kernel also needs no
/// caller-supplied key/scratch buffer: one block owns one column and writes its
/// one output directly.
///
/// The fold is IDENTITY-FREE: each active thread seeds its partial with its
/// first element (not with an identity op would need), then the block tree-reduce
/// combines only the valid partials via an activity bound. So @p op need only be
/// associative; it needs no identity and no commutativity.
#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

// WWR_WARP_SIZE (the block-size base) and wwrStream_t for the launcher
// signature. runtime.cuh is also the device-pass gate: it #errors outside a
// CUDA or HIP device compile, so this header carries no guard of its own, like
// parallel_for.cuh.
#include "extension/bridge/gpu_stream_bridge.h"
#include "runtime.cuh"

namespace calaman::device {

/// @brief A trivially-copyable class functor usable as the binary fold `ValT(ValT, ValT)`
///
/// The callability probe `{ f(a, b) } -> ValT` is deliberately NOT here: it would
/// run in HOST context and reject a `__device__`-only `operator()`. It lives on
/// the kernel template below instead, a device entity. This mirrors
/// wwr.extension.parallel_for's `device_functor`, which splits the checks the
/// same way and for the same reason. @p ValT names the value type the call sites
/// read as documentation; the host side here checks only the storage shape.
template<typename F, typename ValT>
concept binary_op_functor = std::is_trivially_copyable_v<F> && std::is_class_v<F>;

/// @brief A trivially-copyable class functor usable as the pre-transform `ValT(T)`
///
/// Same split as binary_op_functor: the `{ f(a) } -> ValT` probe is on the
/// kernel, not here.
template<typename F, typename T, typename ValT>
concept unary_transform_functor = std::is_trivially_copyable_v<F> && std::is_class_v<F>;

/// @brief Pre-transform that passes each element through unchanged (ValT == T)
///
/// Seeds the common case reduce_columns() wraps. `__host__ __device__` so the
/// same type is usable either side of the host/device line in a device TU.
template<typename T>
struct identity_functor {
  __host__ __device__ T operator()(const T &x) const { return x; }
};

namespace kernels {

/// @brief [kernel] Reduce one column per block: `d_result[j] = fold_i pre(A(i, j))`
///
/// The `requires` carries the callability checks the concepts above cannot: a
/// `__global__` template is a device entity, so the probes accept the functors'
/// `__device__`-only `operator()`.
///
/// Block j == blockIdx.x owns column j (contiguous but for the leading
/// dimension @p lda). Each thread folds its strided share of the @p n rows into
/// a private partial, seeded by its FIRST element so no identity is needed; the
/// partials then tree-reduce in shared memory. The `tid + s < nactive` bound
/// skips the ragged tail, so only the valid partials are ever combined. @p n is
/// at least 1 (the launcher returns early otherwise), so every block has at
/// least one active thread.
template<typename T, typename ValT, typename UnaryOp, typename BinaryOp>
  requires requires(const UnaryOp pre, const BinaryOp op, const T t, const ValT v) {
    { pre(t) } -> std::same_as<ValT>;
    { op(v, v) } -> std::same_as<ValT>;
  }
__global__ void reduce_columns_kernel(const T *const d_A, ValT *const d_result, const std::size_t n,
                                      const std::size_t lda, const UnaryOp pre, const BinaryOp op) {
  // 4 warps per block, matching lacpy/horner and parallel_for. A configure-time
  // value (WWR_WARP_SIZE: 32 default, 64 on CDNA), so 128 or 256, and a power of
  // two -- the halving tree-reduce below relies on that.
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  static_assert(kBlockSize <= 1024,
                "block size exceeds the 1024 threads/block both backends cap at");
  __shared__ ValT sdata[kBlockSize];

  const std::size_t col = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  const T *const column = d_A + col * lda; // column j starts at A + j*lda

  // Threads that get at least one row: min(n, kBlockSize). n >= 1, so >= 1.
  const unsigned int nactive =
      (n < static_cast<std::size_t>(kBlockSize)) ? static_cast<unsigned int>(n) : kBlockSize;

  if (tid < nactive) {
    // Seed with this thread's first element (identity-free), then fold the rest
    // of its strided share -- rows tid, tid + kBlockSize, tid + 2*kBlockSize, ...
    ValT acc = pre(column[tid]);
    for (std::size_t i = static_cast<std::size_t>(tid) + kBlockSize; i < n; i += kBlockSize) {
      acc = op(acc, pre(column[i]));
    }
    sdata[tid] = acc;
  }
  __syncthreads();

  // Tree-reduce the nactive valid partials into sdata[0]. `tid + s < nactive`
  // skips slots past the ragged tail, so no identity element is ever read.
  for (unsigned int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s && tid + s < nactive) {
      sdata[tid] = op(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_result[col] = sdata[0];
  }
}

} // namespace kernels

/// @brief Reduce each column of a column-major matrix with a pre-transform and a fold
///
/// Applies @p pre element-wise before accumulating under @p op, so the input
/// type T and the reduction type ValT may differ. Enqueues one kernel on
/// @p stream and returns WITHOUT synchronizing; the caller synchronizes when it
/// needs @p d_result. Launches nothing when @p n or @p ncols is 0.
///
/// @tparam T        Input element type
/// @tparam ValT     Reduction/output value type; may differ from T
/// @tparam UnaryOp  Per-element pre-transform `ValT(T)`, trivially-copyable class
/// @tparam BinaryOp Associative fold `ValT(ValT, ValT)`, trivially-copyable class
/// @param stream   Stream the launch is enqueued on; A and result live on its device
/// @param d_A      Column-major matrix; column j starts at `d_A + j*lda`
/// @param d_result Output of length @p ncols; `d_result[j]` is the fold of column j
/// @param n        Rows reduced per column
/// @param ncols    Number of columns (one block each; must fit a 32-bit grid dim)
/// @param lda      Leading dimension (column stride); lda >= n, so columns do not overlap
/// @param pre      Per-element pre-transform applied before the fold
/// @param op       Associative binary fold (no identity or commutativity required)
template<typename T, typename ValT, unary_transform_functor<T, ValT> UnaryOp,
         binary_op_functor<ValT> BinaryOp>
void reduce_columns_transform(const wwr::wwrStream_t stream, const T *const d_A,
                              ValT *const d_result, const std::size_t n, const std::size_t ncols,
                              const std::size_t lda, const UnaryOp pre, const BinaryOp op) {
  if (n == 0 || ncols == 0) {
    return;
  }
  constexpr unsigned int kBlockSize = 4 * WWR_WARP_SIZE;
  // One block per column in x. Its 2^31-1 bound maps any realistic column count.
  kernels::reduce_columns_kernel<T, ValT>
      <<<static_cast<unsigned int>(ncols), kBlockSize, 0, stream>>>(d_A, d_result, n, lda, pre, op);
}

/// @brief Reduce each column with a binary fold, no pre-transform (ValT == T)
///
/// Convenience wrapper over reduce_columns_transform() with identity_functor<T>:
/// elements pass through unchanged, so `d_result[j]` is the fold of column j's
/// raw entries. Same async/empty-input contract as reduce_columns_transform.
///
/// @tparam T        Element type
/// @tparam BinaryOp Associative fold `T(T, T)`, trivially-copyable class
template<typename T, binary_op_functor<T> BinaryOp>
void reduce_columns(const wwr::wwrStream_t stream, const T *const d_A, T *const d_result,
                    const std::size_t n, const std::size_t ncols, const std::size_t lda,
                    const BinaryOp op) {
  reduce_columns_transform<T, T>(stream, d_A, d_result, n, ncols, lda, identity_functor<T>{}, op);
}

} // namespace calaman::device
