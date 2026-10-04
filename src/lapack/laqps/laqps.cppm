/**
 * @file laqps.cppm
 * @brief The calaman.laqps module -- the blocked, level-3 Businger-Golub
 *        pivoted-QR panel, LAPACK's ?laqps
 *
 * Factors up to @p nb columns of the trailing submatrix of an m-by-n
 * column-major matrix A, starting at row/column @p offset, with Businger-Golub
 * column pivoting, and reports in @p kb how many columns were actually factored
 * (kb < nb when a column norm collapses). This is the panel the blocked geqp3
 * driver (#12) runs in a loop; laqp2 finishes the tail. Templated over `float`
 * and `double`.
 *
 * The level-3 idea (LAPACK ?laqps / ?geqp3): instead of applying each new
 * reflector to the WHOLE trailing matrix immediately -- laqp2's per-column
 * larf -- laqps accumulates an auxiliary matrix F so that after kb columns ONE
 * gemm applies the whole block's effect to the rest of the trailing matrix:
 *
 *   A(offset+kb:m, kb:n) -= A(offset+kb:m, 0:kb) * F(kb:n, 0:kb)^T
 *
 * Within the block, step k (0-based, rk = offset + k the pivot row) is a host
 * composition of wrapped BLAS plus larfg:
 *   1. Pivot: iamax over vn1[k:n]; if the pivot is not column k, swap columns k
 *      and pvt of A, ROWS k and pvt of F, and the vn1/vn2/jpvt entries.
 *   2. Apply the block so far to the pivot column via one gemv against F:
 *      A(rk:m, k) -= A(rk:m, 0:k) * F(k, 0:k)^T.
 *   3. larfg on A(rk:m, k) -> tau[k]; set A(rk,k) = 1 for the F builds.
 *   4. Build F's new column k: F(k+1:n, k) = tau[k] * A(rk:m, k+1:n)^T * A(rk:m, k)
 *      (a gemv), then fold in the accumulated part via auxv:
 *      auxv(0:k) = -tau[k] * A(rk:m, 0:k)^T * A(rk:m, k);
 *      F(0:n, k) += F(0:n, 0:k) * auxv(0:k).
 *   5. Update only the CURRENT row of A for the next pivot's iamax:
 *      A(rk, k+1:n) -= A(rk, 0:k+1) * F(k+1:n, 0:k+1)^T (a gemv), restore A(rk,k).
 *   6. Downdate the trailing partial norms on the device (see below).
 * After kb steps the deferred gemm (step above) applies the block to the rest.
 *
 * THE DEFERRED NORM-DOWNDATE (the departure from the serial reference). Because
 * the trailing block is not updated until the panel gemm, a column flagged as
 * degraded during an in-block downdate cannot be recomputed until AFTER that
 * gemm. The reference LAPACK threads such columns onto a serial linked list
 * (LSTICC, stored in VN2); this replaces that list with a DEVICE MASK: the
 * downdate kernel (laqps.cu) only ever RAISES a flag, the host zeroes the mask
 * once before step 1, and after the panel gemm the host reads the mask back,
 * COMPACTS the flagged global column indices, and recomputes each flagged
 * column's norm exactly with wwr::nrm2 of its now-updated trailing part. The
 * recompute loop is the same host-driven wwr::nrm2 laqp2 uses, run once per
 * block rather than once per step.
 *
 * Allocation-free shipped surface (CLAUDE.md, test/shared/README.md): A, tau,
 * vn1, vn2 are caller-provided device pointers as in laqp2, and laqps adds three
 * more caller-provided scratch pointers for the level-3 machinery -- the
 * auxiliary matrix F (n-by-nb, device), the vector auxv (length >= nb, device),
 * and the int mask flags (length >= n, device). jpvt is a HOST int array laqps
 * UPDATES (not initialises -- the driver owns the running permutation across
 * blocks), swapping jpvt[k] and jpvt[pvt] on each pivot like laqp2 swaps its own.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/laqp2/geqp3: the
 * iamax index, larfg's host scalars, and the per-step scalar reads/writes are
 * all host-side and block, on the handle's own stream. Complex ?laqps is a
 * deliberate later extension, for the reasons larfg/laqp2 document.
 *
 * Usage:
 *   import calaman.laqps;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   // d_A: m x n; d_tau: length min(m,n); d_vn1,d_vn2: length n;
 *   // d_F: n x nb; d_auxv: length nb; d_flags: int length n; jpvt: host int[n]
 *   int kb = 0;
 *   calaman::laqps<double>(handle, m, n, offset, nb, &kb, d_A, lda, jpvt, d_tau,
 *                          d_vn1, d_vn2, d_F, n, d_auxv, d_flags);
 */

module;

#include "laqps_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.laqps;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_OP_*, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
import wwr.wrappers.blas; // iamax, swap, nrm2, gemv, gemm
import calaman.larfg;     // calaman::larfg
import std;               // std::sqrt, std::min, std::vector

// export import, not a plain import: laqps RETURNS calaman::Status, so a consumer
// of `import calaman.laqps;` must see Status's member functions, not just its
// name -- the same re-export diff_norm does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// laqps_detail, not an anonymous namespace: these helpers are named by the
// exported laqps template's body, which is instantiated in every importer's TU.
// A named (unexported) namespace gives them module linkage, reachable by the
// instantiation yet absent from the module's public surface (as laqp2 does). The
// namespace is MODULE-SPECIFIC (laqps_detail, not a bare detail) because
// calaman.laqp2 defines an identically-named swap_device_scalar: once the two
// live in separate modules, a consumer that imports both (calaman.geqp3) would
// see one qualified name attached to two modules -- ill-formed -- unless the
// namespaces differ.
namespace laqps_detail {

/// @brief Swap two single device scalars through a host staging pair, on @p stream
///
/// Pivot bookkeeping swaps vn1[k]/vn1[pvt] (and vn2) a scalar at a time; there is
/// no BLAS swap of length one worth a kernel launch, so this reads both back,
/// exchanges them on the host, and writes them returned -- blocking, like the
/// larfg scalar reads. Returns the first failing copy's (runtime-domain) Status,
/// otherwise success.
template<typename T>
Status swap_device_scalar(wwr::wwrStream_t stream, T *a, T *b) {
  T ha{};
  T hb{};
  CLM_TRY(wwr::wwrMemcpyAsync(&ha, a, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(&hb, b, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  CLM_TRY(wwr::wwrMemcpyAsync(a, &hb, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrMemcpyAsync(b, &ha, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

/// @brief Write one device scalar from a host value on @p stream, blocking
template<typename T>
Status set_device_scalar(wwr::wwrStream_t stream, T *dst, const T &value) {
  CLM_TRY(wwr::wwrMemcpyAsync(dst, &value, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace laqps_detail

/// @brief Factor up to @p nb columns of A's trailing submatrix (LAPACK ?laqps)
///
/// Blocked Businger-Golub pivoted QR of the panel A(offset:m, 0:n): factors kb
/// columns (kb <= nb, written to @p kb), overwriting A with R in the upper
/// trapezoid and reflector j below the diagonal of column j, filling tau[0:kb],
/// accumulating the auxiliary F, updating jpvt / vn1 / vn2, and applying the
/// block to the rest of the trailing matrix with ONE deferred gemm. The degraded
/// trailing columns are recomputed after that gemm via the device mask @p flags.
///
/// Short-circuits: the first failing step's Status is returned and the
/// factorization stops there. Returns success with kb = 0 when the panel is
/// empty. Each device read/write now returns its OWN domain's Status (the copies
/// a runtime one) rather than masquerading as a BLAS code.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; all device pointers live on its device
/// @param m Row count of A
/// @param n Column count of A (the panel spans all n columns at and after @p offset)
/// @param offset Rows already factored above this panel; vn1/vn2 describe A(offset:m, :)
/// @param nb Maximum number of columns to factor in this block
/// @param kb Out: the number of columns actually factored (0 <= *kb <= nb)
/// @param A Device matrix, m by n, column-major, overwritten with R and the reflectors
/// @param lda Leading dimension of A (>= m)
/// @param jpvt Host int array, length n; 1-based permutation, UPDATED by each pivot swap
/// @param tau Device array, length >= min(m - offset, n); the reflector scalars
/// @param vn1 Device array, length n; running partial column norms (updated)
/// @param vn2 Device array, length n; original partial column norms (updated on recompute)
/// @param F Device auxiliary matrix, n by nb, column-major, leading dimension @p ldf (scratch)
/// @param ldf Leading dimension of F (>= n)
/// @param auxv Device workspace, length >= nb; the per-step accumulated-F intermediate
/// @param flags Device int mask, length >= n; the degraded-column flags (written here)
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status laqps(wwr::wwrblasHandle_t handle, const int m, const int n, const int offset, const int nb,
             int *kb, T *A, const int lda, int *jpvt, T *tau, T *vn1, T *vn2, T *F, const int ldf,
             T *auxv, int *flags) {
  *kb = 0;
  if (m <= 0 || n <= 0 || nb <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // The block factors at most this many columns: nb, but no more than remain in
  // the trailing panel (min(m - offset, n)), and never a column at or below the
  // last reflectable row.
  const int mn = std::min(m - offset, n);
  const int block = std::min(nb, mn);
  if (block <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  const T tol3z = std::sqrt(std::numeric_limits<T>::epsilon());
  const T one = T{1};
  const T zero = T{0};
  const T neg_one = T{-1};
  using std::size_t;

  // The persistent degraded-column mask starts all-zero: a step only ever RAISES
  // a flag, never clears it, so a column degraded on any step stays flagged
  // until the post-gemm recompute. Clear the whole n-length array once here.
  device::laqps_clear_flags(stream, static_cast<size_t>(n), flags);

  int k = 0;
  for (; k < block; ++k) {
    const int rk = offset + k; // pivot row in A
    const int rows = m - rk;   // reflector length / trailing-block rows at/below rk

    // --- 1. Pivot over vn1[k:n] (host pointer mode writes a host int). iamax is
    // 1-based into the tail, so the global pivot column is k + (idx - 1).
    const int tail = n - k;
    int idx = 1;
    if (tail > 1) {
      CLM_TRY(wwr::iamax<T, int>(handle, tail, vn1 + k, 1, &idx));
    }
    const int pvt = k + (idx - 1);

    if (pvt != k) {
      // Swap the two full m-length columns of A, the two length-k ROWS of F
      // (F(k,0:k) <-> F(pvt,0:k), stride ldf), the vn1/vn2 entries, and jpvt.
      CLM_TRY(wwr::swap<T, int>(handle, m, A + static_cast<size_t>(k) * lda, 1,
                                A + static_cast<size_t>(pvt) * lda, 1));
      if (k > 0) {
        CLM_TRY(wwr::swap<T, int>(handle, k, F + k, ldf, F + pvt, ldf));
      }
      CLM_TRY(laqps_detail::swap_device_scalar<T>(stream, vn1 + k, vn1 + pvt));
      CLM_TRY(laqps_detail::swap_device_scalar<T>(stream, vn2 + k, vn2 + pvt));
      const int tmp = jpvt[k];
      jpvt[k] = jpvt[pvt];
      jpvt[pvt] = tmp;
    }

    T *col = A + static_cast<size_t>(k) * lda + rk; // A(rk, k)

    // --- 2. Apply the already-factored part of the block to the pivot column:
    // A(rk:m, k) -= A(rk:m, 0:k) * F(k, 0:k)^T. One gemv, op = N (rows x k), the
    // vector being row k of F (F(k,0), stride ldf).
    if (k > 0) {
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, rows, k, &neg_one,
                           A + static_cast<size_t>(rk), lda, F + k, ldf, &one, col, 1));
    }

    // --- 3. Reflector on A(rk:m, k). larfg writes tau[k] (host) and beta, and
    // overwrites A(rk,k) with beta, A(rk+1:m, k) with the scaled tail.
    T host_tau{};
    T beta{};
    CLM_TRY(larfg<T>(handle, rows, col, col + 1, 1, &host_tau, &beta));
    CLM_TRY(laqps_detail::set_device_scalar<T>(stream, tau + k, host_tau));

    // Set A(rk,k) = 1 so `col` is the full reflector v with v[0] == 1, the form
    // the F-building gemvs below need. Restored to beta after step 5.
    CLM_TRY(laqps_detail::set_device_scalar<T>(stream, col, one));

    const int trail = n - k - 1; // trailing columns k+1:n

    // --- 4a. New F column: F(k+1:n, k) = tau[k] * A(rk:m, k+1:n)^T * A(rk:m, k).
    // gemv op = T over the trailing block (rows x trail), result length trail at
    // F(k+1, k). F(0:k+1, k) is left for step 4b / the auxv fold (auxv zeroes it
    // via beta=0 on the second gemv's output range? no -- F(0:k,k) is written by
    // 4b, F(k,k) stays 0). Explicitly zero F(0:k+1, k) first so the accumulate in
    // 4b lands on a clean slate and the deferred gemm reads a well-defined F.
    for (int i = 0; i <= k && trail >= 0; ++i) {
      CLM_TRY(
          laqps_detail::set_device_scalar<T>(stream, F + static_cast<size_t>(k) * ldf + i, zero));
    }
    if (trail > 0) {
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, rows, trail, &host_tau,
                           A + static_cast<size_t>(k + 1) * lda + rk, lda, col, 1, &zero,
                           F + static_cast<size_t>(k) * ldf + (k + 1), 1));
    }

    // --- 4b. Fold the accumulated block into F's new column:
    //   auxv(0:k) = -tau[k] * A(rk:m, 0:k)^T * A(rk:m, k)   (gemv op = T)
    //   F(0:n, k) += F(0:n, 0:k) * auxv(0:k)                (gemv op = N)
    if (k > 0) {
      const T neg_tau = -host_tau;
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_T, rows, k, &neg_tau,
                           A + static_cast<size_t>(rk), lda, col, 1, &zero, auxv, 1));
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, n, k, &one, F, ldf, auxv, 1, &one,
                           F + static_cast<size_t>(k) * ldf, 1));
    }

    // --- 5. Update only the CURRENT row of A, so the next step's iamax over vn1
    // sees a correctly-downdated row:
    //   A(rk, k+1:n) -= A(rk, 0:k+1) * F(k+1:n, 0:k+1)^T.
    // gemv op = N, (trail x (k+1)) matrix F(k+1, 0) (ldf), vector the row A(rk,0)
    // (stride lda), result the row A(rk, k+1) (stride lda).
    if (trail > 0) {
      CLM_TRY(wwr::gemv<T>(handle, wwr::WWRBLAS_OP_N, trail, k + 1, &neg_one, F + (k + 1), ldf,
                           A + static_cast<size_t>(rk), lda, &one,
                           A + static_cast<size_t>(k + 1) * lda + rk, lda));
    }

    // Restore A(rk,k) = beta (the reflector's leading R entry).
    CLM_TRY(laqps_detail::set_device_scalar<T>(stream, col, beta));

    // --- 6. Deferred downdate of the trailing partial norms. The kernel reads
    // the just-updated row A(rk, k+1:n) and the vn1/vn2 tails, shrinks the cheap
    // columns in place and RAISES flags for the degraded ones (never clears),
    // only when this step's row is not the last reflectable row (rk+1 < m, i.e.
    // there is a sub-diagonal trailing part left to recompute later).
    if (trail > 0 && rk + 1 < m) {
      const size_t count = static_cast<size_t>(trail);
      device::laqps_downdate<T>(stream, count, A + static_cast<size_t>(k + 1) * lda + rk,
                                static_cast<size_t>(lda), vn1 + k + 1, vn2 + k + 1, flags + k + 1,
                                tol3z);
    }
  }

  const int kbb = k;
  *kb = kbb;
  const int rk = offset + kbb; // first row/col below the factored block

  // --- 7. One deferred gemm applies the whole block to the rest of the trailing
  // matrix: A(rk:m, kb:n) -= A(rk:m, 0:kb) * F(kb:n, 0:kb)^T.
  // op(A) = N (rows x kb), op(F) = T (kb x cols), C = A(rk, kb) (lda).
  const int grows = m - rk;
  const int gcols = n - kbb;
  if (kbb < mn && grows > 0 && gcols > 0) {
    CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, grows, gcols, kbb, &neg_one,
                         A + static_cast<size_t>(rk), lda, F + kbb, ldf, &one,
                         A + static_cast<size_t>(kbb) * lda + rk, lda));
  }

  // --- 8. Recompute the degraded columns exactly, now that the deferred gemm has
  // brought the trailing block up to date. Read the device mask once, COMPACT the
  // flagged global column indices (j in [kb, n)), and for each run one wwr::nrm2
  // over its sub-diagonal trailing part A(rk:m, j); reset vn1[j] = vn2[j] = norm.
  // (Columns 0:kb are R's factored columns; only the still-trailing columns can
  // be flagged, since the downdate only ever touches vn1[k+1:n].)
  std::vector<int> host_flags(static_cast<size_t>(n));
  CLM_TRY(wwr::wwrMemcpyAsync(host_flags.data(), flags, static_cast<size_t>(n) * sizeof(int),
                              wwr::wwrMemcpyDeviceToHost, stream));
  CLM_TRY(wwr::wwrStreamSynchronize(stream));
  std::vector<int> degraded; // compacted flagged column indices
  for (int j = kbb; j < n; ++j) {
    if (host_flags[static_cast<size_t>(j)] != 0) {
      degraded.push_back(j);
    }
  }
  if (grows > 0) {
    for (const int j : degraded) {
      T norm = zero;
      CLM_TRY(wwr::nrm2<T>(handle, grows, A + static_cast<size_t>(j) * lda + rk, 1, &norm));
      CLM_TRY(laqps_detail::set_device_scalar<T>(stream, vn1 + j, norm));
      CLM_TRY(laqps_detail::set_device_scalar<T>(stream, vn2 + j, norm));
    }
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
