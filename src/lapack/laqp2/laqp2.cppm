/**
 * @file laqp2.cppm
 * @brief The calaman.laqp2 module -- the unblocked, level-2 Businger-Golub
 *        pivoted-QR panel, LAPACK's ?laqp2
 *
 * Factors an m-by-n column-major matrix with column pivoting into A*P = Q*R,
 * overwriting A the standard LAPACK way: R in the upper trapezoid, the j-th
 * reflector's tail below the diagonal of column j, the Householder scalars in
 * @p tau and the column permutation in @p jpvt. This is the panel the blocked
 * geqp3 driver (#12) hands single columns at a time; here it is the whole
 * matrix as one panel. Templated over `float` and `double`.
 *
 * It is a HOST COMPOSITION, not a kernel: each step reaches for larfg (the
 * reflector), larf (its application to the trailing block) and the wrapped
 * BLAS-1 iamax / swap / nrm2, exactly the outermost-layer discipline larfg and
 * larf follow. The one piece that is genuinely elementwise-over-columns device
 * work -- the LAWN 176 partial-norm downdate -- is launched through
 * wwr.extension.parallel_for from laqp2.cu (declared in laqp2_bridge.h), the
 * same module/.cu split calaman.lacpy uses.
 *
 * Per step i (0-based, i = 0 .. min(m - offset, n) - 1), with offs = offset + i
 * the pivot row:
 *   1. Pivot. iamax over the partial-norm tail vn1[i:n] picks the fullest
 *      remaining column; if it is not column i, swap columns i and the pivot in
 *      A (a full m-length wwr::swap), swap their vn1/vn2 entries, and record the
 *      permutation in jpvt.
 *   2. Reflector. larfg on A(offs:m, i) writes tau[i], overwrites A(offs,i) with
 *      beta, and scales the tail into v_tail = A(offs+1:m, i).
 *   3. Apply left. Set A(offs,i) = 1 so v = &A(offs,i) is the full reflector with
 *      v[0] == 1 (the standard LAPACK trick), larf(Side::L) it onto the trailing
 *      block A(offs:m, i+1:n), then restore A(offs,i) = beta.
 *   4. Downdate. For each trailing column j > i: cheaply shrink vn1[j] by the
 *      Businger-Golub factor, or -- when the running estimate has lost too much
 *      precision relative to the original vn2[j] -- recompute vn1[j] exactly as
 *      the nrm2 of the now-updated trailing column and reset vn2[j]. The kernel
 *      does the cheap path and flags the rest; the host issues the exact norms.
 *
 * Allocation-free shipped surface (CLAUDE.md, test/shared/README.md): vn1, vn2
 * and work are caller-provided device pointers, like LAPACK's ?laqp2 WORK and
 * calaman.diff_norm / larf. jpvt is a HOST int array -- pivot selection already
 * needs the index host-side each step, so there is no device permutation to
 * keep. laqp2 fills jpvt with the final 1-based permutation (jpvt[k] = original
 * index of the column now in position k), initialising it internally; this is
 * the permutation LAPACKE_?geqp3 returns for an all-free entry, so the oracle
 * agrees.
 *
 * offset is carried as LAPACK carries it, for the blocked driver's benefit, but
 * is only required correct for offset == 0: the partial norms vn1/vn2 are taken
 * as already describing A(offset:m, :), and the first reflected row the downdate
 * reads is row offset.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like larfg/larf: iamax's
 * index, nrm2's norm and the larfg/larf host scalars are all host-side and the
 * reductions block. A per-step device->host sync of the pivot index and the
 * recompute flags is inherent to pivoting at this granularity, exactly as larfg
 * already reads alpha back each call.
 *
 * Complex ?laqp2 is a deliberate later extension, for the same reasons larfg and
 * diff_norm document: a complex reflector and the conjugated row the downdate
 * would read differ materially from a trivial instantiation.
 *
 * Usage:
 *   import calaman.laqp2;
 *   import wwr.blas;          // wwrblasHandle_t, wwrblasCreate
 *   // d_A: m x n device matrix, lda; d_tau: length min(m,n);
 *   // d_vn1, d_vn2: length n (initial column norms); d_work: length n;
 *   // jpvt: host int[n]
 *   calaman::laqp2<double>(handle, m, n, 0, d_A, lda, jpvt, d_tau, d_vn1, d_vn2,
 *                          d_work);
 */

module;

#include "laqp2_bridge.h"
// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.laqp2;

import wwr.blas;          // wwrblasHandle_t, wwrblasStatus_t, WWRBLAS_STATUS_*
import wwr.runtime_api;   // wwrMemcpy(Async), wwrStreamSynchronize, wwrSuccess
import wwr.wrappers.blas; // iamax, swap, nrm2
import calaman.larfg;     // calaman::larfg
import calaman.larf;      // calaman::larf, calaman::Side
import std;               // std::sqrt, std::min, std::vector

// export import, not a plain import: laqp2 RETURNS calaman::Status, so a consumer
// of `import calaman.laqp2;` must see Status's member functions, not just its
// name -- the same re-export diff_norm does.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// laqp2_detail, not an anonymous namespace: this helper is named by the exported
// laqp2 template's body, which is instantiated in every importer's TU. An
// anonymous namespace would give it internal linkage, invisible there; a named
// (unexported) namespace gives it module linkage, reachable by the
// instantiation yet absent from the module's public surface. The namespace is
// MODULE-SPECIFIC (laqp2_detail, not a bare detail) because calaman.laqps
// defines an identically-named swap_device_scalar: once the two live in separate
// modules, a consumer that imports both (calaman.geqp3) would see one qualified
// name attached to two modules -- ill-formed -- unless the namespaces differ.
namespace laqp2_detail {

/// @brief Swap two single device scalars through a host staging pair, on @p stream
///
/// Pivot bookkeeping swaps vn1[i]/vn1[pvt] (and vn2) a scalar at a time; there is
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

} // namespace laqp2_detail

/// @brief Factor A with Businger-Golub column pivoting (LAPACK ?laqp2)
///
/// Produces A*P = Q*R in place: R in A's upper trapezoid, reflector j below the
/// diagonal of column j, tau[j] the j-th scalar, jpvt the column permutation.
/// Pivoting, reflector and trailing update run as a host loop over the wrapped
/// BLAS plus larfg/larf; the partial-norm downdate runs on the device.
///
/// Short-circuits: the first failing step's Status is returned and the
/// factorization stops there. Returns success and writes nothing when the panel
/// is empty. Each device read/write now returns its OWN domain's Status (the
/// copies a runtime one) rather than masquerading as a BLAS code.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; A, tau, vn1, vn2, work live on its device
/// @param m Row count of A
/// @param n Column count of A
/// @param offset Rows already factored above this panel; vn1/vn2 describe A(offset:m, :)
/// @param A Device matrix, m by n, column-major, overwritten with R and the reflectors
/// @param lda Leading dimension of A (>= m)
/// @param jpvt Host int array, length n; filled with the 1-based column permutation
/// @param tau Device array, length >= min(m - offset, n); the reflector scalars
/// @param vn1 Device array, length n; running partial column norms (updated)
/// @param vn2 Device array, length n; original partial column norms (updated on recompute)
/// @param work Device workspace, length >= n; larf's intermediate per step
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status laqp2(wwr::wwrblasHandle_t handle, const int m, const int n, const int offset, T *A,
             const int lda, int *jpvt, T *tau, T *vn1, T *vn2, T *work) {
  if (m <= 0 || n <= 0) {
    return wwr::WWRBLAS_STATUS_SUCCESS;
  }

  // Order every scalar read/write on the handle's own stream, so they follow the
  // caller's uploads and this routine's BLAS work -- the larfg discipline.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // jpvt starts as the identity permutation (1-based): position k holds original
  // column k. Each pivot swaps two entries, so the final jpvt is the permutation
  // geqp3 reports.
  for (int j = 0; j < n; ++j) {
    jpvt[j] = j + 1;
  }

  // tol3z = sqrt(eps): the LAWN 176 relative-precision threshold the downdate
  // tests the accumulated shrink against.
  const T tol3z = std::sqrt(std::numeric_limits<T>::epsilon());

  const int mn = std::min(m - offset, n);
  std::vector<int> flags; // host mirror of the per-step recompute flags

  for (int i = 0; i < mn; ++i) {
    const int offs = offset + i; // pivot row in A
    const int rows = m - offs;   // reflector length / trailing-block rows

    // --- 1. Pivot: iamax over vn1[i:n] (host pointer mode writes a host int). It
    // returns a 1-based index into the tail, so the global pivot column is
    // i + (idx - 1).
    const int tail = n - i;
    int idx = 1;
    if (tail > 1) {
      CLM_TRY(wwr::iamax<T, int>(handle, tail, vn1 + i, 1, &idx));
    }
    const int pvt = i + (idx - 1);

    if (pvt != i) {
      // Swap the two full m-length columns of A, then their partial-norm entries
      // and the permutation record. The vn1/vn2 swaps are single-element device
      // copies through a host staging scalar on the handle's stream.
      CLM_TRY(wwr::swap<T, int>(handle, m, A + static_cast<std::size_t>(i) * lda, 1,
                                A + static_cast<std::size_t>(pvt) * lda, 1));
      CLM_TRY(laqp2_detail::swap_device_scalar<T>(stream, vn1 + i, vn1 + pvt));
      CLM_TRY(laqp2_detail::swap_device_scalar<T>(stream, vn2 + i, vn2 + pvt));
      const int tmp = jpvt[i];
      jpvt[i] = jpvt[pvt];
      jpvt[pvt] = tmp;
    }

    // --- 2. Reflector on A(offs:m, i). larfg writes tau[i] (host) and beta, and
    // overwrites A(offs,i) with beta, A(offs+1:m, i) with the scaled tail.
    T *col = A + static_cast<std::size_t>(i) * lda + offs;
    T host_tau{};
    T beta{};
    CLM_TRY(larfg<T>(handle, rows, col, col + 1, 1, &host_tau, &beta));
    // tau lives on the device for the caller; write tau[i] back on the stream.
    CLM_TRY(wwr::wwrMemcpyAsync(tau + i, &host_tau, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
    CLM_TRY(wwr::wwrStreamSynchronize(stream));

    // --- 3. Apply the reflector to the trailing block A(offs:m, i+1:n) from the
    // left. larf needs the full reflector v with v[0] == 1, so temporarily set
    // A(offs,i) = 1, apply, then restore the diagonal to beta.
    const int trail = n - i - 1;
    if (trail > 0) {
      const T one = T{1};
      CLM_TRY(wwr::wwrMemcpyAsync(col, &one, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
      CLM_TRY(wwr::wwrStreamSynchronize(stream));

      T *trailing = A + static_cast<std::size_t>(i + 1) * lda + offs;
      CLM_TRY(larf<T>(handle, Side::L, rows, trail, col, 1, host_tau, trailing, lda, work));

      CLM_TRY(wwr::wwrMemcpyAsync(col, &beta, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
      CLM_TRY(wwr::wwrStreamSynchronize(stream));

      // --- 4. Partial-norm downdate over the trailing columns. The kernel reads
      // the reflected row A(offs, i+1:n) (stride lda) and the vn1/vn2 tails
      // vn1[i+1:n] / vn2[i+1:n], shrinks the cheap ones in place and flags the
      // rest. The flags are reused as the work int scratch -- reinterpret the
      // caller's T work buffer, which has length >= n >= trail ints when
      // sizeof(T) >= sizeof(int) (true for float/double).
      int *dev_flags = reinterpret_cast<int *>(work);
      const std::size_t count = static_cast<std::size_t>(trail);
      device::laqp2_downdate<T>(stream, count, A + static_cast<std::size_t>(i + 1) * lda + offs,
                                static_cast<std::size_t>(lda), vn1 + i + 1, vn2 + i + 1, dev_flags,
                                tol3z);

      // Read the flags back and recompute the exact norm of each flagged
      // trailing column -- nrm2 of A(offs+1:m, j) (the sub-diagonal part, since
      // row offs is now R's entry). Reset vn2[j] = vn1[j] for the recomputed one.
      flags.resize(count);
      CLM_TRY(wwr::wwrMemcpyAsync(flags.data(), dev_flags, count * sizeof(int),
                                  wwr::wwrMemcpyDeviceToHost, stream));
      CLM_TRY(wwr::wwrStreamSynchronize(stream));
      const int sub = rows - 1; // rows below the diagonal in a trailing column
      for (std::size_t k = 0; k < count; ++k) {
        if (flags[k] == 0) {
          continue;
        }
        const int j = i + 1 + static_cast<int>(k);
        T norm = T{0};
        if (sub > 0) {
          CLM_TRY(wwr::nrm2<T>(handle, sub, A + static_cast<std::size_t>(j) * lda + offs + 1, 1,
                               &norm));
        }
        // Write the exact norm into vn1[j] and vn2[j] on the stream.
        CLM_TRY(wwr::wwrMemcpyAsync(vn1 + j, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
        CLM_TRY(wwr::wwrMemcpyAsync(vn2 + j, &norm, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
        CLM_TRY(wwr::wwrStreamSynchronize(stream));
      }
    }
  }

  return wwr::WWRBLAS_STATUS_SUCCESS;
}

} // namespace calaman
