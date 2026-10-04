/**
 * @file gehrd.cppm
 * @brief The calaman.gehrd module -- blocked reduction of a general matrix to
 *        upper Hessenberg form, LAPACK's ?gehrd
 *
 * Reduces a real n-by-n column-major matrix A to upper Hessenberg form H by an
 * orthogonal similarity Q^T A Q = H, overwriting A the standard LAPACK way: H in
 * the upper triangle and first subdiagonal, reflector i's tail below the
 * subdiagonal of column i, and the Householder scalars in @p tau. A is assumed
 * already upper triangular outside the window [ilo, ihi] (normally set by a prior
 * calaman.gebal; else 1 and n). Templated over `float` and `double`.
 *
 * The blocked driver over the unblocked calaman.gehd2: each step reduces a panel
 * of up to nb columns with calaman.lahr2 (yielding V, the triangular factor T and
 * the auxiliary Y = A V T), updates the trailing block A(1:ihi, i+ib:ihi) from
 * the right with one Y gemm, then applies the block reflector to A(i+1:ihi,
 * i+ib:n) from the left with calaman.larfb. The remaining tail -- too small to
 * block -- falls to calaman.gehd2. ?gehrd's INFO contract (zeros tau outside
 * [ilo-1, ihi-1]) is honoured; ?gehd2 does not, so the driver zeroes it.
 *
 * Allocation-free shipped surface (CLAUDE.md): A, @p tau and @p work are
 * caller-provided device pointers. @p tau has length >= n-1; @p work length
 * >= n*nb + nb*nb (the Y panel reused as the larfb workspace, plus the T factor),
 * with nb the internal block size below. ilo/ihi are 1-based, the convention
 * calaman.gebal reports, so the gebal -> gehrd pipeline forwards them unchanged.
 *
 * Requires the handle's DEFAULT (host) pointer mode, like gehd2/lahr2/larfb.
 * Complex ?gehrd is a deliberate later extension, for the reason larfg documents.
 *
 * Usage:
 *   import calaman.gehrd;
 *   import wwr.blas;   // wwrblasHandle_t, wwrblasCreate
 *   // d_A: n x n device matrix, lda; d_tau: length n-1; d_work: length n*nb+nb*nb
 *   calaman::gehrd<double>(handle, n, 1, n, d_A, lda, d_tau, d_work);
 */

module;

// CLM_TRY -- a macro, so it arrives by #include in the global module fragment,
// not by import. Resolved root-relative via the src/ root calaman.error_handling
// exports; needs calaman::Status visible at expansion, which the import below
// (export import) supplies.
#include "error_handling/error_macros.h"

export module calaman.gehrd;

import wwr.blas;          // wwrblasHandle_t, wwrblasGetStream, WWRBLAS_OP_*, status
import wwr.runtime_api;   // wwrStream_t, wwrMemcpyAsync, wwrMemsetAsync, wwrStreamSynchronize
import wwr.wrappers.blas; // gemm, trmm, axpy -- the trailing and panel Y updates
import calaman.common;    // Side, Trans, Direct, StoreV (the larfb selectors)
import calaman.gehd2;     // calaman::gehd2 -- the unblocked tail
import calaman.lahr2;     // calaman::lahr2 -- the per-panel V/T/Y reduction
import calaman.larfb;     // calaman::larfb -- apply the block reflector from the left
import std;               // std::min, std::max, std::size_t

// export import, not a plain import: gehrd RETURNS calaman::Status, so a consumer
// of `import calaman.gehrd;` must see Status's member functions, not just its
// name -- the same re-export gehd2/lahr2/larfb do.
export import calaman.error_handling; // Status -- the cross-domain return type

namespace calaman {

// Internal block size for the level-3 path (the reference's ilaenv NB). Small on
// purpose: large enough to turn each panel's trailing update into a real level-3
// gemm/larfb, small enough that modest test shapes span several blocks and a
// non-multiple of nb. A different nb from the reference's choice changes only the
// floating-point accumulation order, not the unique Hessenberg factor.
inline constexpr int kGehrdBlockSize = 32;

/// @brief Reduce A to upper Hessenberg form by an orthogonal similarity (?gehrd)
///
/// Computes Q^T*A*Q = H in place: H in A's upper triangle and first subdiagonal,
/// reflector i's tail below the subdiagonal of column i, tau[i-1] its scalar.
/// A is assumed upper triangular in rows/columns 1:ilo-1 and ihi+1:n. Runs the
/// blocked lahr2 -> Y gemm -> larfb panel loop, then the unblocked calaman.gehd2
/// on the tail; a failing step short-circuits and returns its Status. Validates
/// arguments as LAPACK's INFO contract does, returning wwrErrorInvalidValue on the
/// first illegal one. Zeros tau[0:ilo-1] and tau[ihi-1:n-1] as ?gehrd specifies.
///
/// @tparam T Element type; one of the instantiated types (float, double)
/// @param handle GPU BLAS handle in host pointer mode; A, tau, work live on its device
/// @param n Order of the matrix A (>= 0)
/// @param ilo 1-based first index of the active window; 1 <= ilo <= max(1,n)
/// @param ihi 1-based last index of it; min(ilo,n) <= ihi <= n
/// @param A Device matrix, n by n, column-major, overwritten with H and the reflectors
/// @param lda Leading dimension of A (>= max(1,n))
/// @param tau Device array, length >= n-1; the reflector scalars (rest zeroed)
/// @param work Device workspace, length >= n*nb + nb*nb (the Y panel + the T factor)
/// @return The Status of the failing step, otherwise WWRBLAS_STATUS_SUCCESS
export template<typename T>
Status gehrd(wwr::wwrblasHandle_t handle, const int n, const int ilo, const int ihi, T *A,
             const int lda, T *tau, T *work) {
  // Argument validation, in LAPACK's INFO order (-1, -2, -3, -5).
  if (n < 0 || ilo < 1 || ilo > std::max(1, n) || ihi < std::min(ilo, n) || ihi > n ||
      lda < std::max(1, n)) {
    return wwr::wwrErrorInvalidValue;
  }

  using std::size_t;

  // Order the tau zeroing on the handle's own stream, after the caller's uploads.
  wwr::wwrStream_t stream{};
  CLM_TRY(wwr::wwrblasGetStream(handle, &stream));

  // Set tau(1:ilo-1) and tau(ihi:n-1) to zero (1-based tau(ilo:ihi-1) are the
  // reflector scalars gehd2/lahr2 write) -- the reference's INFO-contract head
  // and tail clear. gehd2 does not do this; the blocked driver must.
  if (ilo > 1) {
    CLM_TRY(wwr::wwrMemsetAsync(tau, 0, static_cast<size_t>(ilo - 1) * sizeof(T), stream));
  }
  if (ihi < n) {
    CLM_TRY(wwr::wwrMemsetAsync(tau + (ihi - 1), 0, static_cast<size_t>(n - ihi) * sizeof(T),
                                stream));
  }
  CLM_TRY(wwr::wwrStreamSynchronize(stream));

  const int nh = ihi - ilo; // number of reflectors H(ilo) .. H(ihi-1)
  constexpr int nb = kGehrdBlockSize;

  // i is the reference's 1-based loop column I; it also carries the value the
  // final unblocked gehd2 starts from. Unblocked whole-window path: a block that
  // spans (or exceeds) the window reduces it all through gehd2 in one shot.
  int i = ilo;

  if (nb < nh) {
    // Blocked path. The crossover nx: stop blocking once the remaining window is
    // no larger than one block, handing that tail to gehd2. nx == nb mirrors the
    // common ilaenv default (crossover == block size); its exact value changes
    // only the rounding, never the factor.
    constexpr int nx = nb;
    const int ldwork = std::max(1, n); // Y panel (n x nb) and the larfb workspace
    T *const Y = work;                 // n x nb, ld == ldwork
    T *const Tf = work + static_cast<size_t>(ldwork) * nb; // nb x nb, ld == nb

    // DO I = ILO, IHI-1-NX, NB.
    for (i = ilo; i <= ihi - 1 - nx; i += nb) {
      const int ib = std::min(nb, ihi - i);

      // Reduce the panel A(:, i:i+ib-1): V in A below the subdiagonal, scalars in
      // tau(i:i+ib-1), the triangular factor in Tf and Y = A V T. lahr2's k is
      // the 1-based panel column i (the reference's K), matching its oracle.
      CLM_TRY(lahr2<T>(handle, ihi, i, ib, A + static_cast<size_t>(i - 1) * lda, lda,
                       tau + (i - 1), Tf, nb, Y, ldwork));

      // Update the trailing block A(1:ihi, i+ib:ihi) from the right with Y:
      // A(1:ihi, i+ib:ihi) -= Y * A(i+ib:ihi, i:i+ib-1)^T. The reference sets the
      // reflector's unit entry A(i+ib, i+ib-1) = 1 for the gemm, then restores the
      // beta it overwrote. ei holds beta; the single-element swaps ride the stream.
      const int trail_cols = ihi - i - ib + 1; // IHI-I-IB+1
      if (trail_cols > 0) {
        T *const unit = A + static_cast<size_t>(i + ib - 2) * lda + (i + ib - 1);
        T ei{};
        CLM_TRY(wwr::wwrMemcpyAsync(&ei, unit, sizeof(T), wwr::wwrMemcpyDeviceToHost, stream));
        CLM_TRY(wwr::wwrStreamSynchronize(stream));
        const T one = T{1};
        CLM_TRY(wwr::wwrMemcpyAsync(unit, &one, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
        CLM_TRY(wwr::wwrStreamSynchronize(stream));

        const T neg_one = T{-1};
        CLM_TRY(wwr::gemm<T>(handle, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, ihi, trail_cols, ib,
                             &neg_one, Y, ldwork, A + static_cast<size_t>(i - 1) * lda + (i + ib - 1),
                             lda, &one, A + static_cast<size_t>(i + ib - 1) * lda, lda));

        CLM_TRY(wwr::wwrMemcpyAsync(unit, &ei, sizeof(T), wwr::wwrMemcpyHostToDevice, stream));
        CLM_TRY(wwr::wwrStreamSynchronize(stream));
      }

      // Apply the block reflector H to A(1:i, i+1:i+ib-1) from the right -- the
      // panel's own top i rows, which lahr2 leaves un-updated. Reuse the Y buffer:
      // W := Y(1:i, 1:ib-1) * V1^T (V1 the unit-lower reflector heads A(i+1:.., i)),
      // then A(1:i, i+j+1) -= W(:,j) for each panel column. Skipped when ib == 1.
      if (ib > 1) {
        const T one = T{1};
        const T neg_one = T{-1};
        CLM_TRY(wwr::trmm<T>(handle, wwr::WWRBLAS_SIDE_RIGHT, wwr::WWRBLAS_FILL_MODE_LOWER,
                             wwr::WWRBLAS_OP_T, wwr::WWRBLAS_DIAG_UNIT, i, ib - 1, &one,
                             A + static_cast<size_t>(i - 1) * lda + i, lda, Y, ldwork, Y, ldwork));
        for (int j = 0; j < ib - 1; ++j) {
          CLM_TRY(wwr::axpy<T>(handle, i, &neg_one, Y + static_cast<size_t>(j) * ldwork, 1,
                               A + static_cast<size_t>(i + j) * lda, 1));
        }
      }

      // Apply the block reflector H to A(i+1:ihi, i+ib:n) from the left. V lives
      // in A(i+1:ihi, i:i+ib-1) (columnwise, forward); Tf is its factor; the Y
      // buffer doubles as the larfb workspace (its n*nb cells hold W). Skipped
      // when there are no trailing columns.
      const int larfb_cols = n - i - ib + 1; // N-I-IB+1
      if (larfb_cols > 0) {
        CLM_TRY(larfb<T>(handle, Side::L, Trans::T, Direct::F, StoreV::C, ihi - i, larfb_cols, ib,
                         A + static_cast<size_t>(i - 1) * lda + i, lda, Tf, nb,
                         A + static_cast<size_t>(i + ib - 1) * lda + i, lda, Y, ldwork));
      }
    }
    // After the Fortran DO loop, i holds the first value exceeding ihi-1-nx (or
    // ilo if the body never ran) -- the start column for the unblocked tail.
  }

  // Unblocked reduction of the rest of the window: gehd2(n, i, ihi, ...). For the
  // whole-window unblocked path i == ilo, so this reduces everything.
  return gehd2<T>(handle, n, i, ihi, A, lda, tau, work);
}

} // namespace calaman
